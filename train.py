import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
import argparse
from tqdm import tqdm
from dataset import MNIST_TE_Dataset
from vae import VAE
from autoencoder import Autoencoder
from torch.utils.data import DataLoader
from metrics import compute_metrics
from utils import sample_images, save_dynamics, predict_attributes
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to where npz files of dataset are stored')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to where you want to save experiment')
    parser.add_argument('--device', type=int, default=0, help='GPU device')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--refinement_epochs', type=int, default=1000, help='Epochs during which mask is fixed and no L1 reg')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--timepoints', type=int, default= 10, help='Number of timepoints')
    parser.add_argument('--delta_t', type=float, default= .05, help='Change in t between timepoints')
    parser.add_argument('--pred_timepoints', type=int, default=5, help='Number of timepoints to predict in the future')
    parser.add_argument('--input_dim', type=int, default=32, help='Input dimension')
    parser.add_argument('--latent_ch', type=int, default=256, help='Latent Channels of Encoder')
    parser.add_argument('--latent_dim', type=int, default=3, help='Latent dimension')
    parser.add_argument('--library_dim', type=int, default=7, help='Number of candidate functions in the library')
    parser.add_argument('--beta', type=float, default=.5, help='Beta term in ELBO')
    parser.add_argument("--viz_freq", help="Steps per visualisation.", type=int, default=10000)
    # Loss weights'
    parser.add_argument('--lambda_1', type=float, default=.005, help='Weight on x-level derivative loss')
    parser.add_argument('--lambda_2', type=float, default=.001, help='Weight on z-level derivative loss')
    parser.add_argument('--lambda_3', type=float, default= .00001, help='Weight on regularization loss')
    parser.add_argument('--lambda_4', type=float, default= .0001, help='Weight on supervised loss' )

    # Evaluation and checkpoints
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency of evaluation')
    parser.add_argument('--best_loss', type=float, default=float('inf'), help='Initial best loss')

    return parser.parse_args()


def setup_dataloader(args):
    train_file = os.path.join(args.dataset_dir, 'mnist_train_traj.npz')
    val_file = os.path.join(args.dataset_dir, 'mnist_val_traj.npz')
    test_file = os.path.join(args.dataset_dir, 'mnist_test_traj.npz')

    # Create datasets
    train_dataset = MNIST_TE_Dataset(train_file)

    val_dataset = MNIST_TE_Dataset(val_file)
    test_dataset = MNIST_TE_Dataset(test_file)

    # Create DataLoaders
    args.bs = args.timepoints * 40 #TODO: make not hardcoded - done so samples from a single trajecotry are in order in one batch
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    return train_loader, val_loader, test_loader


def setup_logging(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]

    # Configure logger
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt")),
            logging.StreamHandler(),
        ],
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger("TrainingLogger")

def run_epoch(args, model, loader, optimizer=None, training=True):
    """Run a single epoch for training or validation."""
    mode = "Train" if training else "Validation"
    model.train(training)
    epoch_loss = 0
    individual_losses = {
        'potential_outcome_loss': 0.0,
        'treatment_loss': 0.0,
        'kl': 0.0,
        'nll': 0.0,
        'elbo': 0.0,
        'sindy_x_loss': 0.0,
        'sindy_v_loss': 0.0,
        'sindy_l1_loss': 0.0,
        'count': 0
    }

    # Iterate over the data loader
    progress_bar = tqdm(loader, desc=f"{mode} Epoch", total=len(loader), dynamic_ncols=True)

    for i, batch in enumerate(progress_bar):

        args.iter = i + 1 + (args.epoch) * len(loader)
   
        # Prepare data     
        x = batch["true_image"]
        x_dot = batch["true_derivative"] 
  
        treatment = batch["true_treatment"]
        size = batch["true_size"]

        # Forward pass
        losses = model(x.to(args.device), x_dot.to(args.device), treatment.to(args.device), size.to(args.device))
        
        # Accumulate individual losses
        individual_losses['potential_outcome_loss'] += losses['potential_outcome_loss'].item()
        individual_losses['treatment_loss'] += losses['treatment_loss'].item()
        individual_losses['kl'] += losses['kl'].item()
        individual_losses['nll'] += losses['nll'].item()
        individual_losses['elbo'] += losses['elbo'].item()
        individual_losses['sindy_x_loss'] += losses['sindy_x_loss'].item()
        individual_losses['sindy_v_loss'] += losses['sindy_v_loss'].item()
        individual_losses['sindy_l1_loss'] += losses['sindy_l1_loss'].item()
        individual_losses['count'] += 1

        # Compute total loss
        total_loss = (
            losses['nll'] +  # Negative log likelihood (reconstruction loss)
            (losses['treatment_loss'] + losses['potential_outcome_loss']) * args.lambda_4 +  # Supervised losses scaled by lambda_4
            losses['sindy_x_loss'] * args.lambda_1 +  # Regularization on input space (encoder) scaled by lambda_1
            losses['sindy_v_loss'] * args.lambda_2 +  # Regularization on latent space (latent z) scaled by lambda_2
            losses['sindy_l1_loss'] * args.lambda_3 +  # L1 regularization on latent space scaled by lambda_3
            losses['kl'] * args.beta  # KL divergence scaled by lambda_kl (assumed as an additional weighting factor)
        ).float()
        if training:
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Sum up the total loss
        epoch_loss += total_loss.item()

        progress_bar.set_postfix({
            "Loss": total_loss.item(),
            "NLL": losses['nll'].item(),
            "KL": losses['kl'].item(),
            "PO Loss": losses['potential_outcome_loss'].item(),
            "Treat Loss": losses['treatment_loss'].item(),
            "SINDY X": losses['sindy_x_loss'].item(),
            "SINDY V": losses['sindy_v_loss'].item(),
            "SINDY L1": losses['sindy_l1_loss'].item()
        })

        if args.iter % args.viz_freq == 0 or (args.iter in args.early_evals):
            with torch.enable_grad():
                model.eval()
                sample_images(args, model, batch)
                save_dynamics(args, model, batch)
                predict_attributes(args, model, batch)
            if training: 
                model.train()
           
    progress_bar.close()

    if (args.epoch + 1) % 500 == 0:
        model.sindy_layer.update_mask()
    if not training and (args.epoch + 1) % 100 == 0:
        plot_comparison(args, x, x_hat)
        save_dynamics(args, model.eval(), batch)
    
    # Ensure there is a count to avoid division by zero
    if individual_losses['count'] > 0:
        individual_losses['nll'] /= individual_losses['count']
        individual_losses['kl'] /= individual_losses['count']
        individual_losses['potential_outcome_loss'] /= individual_losses['count']
        individual_losses['treatment_loss'] /= individual_losses['count']
        individual_losses['sindy_x_loss'] /= individual_losses['count']
        individual_losses['sindy_v_loss'] /= individual_losses['count']
        individual_losses['sindy_l1_loss'] /= individual_losses['count']
    else:
        print("Warning: Division by zero encountered. Losses cannot be averaged as count is zero.")
    
    return epoch_loss / len(loader), individual_losses 

def train(args, model, train_loader, val_loader, optimizer, logger):
    logger.info("Starting regular training phase (with L1 Reg.)...")
    
    args.iter = 0 
    args.early_evals = set([args.iter + 1] + [args.iter + 2**n for n in range(3, 14)])

    for epoch in range(args.epochs):
        args.epoch = epoch
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Training phase
        train_loss, train_individual_losses = run_epoch(args, model, train_loader, optimizer, training=True)
        
        # Log total training loss
        logger.info(f"[Train] Epoch {epoch + 1}/{args.epochs} - Total Loss: {train_loss:.4f}")

        # Log averaged individual losses for training
        logger.info(
            f"[Train] Average Losses - "
            f"NLL: {train_individual_losses.get('nll', 0.0):.4f}, "
            f"KL: {train_individual_losses.get('kl', 0.0):.4f}, "
            f"PO Loss: {train_individual_losses.get('potential_outcome_loss', 0.0):.4f}, "
            f"Treatment Loss: {train_individual_losses.get('treatment_loss', 0.0):.4f}, "
            f"SINDY X Loss: {train_individual_losses.get('sindy_x_loss', 0.0):.4f}, "
            f"SINDY V Loss: {train_individual_losses.get('sindy_v_loss', 0.0):.4f}, "
            f"SINDY L1 Loss: {train_individual_losses.get('sindy_l1_loss', 0.0):.4f}"
        )

        # Validation phase
        if (epoch + 1) % args.eval_freq == 0:
            val_loss, val_individual_losses = run_epoch(args, model, val_loader, training=False)
            logger.info(f"[Validation] Epoch {epoch + 1}/{args.epochs} - Total Loss: {val_loss:.4f}")

            # Log averaged individual losses for validation
            logger.info(
                f"[Validation] Average Losses - "
                f"NLL: {val_individual_losses.get('nll', 0.0):.4f}, "
                f"KL: {val_individual_losses.get('kl', 0.0):.4f}, "
                f"Potential Outcome Loss: {val_individual_losses.get('potential_outcome_loss', 0.0):.4f}, "
                f"Treatment Loss: {val_individual_losses.get('treatment_loss', 0.0):.4f}, "
                f"SINDY X Loss: {val_individual_losses.get('sindy_x_loss', 0.0):.4f}, "
                f"SINDY V Loss: {val_individual_losses.get('sindy_v_loss', 0.0):.4f}, "
                f"SINDY L1 Loss: {val_individual_losses.get('sindy_l1_loss', 0.0):.4f}"
            )

            # Save the best model
            if val_loss < args.best_loss:
                args.best_loss = val_loss
                save_dict = {
                    "epoch": epoch,
                    "step": epoch * len(train_loader),
                    "best_loss": args.best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hparams": vars(args),
                }
                ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
                torch.save(save_dict, ckpt_path)
                logger.info(f"Model saved: {ckpt_path}")

    sparsity_mask = (model.sindy_layer.coefficients.data.abs() > 0).float()
    logger.info("Sparsity pattern Υ determined.")
    return sparsity_mask

def main():
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args)
    logger.info(f"Arguments: {vars(args)}")

    # Setup DataLoader
    train_loader, val_loader, _ = setup_dataloader(args)

    # Initialize model and optimizer
    model = VAE(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Model initialized.")

    # Train and refine
    sparsity_mask = train(args, model, train_loader, val_loader, optimizer, logger)
    logger.info("Training complete.")

    logger.info("Starting refinement phase...")
    refine(args, model, train_loader, val_loader, optimizer, logger, sparsity_mask)
    logger.info("Refinement complete.")


if __name__ == "__main__":
    main()
