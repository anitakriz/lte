import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
import argparse
from tqdm import tqdm
from dataset import  MNIST_UNTREATED_Dataset, MNIST_TE_Dataset, PendulumDataset, Dot_Dataset
from hps import parse_args
from vae import VAE
from autoencoder import Autoencoder
# from autoencoder_conv import ConvAutoencoder
from torch.utils.data import DataLoader
from metrics import compute_metrics
from utils import sample_images, save_dynamics, predict_attributes, plot_recon
import torch
from scipy.special import binom
from torch.utils.tensorboard import SummaryWriter


def library_size(n, poly_order, ode_order, include_sine = False, include_constant = True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom((n*ode_order) + k - 1,k))
    if include_sine:
        l += n * ode_order
    if not include_constant:
        l -= 1
    return l

def setup_dataloader(args):
    
    if args.dataset == 'MNIST_UNTREATED':
        train_file = os.path.join(args.dataset_dir, 'mnist_train_traj.npz')
        val_file = os.path.join(args.dataset_dir, 'mnist_val_traj.npz')
        test_file = os.path.join(args.dataset_dir, 'mnist_test_traj.npz')

        # Create datasets
        train_dataset = MNIST_UNTREATED_Dataset(train_file)
        val_dataset = MNIST_UNTREATED_Dataset(val_file)
        test_dataset = MNIST_UNTREATED_Dataset(test_file)

    elif args.dataset == 'MNIST_TE':
        train_file = os.path.join(args.dataset_dir, 'mnist_train_traj.npz')
        val_file = os.path.join(args.dataset_dir, 'mnist_val_traj.npz')
        test_file = os.path.join(args.dataset_dir, 'mnist_test_traj.npz')

        # Create datasets
        train_dataset = MNIST_TE_Dataset(train_file)
        val_dataset = MNIST_TE_Dataset(val_file)
        test_dataset = MNIST_TE_Dataset(test_file)

    if args.dataset == 'DOT':
        train_file = os.path.join(args.dataset_dir, 'dot_train_traj.npz')
        val_file = os.path.join(args.dataset_dir, 'dot_val_traj.npz')
        test_file = os.path.join(args.dataset_dir, 'dot_test_traj.npz')

        # Create datasets
        train_dataset = Dot_Dataset(train_file)
        val_dataset = Dot_Dataset(val_file)
        test_dataset = Dot_Dataset(test_file)

    elif args.dataset == 'Pendulum':

        train_file = os.path.join(args.dataset_dir, 'pendulum_train.npz')
        val_file = os.path.join(args.dataset_dir, 'pendulum_val.npz')
        test_file = os.path.join(args.dataset_dir, 'pendulum_test.npz')

        # Create datasets
        train_dataset = PendulumDataset(train_file)
        val_dataset = PendulumDataset(val_file)
        test_dataset = PendulumDataset(test_file)

    # Create DataLoaders
    # args.bs = args.timepoints * 40 #TODO: make not hardcoded - done so samples from a single trajecotry are in order in one batch
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle = True)

    return train_loader, val_loader, test_loader


def setup_logging(args):
    os.makedirs(args.save_exp, exist_ok=True)

    # Reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]

    # Configure logger
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_exp, "trainlog.txt")),
            logging.StreamHandler(),
        ],
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger("TrainingLogger")

def run_epoch(args, model, loader, optimizer=None, training=True, refinement = False):

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
        'recon_loss': 0.0,
        'sindy_x_loss': 0.0,
        'sindy_z_loss': 0.0,
        'sindy_l1_loss': 0.0,
        'static_loss': 0.0,
        'count': 0
    }

    # Iterate over the data loader
    progress_bar = tqdm(loader, desc=f"{mode} Epoch", total=len(loader), dynamic_ncols=True)

    for i, batch in enumerate(progress_bar):    
        
        if args.dataset == 'MNIST_UNTREATED' or args.dataset == 'DOT':
            losses = model(x = batch['x'].to(args.device), dx = batch['dx'].to(args.device), size = batch['size'].to(args.device))
        elif args.dataset == 'Pendulum':   
            losses = model(x = batch['x'].to(args.device), dx = batch['dx'].to(args.device), ddx = batch['ddx'].to(args.device))

        # Common losses
        if args.sindy_enabled:
            for key in ['sindy_x_loss', 'sindy_z_loss']:
                individual_losses[key] += losses[key].item()
            # Refinement
            if not refinement:
                individual_losses['sindy_l1_loss'] += losses['sindy_l1_loss'].item()
        # Model-specific losses
        if isinstance(model, VAE):
            for key in ['kl', 'nll', 'elbo']:
                individual_losses[key] += losses[key].item()
        elif isinstance(model, Autoencoder):
            individual_losses['recon_loss'] += losses['recon_loss'].item()

        # Supervised losses
        if args.supervised:
            for key in ['potential_outcome_loss']:
                individual_losses[key] += losses[key].item()
        if args.static:
            individual_losses['static_loss'] += losses['static_loss'].item()

        individual_losses['count'] += 1

         # Calculate total loss
        total_loss = losses.get('recon_loss', 0.0)  # Start with reconstruction loss

        if args.sindy_enabled:
            # Add auxiliary losses
            total_loss += sum([
                losses.get('nll', 0.0),  # if VAE
                losses.get('kl', 0.0) * args.beta,  # if VAE
                losses.get('potential_outcome_loss', 0.0) * args.lambda_4,  # if supervised
                losses['sindy_x_loss'] * args.lambda_1,  # dx/dt loss
                losses['sindy_z_loss'] * args.lambda_2,  # dz/dt loss
                losses.get('sindy_l1_loss', 0.0) * args.lambda_3,  # L1 reg loss
                losses.get('static_loss', 0.0) * args.lambda_4,  # Static L1 loss
            ]).float()

        if training: 
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        epoch_loss += total_loss.item()

        progress_bar.set_postfix({
            "Loss": total_loss.item(),
            "NLL": losses.get('nll', 0.0),
            "KL": losses.get('kl', 0.0),
            "Recon Loss": losses.get('recon_loss', 0.0),
            "PO Loss": losses.get('potential_outcome_loss', 0.0),
            "Treat Loss": losses.get('treatment_loss', 0.0),
            "SINDY X": losses.get('sindy_x_loss', 0.0),
            "SINDY Z": losses.get('sindy_z_loss', 0.0),
            "SINDY L1": losses.get('sindy_l1_loss', 0.0),
            "static loss": losses.get('static_loss', 0.0)
        })
           
    progress_bar.close()

    if (args.epoch + 1) % 500 == 0 and not refinement:
        model.sindy_model.update_mask()
    if not training and (args.epoch + 1) % args.viz_freq == 0:
        if isinstance(model, VAE):
            sample_images(args, model, batch)
        if isinstance(model, Autoencoder):
            plot_recon(args, model, batch['x'].to(args.device))
        save_dynamics(args, model, batch)    

    # Average the losses
    for key in individual_losses:
        if key != 'count':
            individual_losses[key] /= individual_losses['count'] or 1  # Avoid division by zero

    return epoch_loss / len(loader), individual_losses

def train(args, model, train_loader, val_loader, optimizer, logger):
    logger.info("Starting regular training phase (with L1 Reg.)...")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir = args.save_exp)
    args.sindy_enabled = False
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
            f"Recon Loss: {train_individual_losses.get('recon_loss', 0.0):.4f}, "
            f"PO Loss: {train_individual_losses.get('potential_outcome_loss', 0.0):.4f}, "
            f"Treatment Loss: {train_individual_losses.get('treatment_loss', 0.0):.4f}, "
            f"SINDY X Loss: {train_individual_losses.get('sindy_x_loss', 0.0):.4f}, "
            f"SINDY Z Loss: {train_individual_losses.get('sindy_z_loss', 0.0):.4f}, "
            f"SINDY L1 Loss: {train_individual_losses.get('sindy_l1_loss', 0.0):.4f}",
            f"Static Loss: {train_individual_losses.get('static_loss', 0.0):.4f}"
        )

        # Log individual losses for training to TensorBoard
        writer.add_scalar('Train/Total Loss', train_loss, epoch)
        writer.add_scalar('Train/NLL Loss', train_individual_losses.get('nll', 0.0), epoch)
        writer.add_scalar('Train/KL Loss', train_individual_losses.get('kl', 0.0), epoch)
        writer.add_scalar('Train/Recon Loss', train_individual_losses.get('recon_loss', 0.0), epoch)
        writer.add_scalar('Train/PO Loss', train_individual_losses.get('potential_outcome_loss', 0.0), epoch)
        writer.add_scalar('Train/Treatment Loss', train_individual_losses.get('treatment_loss', 0.0), epoch)
        writer.add_scalar('Train/SINDY X Loss', train_individual_losses.get('sindy_x_loss', 0.0), epoch)
        writer.add_scalar('Train/SINDY Z Loss', train_individual_losses.get('sindy_z_loss', 0.0), epoch)
        writer.add_scalar('Train/SINDY L1 Loss', train_individual_losses.get('sindy_l1_loss', 0.0), epoch)
        writer.add_scalar('Train/Static Loss', train_individual_losses.get('static_loss', 0.0), epoch)


        # Validation phase
        if (epoch + 1) % args.eval_freq == 0:
            val_loss, val_individual_losses = run_epoch(args, model, val_loader, training=False)
            logger.info(f"[Validation] Epoch {epoch + 1}/{args.epochs} - Total Loss: {val_loss:.4f}")

            # Log averaged individual losses for validation
            logger.info(
                f"[Validation] Average Losses - "
                f"NLL: {val_individual_losses.get('nll', 0.0):.4f}, "
                f"KL: {val_individual_losses.get('kl', 0.0):.4f}, "
                f"Recon Loss: {val_individual_losses.get('recon_loss', 0.0):.4f}, "
                f"Potential Outcome Loss: {val_individual_losses.get('potential_outcome_loss', 0.0):.4f}, "
                f"Treatment Loss: {val_individual_losses.get('treatment_loss', 0.0):.4f}, "
                f"SINDY X Loss: {val_individual_losses.get('sindy_x_loss', 0.0):.4f}, "
                f"SINDY Z Loss: {val_individual_losses.get('sindy_z_loss', 0.0):.4f}, "
                f"SINDY L1 Loss: {val_individual_losses.get('sindy_l1_loss', 0.0):.4f}",
                f"Static Loss: {val_individual_losses.get('static', 0.0):.4f}"
            )

            # Log individual losses for validation to TensorBoard
            writer.add_scalar('Val/Total Loss', val_loss, epoch)
            writer.add_scalar('Val/NLL Loss', val_individual_losses.get('nll', 0.0), epoch)
            writer.add_scalar('Val/KL Loss', val_individual_losses.get('kl', 0.0), epoch)
            writer.add_scalar('Val/Recon Loss', val_individual_losses.get('recon_loss', 0.0), epoch)
            writer.add_scalar('Val/PO Loss', val_individual_losses.get('potential_outcome_loss', 0.0), epoch)
            writer.add_scalar('Val/Treatment Loss', val_individual_losses.get('treatment_loss', 0.0), epoch)
            writer.add_scalar('Val/SINDY X Loss', val_individual_losses.get('sindy_x_loss', 0.0), epoch)
            writer.add_scalar('Val/SINDY Z Loss', val_individual_losses.get('sindy_z_loss', 0.0), epoch)
            writer.add_scalar('Val/SINDY L1 Loss', val_individual_losses.get('sindy_l1_loss', 0.0), epoch)
            writer.add_scalar('Val/Static Loss', val_individual_losses.get('static_loss', 0.0), epoch)

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
                ckpt_path = os.path.join(args.save_exp, "checkpoint.pt")
                torch.save(save_dict, ckpt_path)
                logger.info(f"Model saved: {ckpt_path}")
        # Enable auxiliary losses after specified epochs
        if epoch >= args.start_sindy:
            args.sindy_enabled = True
        
    # Close the TensorBoard writer
    writer.close()

    # sparsity_mask = (model.sindy_model.coefficients_mask.data.abs() > 0).float()
    logger.info("Sparsity pattern Υ determined.")
    return 

def refine(args, model, train_loader, val_loader, optimizer, logger):

    logger.info("Starting refinement phase")

    for epoch in range(args.refinement_epochs):
        args.epoch = epoch + args.epochs
        logger.info(f"Refinement Epoch {epoch + 1}/{args.refinement_epochs}")

        # Training phase
        train_loss, train_individual_losses = run_epoch(args, model, train_loader, optimizer, training=True, refinement = True)
        
        # Log total training loss
        logger.info(f"[Refinement] Epoch {epoch + 1}/{args.epochs} - Total Loss: {train_loss:.4f}")

        # Log averaged individual losses for training
        logger.info(
            f"[Refinement - Train] Average Losses - "
            f"NLL: {train_individual_losses.get('nll', 0.0):.4f}, "
            f"KL: {train_individual_losses.get('kl', 0.0):.4f}, "
            f"Recon Loss: {train_individual_losses.get('recon_loss', 0.0):.4f}, "
            f"PO Loss: {train_individual_losses.get('potential_outcome_loss', 0.0):.4f}, "
            f"Treatment Loss: {train_individual_losses.get('treatment_loss', 0.0):.4f}, "
            f"SINDY X Loss: {train_individual_losses.get('sindy_x_loss', 0.0):.4f}, "
            f"SINDY Z Loss: {train_individual_losses.get('sindy_z_loss', 0.0):.4f}, "
        )

        # Validation phase
        if (epoch + 1) % args.eval_freq == 0:
            val_loss, val_individual_losses = run_epoch(args, model, val_loader, training=False)
            logger.info(f"[Refinement - Validation] Epoch {epoch + 1}/{args.epochs} - Total Loss: {val_loss:.4f}")

            # Log averaged individual losses for validation
            logger.info(
                f"[Refinement - Validation] Average Losses - "
                f"NLL: {val_individual_losses.get('nll', 0.0):.4f}, "
                f"KL: {val_individual_losses.get('kl', 0.0):.4f}, "
                f"Recon Loss: {val_individual_losses.get('recon_loss', 0.0):.4f}, "
                f"Potential Outcome Loss: {val_individual_losses.get('potential_outcome_loss', 0.0):.4f}, "
                f"Treatment Loss: {val_individual_losses.get('treatment_loss', 0.0):.4f}, "
                f"SINDY X Loss: {val_individual_losses.get('sindy_x_loss', 0.0):.4f}, "
                f"SINDY Z Loss: {val_individual_losses.get('sindy_z_loss', 0.0):.4f}, "
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
                ckpt_path = os.path.join(args.save_exp, "checkpoint.pt")
                torch.save(save_dict, ckpt_path)
                logger.info(f"Model saved: {ckpt_path}")


def main():
    # Parse arguments
    args = parse_args()
    args.library_dim = library_size(n = args.latent_dim , poly_order = args.poly_order, include_sine = args.include_sine, ode_order = args.order)

    # if the latent dimensions is greater than the number of odes being discovered, then some of the latent dimensions are static (not changing wrt to time)
    if args.num_ode < args.latent_dim:
        args.static = True
    else:
        args.static = False

    for model_idx in range(args.num_models):

        args.best_loss = float('inf')

        print(f"Training model {model_idx + 1}/{args.num_models}")

        # Setup logging
        args.save_exp = os.path.join(args.save_dir, f"model_{model_idx + 1}")
        logger = setup_logging(args)
        logger.info(f"Arguments: {vars(args)}")
     
        # Setup DataLoader
        train_loader, val_loader, _ = setup_dataloader(args)

        # Initialize model and optimizer
        #model = VAE(args).to(args.device)
        from autoencoder import Autoencoder
        #from autoencoder_conv import ConvAutoencoder as Autoencoder
        model = Autoencoder(args).to(args.device)
        #model = ConvAutoencoder(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info("Model initialized.")

        # Train and refine
        train(args, model, train_loader, val_loader, optimizer, logger)
        logger.info("Training complete.")

        logger.info("Starting refinement phase...")
        refine(args, model, train_loader, val_loader, optimizer, logger)
        logger.info("Refinement complete.")

        print(f"Finished training model {model_idx + 1}/{args.num_models}")

if __name__ == "__main__":
    main()
