import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os 
from metrics import compute_metrics
from vae import VAE
from autoencoder import Autoencoder

def plot_recon(args, model, x):
    """Function to plot true vs predicted (x_hat) for the validation set."""
    # Ensure the plot directory exists
    plot_dir = os.path.join(args.save_exp, "plots/recon")
    os.makedirs(plot_dir, exist_ok=True)

    # Select the first 20 samples from x and x_hat
    num_samples = 10
    x_samples = x[:num_samples]
    x_hat_samples = model.get_recon(x_samples)

    # Create a figure with two rows: true images and predicted images
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))

    for i in range(num_samples):
        # Convert x and x_hat to numpy arrays and reshape them
        x_img = x_samples[i].cpu().numpy().reshape(args.input_dim, args.input_dim)
        x_hat_img = x_hat_samples[i].cpu().detach().numpy().reshape(args.input_dim, args.input_dim)

        # Rescale the images to the range [0, 255] and convert to uint8
        x_img = (x_img * 255).clip(0, 255).astype("uint8")
        x_hat_img = (x_hat_img * 255).clip(0, 255).astype("uint8")

        # Plot the true image in the first row
        axes[0, i].imshow(x_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("True Images")

        # Plot the predicted image in the second row
        axes[1, i].imshow(x_hat_img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Predicted Images (x_hat)")

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"plot_epoch_{args.epoch + 1}.png")
    plt.savefig(plot_path)
    plt.close()

def sample_images(args, model, batch):
    """
    Visualize the VAE progress during training by generating reconstructions and samples.

    Args:
        args: Namespace object containing arguments like save directory, visualization frequency, etc.
        model: The VAE model in evaluation mode.
        batch: A dictionary containing batch data with keys:
            - "true_image": Original images.
            - "true_treatment": Treatment variables for the batch.
            - "true_size": Size variables for the batch.
    """
    save_path = os.path.join(args.save_exp, "plots")
    os.makedirs(save_path, exist_ok=True)

    # Select 10 samples from the batch (you can change this to random selection if needed)
    sample_indices = torch.randperm(len(batch["true_image"]))[:10]  # Randomly select 10 samples
    selected_images = batch["true_image"][sample_indices]
    selected_treatments = batch["true_treatment"][sample_indices]
    selected_sizes = batch["true_size"][sample_indices]

    fig, axes = plt.subplots(13, 10, figsize=(15, 15))  # 12 rows, 10 columns
    fig.suptitle(f"VAE Visualization at Iter {args.iter}", fontsize=16)

    # First row: Original images
    for i, img in enumerate(selected_images):
        axes[0, i].imshow(img.cpu().numpy() * 255, cmap='gray')  # Grayscale image
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

    # Second row: Reconstructed images
    x_recon = model.get_recon(selected_images.to(args.device), selected_treatments.to(args.device), selected_sizes.to(args.device))
    for i, img in enumerate(x_recon):
        axes[1, i].imshow(img.squeeze(0).detach().cpu().numpy() * 255, cmap='gray')  # Grayscale image
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    # Rows 3-12: Sampled images at different temperatures
    for t_idx, t in enumerate(torch.arange(0.0, 1.1, 0.1)): # Sampling at 10 different temperatures
        samples, _ = model.sample(selected_treatments.to(args.device), selected_sizes.to(args.device), t=t)
 
        for i, img in enumerate(samples):
            img = img.squeeze(0).detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            axes[t_idx + 2, i].imshow(img, cmap='gray')  # Grayscale image
            axes[t_idx + 2, i].set_title(f"Sample T={t:.1f}")
            axes[t_idx + 2, i].axis("off")

    # Save the plot
    file_path = os.path.join(save_path, f"vae_viz_iter_{args.iter}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
    plt.savefig(file_path)
    plt.close(fig)

def save_dynamics(args, model, batch):
    # Get the dynamics information from the model

    if args.dataset == 'Pendulum':
        dynamics_info = model.get_ode(batch['x'].to(args.device), batch['dx'].to(args.device), batch['ddx'].to(args.device))
    elif args.dataset == 'MNIST_UNTREATED' or 'DOT':
        dynamics_info = model.get_ode(batch['x'].to(args.device), batch['dx'].to(args.device))

    # dynamics_info = model.get_ode(
    #     batch["true_image"].to(args.device), 
    #     batch["true_treatment"].to(args.device), 
    #     batch["true_size"].to(args.device)
    # )

    # Create a directory for saving dynamics information if it doesn't exist
    ode_dir = os.path.join(args.save_exp, "ode")
    os.makedirs(ode_dir, exist_ok=True)

    # Define the path for saving the dynamics info
    ode_path = os.path.join(ode_dir, "dynamics_info.txt")

    # if args.dataset == 'MNIST': #TODO
    #     true_rmse, cf_rmse = compute_metrics(args, model)

    # Open the file in append mode to save dynamics info for each epoch
    with open(ode_path, 'a') as f:
        # Write a header with the epoch number
        f.write(f"Dynamics Information for Epoch {args.epoch + 1}\n")
        
        # Write the dynamics values
        f.write("Dynamics:\n")

        for term, values in dynamics_info['dynamics'].items():
            f.write(f"{term}: {values}\n")
        
        # Write the ODE equations
        f.write("ODE Equations:\n")
        for z_key, ode_eq in dynamics_info["ode_equations"].items():
            f.write(f"{z_key}:\t{ode_eq}\n")
        
        # Write the RMSE metrics TODO
        # if args.dataset == 'MNIST':
        #     f.write("RMSE Metrics:\n")
        #     f.write(f"True RMSE: {true_rmse.item() if isinstance(true_rmse, torch.Tensor) else true_rmse}\n")  # Ensure tensor is converted to scalar
        #     f.write(f"Counterfactual RMSE: {cf_rmse.item() if isinstance(cf_rmse, torch.Tensor) else cf_rmse}\n")  # Same here
            
        #     # Add a separator between different epochs for clarity
        #     f.write("\n" + "="*50 + "\n")

def predict_attributes(args, model, batch):
    """
    Predict attributes for a batch and save dynamics information to a file.

    Args:
    - args: Arguments/configurations for the model (includes save_exp, device, etc.).
    - model: The VAE model instance.
    - batch: A dictionary containing input data (e.g., images, treatments).

    Saves:
    - A file containing predictions, metrics, and dynamics information at each iteration.
    """
 
    # Create a directory for saving dynamics information if it doesn't exist
    preds_dir = os.path.join(args.save_exp, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Define the path for saving the dynamics info
    preds_path = os.path.join(preds_dir, "predictions_ncm.txt")

    # Predict v_hat and obs_v using the model
    v_hat, obs_v = model.predict_v_hat(batch["true_image"].to(args.device), batch["true_treatment"].to(args.device), batch["true_size"].to(args.device))

    # Calculate Losses
    treatment_loss = nn.BCEWithLogitsLoss()(v_hat[:, 0], obs_v[:, 0])  # Treatment prediction loss
    outcome_loss = nn.MSELoss()(v_hat[:, -1], obs_v[:, -1])  # Outcome prediction loss

    # Optionally, compute accuracy for treatment prediction
    treatment_preds = torch.sigmoid(v_hat[:, 0]) > 0.5  # Convert logits to binary predictions
    treatment_accuracy = (treatment_preds == obs_v[:, 0]).float().mean().item()  # Accuracy

    # Write predictions and metrics to the file
    with open(preds_path, 'a') as f:
        # Write a header with the iteration number
        f.write(f"Predictions at Iter {args.iter}\n")

        # Write metrics to the file
        f.write(f"Treatment Loss: {treatment_loss.item():.4f}\n")
        f.write(f"Outcome Loss: {outcome_loss.item():.4f}\n")
        f.write(f"Treatment Accuracy: {treatment_accuracy:.4f}\n")

        f.write("=" * 50 + "\n")