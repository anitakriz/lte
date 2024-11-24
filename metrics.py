import numpy as np
import torch
import os
import cv2 

def compute_metrics(args, model, batch):

    # # For CF RMSE: TODO for entire dataset
        # val_file = os.path.join(args.dataset_dir, 'mnist_val_traj.npz')
        # data = np.load(val_file, allow_pickle=True)

    true_rmse, cf_rmse = get_rmse(args, batch, model)

    return true_rmse, cf_rmse

def get_rmse(args, batch, model, visualize = True):
    """
    Calculate RMSE for true and counterfactual trajectories.

    Parameters:
    -----------
    data: dict
        Loaded npz file with keys:
        - 'true_treatments': Treatment for the true trajectory [shape: (samples,)]
        - 'cf_treatments': Treatment for the counterfactual trajectory [shape: (samples,)]
        - 'true_images': Images of the true trajectory [shape: (samples, timepoints/sample, W, H)]
        - 'cf_images': Images of the counterfactual trajectory [shape: (samples, timepoints/sample, W, H)]
        - 'true_sizes': Sizes (potential outcomes) of the true trajectory [shape: (samples, timepoints/sample)]
        - 'cf_sizes': Sizes (potential outcomes) of the counterfactual trajectory [shape: (samples, timepoints/sample)]
    model: object
        Autoencoder + SINDy model with method `predict_next_latents(init_image, treatment, num_steps)`

    Returns:
    --------
    true_rmse: np.ndarray
        RMSE for the true trajectory at each timestep [shape: (timesteps,)]
    cf_rmse: np.ndarray
        RMSE for the counterfactual trajectory at each timestep [shape: (timesteps,)]
    """
    # Extract relevant data TODO! for entire datset
        # true_images = torch.tensor(data["true_images"], dtype = torch.float32)
        # cf_images = torch.tensor(data["cf_images"], dtype = torch.float32)
        # init_image = true_images[:, 0, :, :]  # Initial image (samples, H, W)
        # true_treatment = torch.tensor(data["true_treatments"], dtype = torch.float32)
        # cf_treatment = torch.tensor(data["cf_treatments"], dtype = torch.float32)
        # true_size = torch.tensor(data['true_sizes'], dtype=torch.float32)
        # cf_size = torch.tensor(data['cf_sizes'], dtype=torch.float32)
    # Predict latent trajectories for true and counterfactual treatments
    true_images = torch.tensor(batch["true_image"], dtype=torch.float32)
    cf_images = torch.tensor(batch["cf_image"], dtype=torch.float32)
    img_by_traj = true_images.view(args.bs // args.timepoints, args.timepoints, args.input_dim, args.input_dim)
    init_image = img_by_traj[:, 0, :, :]
    true_treatment = torch.tensor(batch["true_treatment"], dtype=torch.float32)
    cf_treatment = torch.tensor(batch["cf_treatment"], dtype=torch.float32)
    true_size = torch.tensor(batch['true_size'], dtype=torch.float32)
    cf_size = torch.tensor(batch['cf_size'], dtype=torch.float32)

    # Select the first sample of each trajectory
    selected_indices = torch.arange(0, args.bs, args.timepoints)
    # For the treatment and size inputs, select the corresponding values at the initial time step
    init_true_treatment = true_treatment[selected_indices]
    init_cf_treatment = cf_treatment[selected_indices]
    init_true_size = true_size[selected_indices]
    init_cf_size = cf_size[selected_indices]

    true_z_traj_hat = model.predict_next_latents(init_image.to(next(model.parameters()).device), init_true_treatment.to(next(model.parameters()).device), init_true_size.to(next(model.parameters()).device),  num_steps=args.pred_timepoints, dt = .05)
    cf_z_traj_hat = model.predict_next_latents(init_image.to(next(model.parameters()).device), init_cf_treatment.to(next(model.parameters()).device), init_true_size.to(next(model.parameters()).device), num_steps=args.pred_timepoints, dt = .05)

    # Extract predicted sizes (first latent dimension) for true and counterfactual
    true_size_hat = true_z_traj_hat[:, :, 0]  # Shape: (samples, timesteps)
    cf_size_hat = cf_z_traj_hat[:, :, 0]      # Shape: (samples, timesteps)
    

    true_size_del = true_size.view(int(args.bs/args.timepoints), args.timepoints)[:, 1:1 + args.pred_timepoints]
    cf_size_del = cf_size.view(int(args.bs/args.timepoints), args.timepoints)[:, 1:1 + args.pred_timepoints]

    # Compute RMSE for forecasting tau steps ahead
    true_rmse = torch.sqrt(torch.mean((true_size_del - true_size_hat.cpu()) ** 2, dim=0))  # Shape: (timesteps,)
    cf_rmse = torch.sqrt(torch.mean((cf_size_del - cf_size_hat.cpu()) ** 2, dim=0))        # Shape: (timesteps,)

    return true_rmse.detach().tolist(), cf_rmse.detach().tolist()


    # if visualize:
    #     visualize_trajectories(args, 
    #         true_images.cpu().numpy(),
    #         cf_images.cpu().numpy(),
    #         true_size[:, 1: 1 + timesteps].numpy(),
    #         cf_size[:, 1: 1 + timesteps].numpy(),
    #         true_size_hat.cpu(),
    #         cf_size_hat.cpu(),
    #         init_image.cpu().numpy(),
    #     )
# import matplotlib.pyplot as plt

# def visualize_trajectories(args, true_images, cf_images, true_size, cf_size, true_size_hat, cf_size_hat, base_images, 
#                            num_samples=5, img_size=(28, 28)):
#     """
#     Visualize the true and counterfactual trajectories alongside their predictions.

#     Parameters:
#     -----------
#     true_images: np.ndarray
#         Images of the true trajectory [shape: (samples, timesteps, W, H)].
#     cf_images: np.ndarray
#         Images of the counterfactual trajectory [shape: (samples, timesteps, W, H)].
#     true_size: np.ndarray
#         Actual sizes for the true trajectory [shape: (samples, timesteps)].
#     cf_size: np.ndarray
#         Actual sizes for the counterfactual trajectory [shape: (samples, timesteps)].
#     true_size_hat: torch.Tensor
#         Predicted sizes for the true trajectory [shape: (samples, timesteps)].
#     cf_size_hat: torch.Tensor
#         Predicted sizes for the counterfactual trajectory [shape: (samples, timesteps)].
#     base_images: np.ndarray
#         Base images for size generation [shape: (samples, W, H)].
#     num_samples: int
#         Number of samples to visualize (default: 5).
#     img_size: tuple
#         Size of the generated images (default: (28, 28)).
#     """
#     timesteps = true_size_hat.shape[1]
#     num_samples = min(num_samples, true_images.shape[0])  # Ensure num_samples <= available samples

#     # Create a single figure for all samples
#     fig, axes = plt.subplots(4 * num_samples, timesteps, figsize=(2 * timesteps, 8 * num_samples))
#     fig.suptitle("Trajectories Visualization", fontsize=14)

#     for sample_idx in range(num_samples):
#         # Generate predicted true and counterfactual images
#         predicted_true_images = [
#             generate_image((base_images[sample_idx] * 255).astype(np.uint8), size, img_size)
#             for size in true_size_hat[sample_idx]
#         ]
        
#         predicted_cf_images = [
#             generate_image(base_images[sample_idx] * 255, size, img_size)
#             for size in cf_size_hat[sample_idx]
#         ]

#         for t in range(timesteps):
#             # Row 1: True trajectory
#             ax = axes[4 * sample_idx, t] 
#             ax.imshow(true_images[sample_idx, t] * 255, cmap="gray")
#             ax.axis("off")
#             if t == 0:
#                 ax.set_title(f"Sample {sample_idx}\nTrue", fontsize=12)

#             # Add true size text above the true trajectory image
#             ax.text(0.5, 1.05, f"True Size: {true_size[sample_idx][t]:.4f}", 
#                     transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="blue")

#             # Row 2: Predicted true trajectory
#             ax = axes[4 * sample_idx + 1, t] 
#             ax.imshow(predicted_true_images[t] * 255, cmap="gray")
#             ax.axis("off")
#             if t == 0:
#                 ax.set_title("Predicted True", fontsize=12)
#             # Add predicted size above the image with 4 decimal places
#             ax.text(0.5, 1.05, f"Pred Size: {true_size_hat[sample_idx][t]:.4f}", 
#                     transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="red")

#             # Row 3: Counterfactual trajectory
#             ax = axes[4 * sample_idx + 2, t] 
#             ax.imshow(cf_images[sample_idx, t] * 255, cmap="gray")
#             ax.axis("off")
#             if t == 0:
#                 ax.set_title("Counterfactual", fontsize=12)

#             # Add counterfactual size text above the counterfactual image
#             ax.text(0.5, 1.05, f"CF Size: {cf_size[sample_idx][t]:.4f}", 
#                     transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="blue")

#             # Row 4: Predicted counterfactual trajectory
#             ax = axes[4 * sample_idx + 3, t] 
#             ax.imshow(predicted_cf_images[t] * 255, cmap="gray")
#             ax.axis("off")
#             if t == 0:
#                 ax.set_title("Predicted CF", fontsize=12)
#             # Add predicted counterfactual size above the image with 4 decimal places
#             ax.text(0.5, 1.05, f"Pred CF Size: {cf_size_hat[sample_idx][t]:.4f}", 
#                     transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="red")

#     plt.tight_layout()

#     # Save the combined plot
#     plot_dir = os.path.join(args.save_dir, "plots/traj")
#     os.makedirs(plot_dir, exist_ok=True)
#     plot_path = os.path.join(plot_dir, f"traj_{args.epoch}.png")
#     plt.savefig(plot_path)
#     plt.close()




# def resize_digit(base_img, target_size):
#     """
#     Resize the MNIST digit image to the specified size while maintaining aspect ratio.
    
#     Parameters:
#         base_img (np.ndarray): The original MNIST digit image.
#         target_size (int): The target size for the digit (assumes square output).

#     Returns:
#         np.ndarray: A resized image of the digit.
#     """
#     # Resize image to target size with anti-aliasing
#     resized_img = cv2.resize(base_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
#     # Scale back to [0, 255] as integers
#     return (resized_img).astype(np.uint8)

# def generate_image(base_img, size, img_size):
#     """
#     Generate an MNIST digit image with a specified size on a blank canvas, normalized to [0, 1].

#     Parameters:
#         base_img (np.ndarray): The original MNIST digit image.
#         size (float): The relative size of the digit (0, 1].
#         img_size (tuple): The canvas size (height, width).

#     Returns:
#         np.ndarray: A 2D array (float32) representing the image with the resized digit, normalized to [0, 1].
#     """
#     size = np.clip(size.detach().numpy(), 0.0, 1.0) 

#     # Determine target size for the digit
    
#     target_size = max(1, int(size * img_size[0]))  # Ensure size is at least 1 pixel
#     resized_digit = resize_digit(base_img, target_size)

#     # Create a blank canvas
#     canvas = np.zeros(img_size, dtype=np.uint8)

#     # Center the digit on the canvas
#     pos_y = img_size[0] // 2 - target_size // 2
#     pos_x = img_size[1] // 2 - target_size // 2

#     # Place the resized digit on the canvas
#     canvas[pos_y:pos_y + resized_digit.shape[0], pos_x:pos_x + resized_digit.shape[1]] = resized_digit

#     # Normalize to [0, 1] range
#     return canvas.astype(np.float32) / 255.