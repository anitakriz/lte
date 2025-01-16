import numpy as np
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class PendulumDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the .npz file containing the dataset.
        """
        # Load the data from the .npz file
        data = np.load(file_path)
        # Ensure the required keys 'x', 'dx', and optionally 'ddx' exist in the loaded data
        required_keys = ['x', 'dx', 'ddx', 'z', 'dz']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key '{key}' in the dataset file.")
                
        self.data = {key: torch.tensor(value, dtype=torch.float32) for key, value in data.items() if key in required_keys}

    def __len__(self):
        return self.data['x'].shape[0]  # Number of samples in the dataset
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the input data for the given index.
        """
        sample = {key: value[idx] for key, value in self.data.items()}
        return sample

class MNIST_UNTREATED_Dataset(Dataset):
    def __init__(self, npz_file):

        # Load the dataset from the .npz file
        data = np.load(npz_file, allow_pickle=True)
        
        # Load data and convert to torch.float32
        self.image_ids = torch.tensor(data['image_ids'], dtype=torch.long)  # Assuming ids are integers
        self.c = torch.tensor(data['c'], dtype=torch.long)  # Assuming labels are integers

        # Process the image data and derivatives
        self.x = torch.tensor(data['x'], dtype=torch.float32)  # Exclude first timestep
        self.dx = torch.tensor(data['dx'], dtype=torch.float32)

        # Process size data (true_size, cf_size) by ensuring size is given for each timepoint of a sample
        self.o = torch.tensor(data['o'], dtype=torch.float32).flatten()

        # Get number of trajectories and timesteps
        self.num_trajectories, self.num_timesteps, *_ = self.x.shape

        # Flattening the dataset for per-time-point samples (i.e [samples, timepoints, H, W] -> [samples * timepoints, H, W])
        self.flattened_data = {
            "image_id": torch.repeat_interleave(self.image_ids, self.num_timesteps), # [sample * timepoints]
            "digit": torch.repeat_interleave(self.c, self.num_timesteps), # [sample * timepoints]
            "time_step": torch.tile(torch.arange(self.num_timesteps), (self.num_trajectories,)), # [sample * timepoints ]
            "x": self.x.reshape(-1, *self.x.shape[2:]), # [sample * timepoints, H, W ]
            "dx": self.dx.reshape(-1, *self.dx.shape[2:]), # [sample * timepoints, H, W ]
            "size": self.o, # [sample * timepoints]
        }


    def __len__(self):
        """Return total number of samples."""
        return self.num_trajectories * self.num_timesteps

    def __getitem__(self, idx):
        """Retrieve the data for a specific flattened index."""
  
        return {key: value[idx] if "x" not in key and "dx" not in key 
                else value[idx].float()  # Ensure all images and derivatives are float32
                for key, value in self.flattened_data.items()}


class Dot_Dataset(Dataset):
    def __init__(self, npz_file):

        # Load the dataset from the .npz file
        data = np.load(npz_file, allow_pickle=True)
        
        # Load data and convert to torch.float32
        self.image_ids = torch.tensor(data['image_ids'], dtype=torch.long)  # Assuming ids are integers

        # Process the image data and derivatives
        self.x = torch.tensor(data['x'], dtype=torch.float32)  # Exclude first timestep
        self.dx = torch.tensor(data['dx'], dtype=torch.float32)

        # Process size data (true_size, cf_size) by ensuring size is given for each timepoint of a sample
        self.o = torch.tensor(data['o'], dtype=torch.float32).flatten()

        # Get number of trajectories and timesteps
        self.num_trajectories, self.num_timesteps, *_ = self.x.shape

        # Flattening the dataset for per-time-point samples (i.e [samples, timepoints, H, W] -> [samples * timepoints, H, W])
        self.flattened_data = {
            "time_step": torch.tile(torch.arange(self.num_timesteps), (self.num_trajectories,)), # [sample * timepoints ]
            "x": self.x.reshape(-1, *self.x.shape[2:]), # [sample * timepoints, H, W ]
            "dx": self.dx.reshape(-1, *self.dx.shape[2:]), # [sample * timepoints, H, W ]
            "size": self.o, # [sample * timepoints]
        }


    def __len__(self):
        """Return total number of samples."""
        return self.num_trajectories * self.num_timesteps

    def __getitem__(self, idx):
        """Retrieve the data for a specific flattened index."""
  
        return {key: value[idx] if "x" not in key and "dx" not in key 
                else value[idx].float()  # Ensure all images and derivatives are float32
                for key, value in self.flattened_data.items()}

class MNIST_TE_Dataset(Dataset):
    def __init__(self, npz_file):
        """
        Initialize the MNIST_TE_Dataset from a .npz file.

        Args:
            npz_file (str): Path to the .npz file containing the dataset.
        """
        # Load the dataset from the .npz file
        data = np.load(npz_file, allow_pickle=True)
        
        # Load data and convert to torch.float32
        self.image_ids = torch.tensor(data['image_ids'], dtype=torch.long)  # Assuming ids are integers
        self.labels = torch.tensor(data['labels'], dtype=torch.long)  # Assuming labels are integers

        # Process the image data and derivatives
        self.true_images = torch.tensor(data['true_images'], dtype=torch.float32)  # Exclude first timestep
        self.true_derivatives = torch.tensor(data['true_derivatives'], dtype=torch.float32)
        self.true_treatments = torch.tensor(data['true_treatments'], dtype=torch.float32)

        self.cf_images = torch.tensor(data['cf_images'], dtype=torch.float32)  # Exclude first timestep
        self.cf_derivatives = torch.tensor(data['cf_derivatives'], dtype=torch.float32)
        self.cf_treatments = torch.tensor(data['cf_treatments'], dtype=torch.float32)

        # Process size data (true_size, cf_size) by ensuring size is given for each timepoint of a sample
        self.true_size = torch.tensor(data['true_sizes'], dtype=torch.float32).flatten()
        self.cf_size = torch.tensor(data['cf_sizes'], dtype=torch.float32).flatten()

        # Get number of trajectories and timesteps
        self.num_trajectories, self.num_timesteps, *_ = self.true_images.shape

        # Flattening the dataset for per-time-point samples (i.e [samples, timepoints, H, W] -> [samples * timepoints, H, W])
        self.flattened_data = {
            "image_id": torch.repeat_interleave(self.image_ids, self.num_timesteps), # [sample * timepoints]
            "digit": torch.repeat_interleave(self.labels, self.num_timesteps), # [sample * timepoints]
            "time_step": torch.tile(torch.arange(self.num_timesteps), (self.num_trajectories,)), # [sample * timepoints ]
            "true_image": self.true_images.reshape(-1, *self.true_images.shape[2:]), # [sample * timepoints, H, W ]
            "true_derivative": self.true_derivatives.reshape(-1, *self.true_derivatives.shape[2:]), # [sample * timepoints, H, W ]
            "cf_image": self.cf_images.reshape(-1, *self.cf_images.shape[2:]), # [sample * timepoints, H, W ]
            "cf_derivative": self.cf_derivatives.reshape(-1, *self.cf_derivatives.shape[2:]), # [sample * timepoints, H, W ]
            "true_treatment": torch.tile(self.true_treatments, (self.num_timesteps, 1)).T.flatten(), # [sample * timepoints]
            "cf_treatment": torch.tile(self.cf_treatments, (self.num_timesteps, 1)).T.flatten(), # [sample * timepoints]
            "true_size": self.true_size, # [sample * timepoints]
            "cf_size": self.cf_size, # [sample * timepoints]
        }


    def __len__(self):
        """Return total number of samples."""
        return self.num_trajectories * self.num_timesteps

    def __getitem__(self, idx):
        """Retrieve the data for a specific flattened index."""

        return {key: value[idx] if "image" not in key and "derivative" not in key 
                else value[idx].float()  # Ensure all images and derivatives are float32
                for key, value in self.flattened_data.items()}

if __name__ == '__main__':

    # Path to the saved .npz file
    npz_file = "/usr/local/data/anitakriz/lte/mnist_train_traj.npz"

    # Initialize the dataset
    mnist_te_dataset = MNIST_TE_Dataset(npz_file)


    # Get the length of the dataset
    dataset_length = len(mnist_te_dataset)

    # Print the length
    print(f"Length of the dataset: {dataset_length}")

    from torch.utils.data import DataLoader

    # Create a DataLoader for batching
    batch_size = 32
    mnist_te_dataloader = DataLoader(mnist_te_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through the DataLoader
    for batch in mnist_te_dataloader:
        print(batch["true_images"].shape) 
        X = batch["true_images"].reshape(batch_size * batch["true_images"].shape[1], batch["true_images"].shape[2] * batch["true_images"].shape[3])
        print(X.shape)
        print(done)
       

