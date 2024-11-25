import torch 
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim


# Xavier Initialization Helper Function
def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        input_dim = layer.in_features
        output_dim = layer.out_features
        alpha = input_dim + output_dim
        bound = (6 / alpha) ** 0.5
        nn.init.uniform_(layer.weight, -bound, bound)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size + 1, 128)  # Add 1 for the treatment input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, latent_dim)  # Output: latent_dim (size, digit, treatment)
        self.apply(xavier_init)  

    def forward(self, x, treatment):
        # print(x.shape)
        # print(treatment.shape)
        # print(done)
        x = torch.cat((x, treatment), dim=1)  # Concatenate treatment with the input
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        z = self.fc4(x)
        return z


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, input_size)
        self.apply(xavier_init)  # Apply Xavier initialization

        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

class SINDyLayer(nn.Module):
    def __init__(self, library_dim, latent_dim):
        super(SINDyLayer, self).__init__()

        self.library_dim = library_dim
        self.coefficients = nn.Parameter(torch.randn(library_dim, latent_dim))  # Trainable Ξ (only 1 dim. for z(0) as we dont find the derivative for hte others)
        self.register_buffer('coefficient_mask', torch.ones(library_dim, latent_dim))  

        self.threshold = 1e-3 #sequential thresholding

    def forward(self, z):
        # Compute Θ(z^T), the library of functions
        theta = self.compute_library(z)
        # Apply the mask to enforce sparsity
        masked_coefficients = self.coefficients * self.coefficient_mask
        # Compute Θ(z^T)Ξ
        z_dot_pred = torch.matmul(theta, masked_coefficients)
        l1_reg_loss = torch.mean(torch.abs(masked_coefficients))  # L1 regularization on sparse coefficients

        return z_dot_pred, l1_reg_loss

    def update_mask(self):
        """
        Apply a sequential threshold to the coefficients and update the mask every 500 epochs
        
        Args:
            t (int): The current training step.
        """
        # Update the mask by applying the threshold
        with torch.no_grad():
            self.coefficient_mask = (torch.abs(self.coefficients) > self.threshold).float()
        

    def compute_library(self, z):
        """
        Create the library Θ(z) for dz/dt where z[0] = size, z[1] = digit, z[2] = treatment.
        Size = [timepoints (or batch), library_dim]
        """
        size = z[:, 0:1]  # z[0]
        digit = z[:, 1:2]  # z[1]
        treatment = z[:, 2:3]  # z[2]
        #TODO: need to make more general 
        # Build the library specifically for size dynamics
        theta = torch.cat([
            torch.ones_like(size),       # Constant term
            size,                        # z[0]
            size**2,                     # z[0]^2
            size * digit,                # z[0] * z[1]
            size * treatment,            # z[0] * z[2]
            size**2 * digit,              # z[0]^2 * z[1]
            size**2 * treatment          # z[0]^2 * z[2]
        ], dim=1)

        return theta

    def get_dynamics(self, z=None):
        """
        Return the learned ODE (dynamics) by returning the coefficients with their corresponding library terms.
        Instead of printing, return a dictionary with the terms and coefficients for each equation (z1, z2, z3).
        """
        # Compute the library terms
        theta = self.compute_library(z)
        masked_coefficients = self.coefficients * self.coefficient_mask  # Apply the mask to coefficients
        
        # Define the library terms corresponding to theta
        dynamics_terms = [
            "1", "z[0]", "z[0]^2", 
            "z[0]*z[1]", "z[0]*z[2]", "z[0]^2*z[1]", "z[0]^2*z[2]"
        ]

        # Initialize a dictionary to hold the dynamics for z1, z2, and z3
        dynamics = {"z1": {}, "z2": {}, "z3": {}}
        ode_equations = {"z1": "dz1/dt = ", "z2": "dz2/dt = ", "z3": "dz3/dt = "}

        # Loop over each equation (z1, z2, z3)
        for z_idx in range(3):  # Assume 3 components: z1, z2, z3
            coefficients_column = masked_coefficients[:, z_idx]  # Extract the column for z_idx

            for i, term in enumerate(dynamics_terms):
                coefficient_value = coefficients_column[i].item()
                if abs(coefficient_value) > 1e-5:  # Only include significant coefficients
                    dynamics[f"z{z_idx + 1}"][term] = coefficient_value
                    ode_equations[f"z{z_idx + 1}"] += f"{coefficient_value:.4f} * {term} + "

            # Clean up the trailing '+' in the ODE equation
            ode_equations[f"z{z_idx + 1}"] = ode_equations[f"z{z_idx + 1}"].strip(" + ")

        # Return the dynamics and ODE equations as a dictionary
        return {"dynamics": dynamics, "ode_equations": ode_equations}


# Autoencoder with SINDy
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim, library_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(input_size, latent_dim)
        self.sindy_layer = SINDyLayer(library_dim, latent_dim)

    def forward(self, x, x_dot, treatment, size):     

        # Encoder forward pass
        z = self.encoder(x, treatment)  # z[0] ~ Size, z[1] ~ Digit

        # Enforce one dim represents  the potential outcome, while still letting encoder build latent 
        loss_potential_outcome = nn.MSELoss()(z[:, 0].unsqueeze(1), size)

        # Enforce one dim to represent the treatment, while still letting encoder have info. about treatment
        loss_treatment = nn.BCEWithLogitsLoss()(z[:, -1].unsqueeze(1), treatment.float())

        supervised_loss = loss_potential_outcome + loss_treatment
        # Perform SINDy operations for z[0] (fxn of treatment)
        z_dot_pred, sindy_l1_loss = self.sindy_layer(z) 

        # Compute the true derivative of z_dot based wrt x_dot: dz/dx * dx/dt
        encoder_parameters = list(self.encoder.parameters())
        encoder_weight_list = [w for w in encoder_parameters if len(w.shape) == 2]
        encoder_biases_list = [b for b in encoder_parameters if len(b.shape) == 1]
        z_dot_true = self.compute_derivative(torch.cat((x, treatment), dim=1), torch.cat((x_dot, treatment), dim=1), encoder_weight_list, encoder_biases_list, activation='sigmoid')
        
        # Compute latent-level der. loss
        sindy_z_loss = nn.MSELoss()(z_dot_true, z_dot_pred)

        # Compute the approx. derivative of x_dot: d(DEC)/dz * z_dot_pred
        decoder_parameters = list(self.decoder.parameters())
        decoder_weight_list = [w for w in decoder_parameters if len(w.shape) == 2]
        decoder_biases_list = [b for b in decoder_parameters if len(b.shape) == 1]
        x_dot_pred = self.compute_derivative(z, z_dot_pred, decoder_weight_list, decoder_biases_list, activation='sigmoid')
        
        # Compute image-level der. loss
        sindy_x_loss = nn.MSELoss()(x_dot, x_dot_pred)

        # Decoder forward pass
        x_hat = self.decoder(z)
        # Reconstruction loss
        recon_loss = nn.MSELoss()(x, x_hat)

        # Combine all loss components
        return z, x_hat, {
            'potential_outcome_loss': loss_potential_outcome,
            'treatment_loss': loss_treatment,
            'recon_loss': recon_loss.mean(),
            'sindy_x_loss': sindy_x_loss.mean(),
            'sindy_z_loss': sindy_z_loss.mean(),
            'sindy_l1_loss': sindy_l1_loss.mean()
        }

    def compute_derivative(self, x, dx, weights, biases, activation='sigmoid'):
        """
        Compute the first-order time derivatives dz/dt using the chain rule.

        Arguments:
            x - 2D torch tensor [time_points, state_variables], input to the network.
            dx - 2D torch tensor [time_points, input_dim], first-order time derivatives of the input.
            weights - List of torch tensors containing the network weights.
            biases - List of torch tensors containing the network biases.
            activation - Activation function: 'elu', 'relu', 'sigmoid', or 'linear'.

        Returns:
            dz - Torch tensor [time_points, output_dim], first-order time derivatives of the output.
        """
        a = x  # Activation at the current layer
        dz = dx  # Derivative initialized with dx for the input layer

        for i in range(len(weights)):
            # Perform the linear transformation
            z = torch.matmul(a, weights[i].T) + biases[i]

            if activation == 'sigmoid' and i < len(weights) - 1:
                # Sigmoid activation and its derivative
                a = torch.sigmoid(z)
                gprime = a * (1 - a)
                dz = gprime * torch.matmul(dz, weights[i].T)

            elif activation == 'relu' and i < len(weights) - 1:
                # ReLU activation and its derivative
                a = torch.relu(z)
                gprime = (z > 0).float()
                dz = gprime * torch.matmul(dz, weights[i].T)

            elif activation == 'elu' and i < len(weights) - 1:
                # ELU activation and its derivative
                a = torch.nn.functional.elu(z)
                gprime = torch.where(z > 0, torch.ones_like(z), torch.exp(z))
                dz = gprime * torch.matmul(dz, weights[i].T)

            elif activation == 'linear' or i == len(weights) - 1:
                # Linear activation (no non-linearity in the final layer)
                dz = torch.matmul(dz, weights[i].T)

            # Update `a` for the next layer
            a = z if i == len(weights) - 1 and activation == 'linear' else a

        return dz  # Return the derivative
    
    def predict_next_latents(self, x, treatment, num_steps, dt=1):
        """
        Predict the latent trajectories using learned ODEs (SINDy).

        Arguments:
            x (tensor): The input data at the initial time step [shape: (batch_size, ...)].
            treatment (tensor): Treatment or condition applied [shape: (batch_size, treatment_dim)].
            num_steps (int): Number of time steps to predict.
            dt (float): Time step size (default: 1).

        Returns:
            z_traj (tensor): Predicted latent trajectory [shape: (batch_size, num_steps, latent_dim)].
        """
        # Encode the initial state
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) #TODO: remvoe when not FC
        z = self.encoder(x, treatment)  # Initial latent state [shape: (batch_size, latent_dim)]
      
        # Prepare trajectory storage
        batch_size, latent_dim = z.size()
        z_traj = torch.zeros(batch_size, num_steps, latent_dim).to(z.device)

        # Iterate through time steps
        for step in range(num_steps):

            # Predict derivative (dz/dt) using SINDy layer
            z_dot_pred, _ = self.sindy_layer(z)

            # Update latent state using Euler's method
            z = z + z_dot_pred * dt

            z_traj[:, step, :] = z  # Store the predicted next latent state

        return z_traj


if __name__ == '__main__':

    # Example Usage
    input_size = 784  # Input dimension
    latent_dim = 2  # Latent space dimension
    library_dim = 11

    autoencoder = Autoencoder(input_size, latent_dim, library_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)

    # Dummy data
    x = torch.rand(1024, input_size)  # number of timepoints, input size
    x_dot = torch.rand(1024, input_size)  # number of timepoints, input size
    treatment = torch.randint(0, 2, (1024, 1))  # Random treatment labels
    
    print(x.shape)
    print(x_dot.shape)
    print(treatment.shape)
    # Forward pass
    output = autoencoder(x, x_dot, treatment)
    # Print losses
    print(output)



