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
        self.fc1 = nn.Linear(input_size * input_size, 128)  # Add 1 for the treatment input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, latent_dim)  # Output: latent_dim (size, digit, treatment)
        self.apply(xavier_init)  

    def forward(self, x):
        #x = torch.cat((x, treatment), dim=1)  # Concatenate treatment with the input
        x = x.view(x.shape[0], -1)
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
        self.fc4 = nn.Linear(128, input_size * input_size)
        self.apply(xavier_init)  # Apply Xavier initialization

        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

class SINDyModel(nn.Module):
    def __init__(self, args):
        super(SINDyModel, self).__init__()

        self.library_dim = args.library_dim
        self.coefficients = nn.Parameter(torch.randn(args.library_dim, args.num_ode))  # Trainable Ξ (only 1 dim. for z(0) as we dont find the derivative for hte others)
        self.register_buffer('coefficient_mask', torch.ones(args.library_dim, args.num_ode))  

        self.threshold = args.threshold 
        self.latent_dim = args.latent_dim
        self.order = args.order
        self.poly_order = args.poly_order
        self.include_sine = args.include_sine


    def forward(self, z, dz = None):
        # Compute Θ(z^T), the library of functions
        if self.order == 1:
            theta = self.compute_library_order1(z, self.latent_dim, self.poly_order, self.include_sine)
        if self.order == 2: 
            theta = self.compute_library_order2(z, dz, self.latent_dim, self.poly_order, self.include_sine)
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
        
    def compute_library_order1(self, z, latent_dim, poly_order, include_sine=False):
        """
        Build the SINDy library for a first-order system in PyTorch.
        """
        # Initialize the library with the constant term (ones)
        library = [torch.ones(z.shape[0], device=z.device)]

        # Add the first-order terms for z
        for i in range(latent_dim):
            library.append(z[:, i])

        # Add second-order terms for z
        if poly_order > 1:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    library.append(z[:, i] * z[:, j])  # Interaction within z

        # Add higher-order terms as needed
        if poly_order > 2:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k])  # z terms

        # Include sine terms if specified
        if include_sine:
            for i in range(latent_dim):
                library.append(torch.sin(z[:, i]))
                library.append(torch.sin(dz[:, i]))

        # Stack all the terms to create the library
        return torch.stack(library, dim=1)

    def compute_library_order2(self, z, dz, latent_dim, poly_order, include_sine=False):
        """
        Build the SINDy library for a second-order system in PyTorch.
        Process z and dz separately for clarity.
        """
        # Initialize the library with the constant term (ones)
        library = [torch.ones(z.shape[0], device=z.device)]

        # Add the first-order terms for z
        for i in range(latent_dim):
            library.append(z[:, i])

        # Add the first-order terms for dz
        for i in range(latent_dim):
            library.append(dz[:, i])

        # Add second-order terms (pairwise interactions) for z and dz
        if poly_order > 1:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    library.append(z[:, i] * z[:, j])  # Interaction within z
                    library.append(dz[:, i] * dz[:, j])  # Interaction within dz
                    library.append(z[:, i] * dz[:, j])  # Interaction between z and dz

        # Add higher-order terms as needed
        if poly_order > 2:
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k])  # z terms
                        library.append(dz[:, i] * dz[:, j] * dz[:, k])  # dz terms
                        library.append(z[:, i] * z[:, j] * dz[:, k])  # Mixed terms
                        library.append(z[:, i] * dz[:, j] * dz[:, k])  # Mixed terms

        # Include sine terms if specified
        if include_sine:
            for i in range(latent_dim):
                library.append(torch.sin(z[:, i]))
                library.append(torch.sin(dz[:, i]))

        # Stack all the terms to create the library
        return torch.stack(library, dim=1)


    def get_dynamics(self, z, dz=None):
        """
        Return the learned ODE (dynamics) by returning the coefficients with their corresponding library terms.
        Produces a dictionary with the terms and coefficients for each equation (z1, z2, ...).
        """
        # Compute the library terms
        if self.order == 1:
            theta = self.compute_library_order1(z, self.latent_dim, self.poly_order, self.include_sine)
        elif self.order == 2:
            theta = self.compute_library_order2(z, dz, self.latent_dim, self.poly_order, self.include_sine)
        masked_coefficients = self.coefficients * self.coefficient_mask  # Apply the mask to coefficients

        # Initialize library terms dynamically
        library_terms = ["1"]  # Constant term
        for i in range(self.latent_dim):
            library_terms.append(f"z[{i}]")
        if self.order == 2:
            for i in range(self.latent_dim):
                library_terms.append(f"dz[{i}]")

        if self.poly_order > 1:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    library_terms.append(f"z[{i}]*z[{j}]")
                    if self.order == 2:
                        library_terms.append(f"dz[{i}]*dz[{j}]")
                        library_terms.append(f"z[{i}]*dz[{j}]")

        if self.poly_order > 2:
            for i in range(self.latent_dim):
                for j in range(i, self.latent_dim):
                    for k in range(j, self.latent_dim):
                        library_terms.append(f"z[{i}]*z[{j}]*z[{k}]")
                        if self.order == 2:
                            library_terms.append(f"dz[{i}]*dz[{j}]*dz[{k}]")
                            library_terms.append(f"z[{i}]*z[{j}]*dz[{k}]")
                            library_terms.append(f"z[{i}]*dz[{j}]*dz[{k}]")

        if self.include_sine:
            for i in range(self.latent_dim):
                library_terms.append(f"sin(z[{i}])")
                library_terms.append(f"sin(dz[{i}])")

        # Map coefficients to terms for each dynamic equation
        dynamics = {}
        ode_equations = {}
        num_latent_vars = self.coefficients.shape[1]

        for z_idx in range(num_latent_vars):
            coefficients_column = masked_coefficients[:, z_idx]  # Coefficients for the current equation
            dynamics[f"z{z_idx}"] = {}
            ode_equations[f"z{z_idx}"] = f"dz{z_idx}/dt = "

            for term_idx, term in enumerate(library_terms):
                coefficient_value = coefficients_column[term_idx].item()
                if abs(coefficient_value) > 1e-5:  # Include only significant terms
                    dynamics[f"z{z_idx}"][term] = coefficient_value
                    ode_equations[f"z{z_idx}"] += f"{coefficient_value:.4f} * {term} + "

            # Remove trailing '+' in the ODE equation
            ode_equations[f"z{z_idx}"] = ode_equations[f"z{z_idx}"].strip(" + ")

        return {'dynamics': dynamics, 'ode_equations': ode_equations}

# Autoencoder with SINDy
class Autoencoder(nn.Module):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(args.input_dim, args.latent_dim)
        self.decoder = Decoder(args.input_dim, args.latent_dim)
        self.sindy_model = SINDyModel(args)

        self.sup_latent = args.supervised
        self.order = args.order
        self.static = args.static

    def forward(self, x, dx, ddx=None, treatment=None, size=None):
        # Encoder forward pass
        z = self.encoder(x)  
        # Decoder forward pass
        x_hat = self.decoder(z)

        # Reconstruction loss
        recon_loss = nn.MSELoss()(x.view(x.shape[0], -1), x_hat)

        # Initialize losses
        loss_potential_outcome = None
        loss_treatment = None

        if self.sup_latent:
            # Enforce one dim represents the potential outcome and another represents treatment
            loss_potential_outcome = nn.MSELoss()(z[:, 0], size)
            # loss_treatment = nn.BCEWithLogitsLoss()(z[:, -1], treatment.float())
    
        # Compute the true derivative of z_dot based wrt x_dot: dz/dx * dx/dt
        encoder_parameters = list(self.encoder.parameters())
        encoder_weight_list = [w for w in encoder_parameters if len(w.shape) == 2]
        encoder_biases_list = [b for b in encoder_parameters if len(b.shape) == 1]

        # Compute the approx. derivative of x_dot: d(DEC)/dz * z_dot_pred
        decoder_parameters = list(self.decoder.parameters())
        decoder_weight_list = [w for w in decoder_parameters if len(w.shape) == 2]
        decoder_biases_list = [b for b in decoder_parameters if len(b.shape) == 1]

        if self.order == 1:
            dz_true = self.compute_derivative(x.view(x.shape[0], -1), dx.view(dx.shape[0], -1), encoder_weight_list, encoder_biases_list, activation='sigmoid')
            dz_pred, sindy_l1_loss = self.sindy_model(z)

            if self.static:
                dz_dynamic = dz_true[:, 0].unsqueeze(1)
                dz_static = dz_true[:, 1].unsqueeze(1)

                sindy_z_loss = nn.MSELoss()(dz_dynamic, dz_pred)
                static_loss = torch.mean(torch.abs(dz_static))  # L1 regularization on static derivative

                dx_pred = self.compute_derivative_wrt_z0(z, dz_pred, decoder_weight_list, decoder_biases_list, activation='sigmoid')
                sindy_x_loss = nn.MSELoss()(dx.view(dx.shape[0], -1), dx_pred)
            else:
                sindy_z_loss = nn.MSELoss()(dz_true, dz_pred)
                dx_pred = self.compute_derivative(z, dz_pred, decoder_weight_list, decoder_biases_list, activation='sigmoid')
                sindy_x_loss = nn.MSELoss()(dx.view(dx.shape[0], -1), dx_pred)

        elif self.order == 2:
            dz_true, ddz_true = self.compute_derivative_order2(x.view(x.shape[0], -1), dx.view(dx.shape[0], -1), ddx.view(ddx.shape[0], -1), encoder_weight_list, encoder_biases_list, activation='sigmoid')
            ddz_pred, sindy_l1_loss = self.sindy_model(z, dz_true)

            sindy_z_loss = nn.MSELoss()(ddz_true, ddz_pred)

            dx_pred, ddx_pred = self.compute_derivative_order2(z, dz_true, ddz_pred, decoder_weight_list, decoder_biases_list, activation='sigmoid')
            sindy_x_loss = nn.MSELoss()(ddx.view(ddx.shape[0], -1), ddx_pred)

        # Combine all loss components
        return_dict = {
            'recon_loss': recon_loss.mean(),
            'sindy_x_loss': sindy_x_loss.mean(),
            'sindy_z_loss': sindy_z_loss.mean(),
            'sindy_l1_loss': sindy_l1_loss.mean()
        }

        if self.static:
            return_dict['static_loss'] = static_loss.mean()
        if loss_potential_outcome is not None:
            return_dict['potential_outcome_loss'] = loss_potential_outcome
        if loss_treatment is not None:
            return_dict['treatment_loss'] = loss_treatment

        return return_dict


    def compute_derivative_wrt_z0(self, z, dz_pred, weights, biases, activation='sigmoid'):
        """
        Compute the first-order derivatives of the output w.r.t. the 0th dimension of z,
        incorporating the predicted dynamics for z_0.

        Arguments:
            z - 2D torch tensor [batch_size, latent_dim], input to the network.
            dz_pred - 2D torch tensor [batch_size, 1], first-order derivatives of the 0th latent variable (z_0).
            weights - List of torch tensors containing the network weights.
            biases - List of torch tensors containing the network biases.
            activation - Activation function: 'elu', 'relu', 'sigmoid', or 'linear'.

        Returns:
            dx_pred - Torch tensor [batch_size, output_dim], first-order derivatives of the output w.r.t. z_0.
        """
        # Initialize dz as zero and set only the 0th dimension to dz_pred
        dz = torch.zeros_like(z)  # [batch_size, latent_dim]
        dz[:, 0] = dz_pred[:, 0]  # Incorporate dz_pred for the 0th dimension of z

        a = z  # Activation at the current layer

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

        return dz  # Return the derivative w.r.t. z_0


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


    # Derivative function
    def compute_derivative_order2(self, x, dx, ddx, weights, biases, activation='sigmoid'):
        a = x  # Input to the network
        dz = dx  # First order derivative (dx) at the input
        ddz = ddx  # Second order derivative (ddx) at the input

        # Loop through each layer
        for i in range(len(weights)):
            w = weights[i]  # Weight matrix
            b = biases[i]   # Bias vector
            z = torch.matmul(a, w.T) + b

            if activation == 'sigmoid' and i < len(weights) - 1:
                # Apply sigmoid activation
                a = torch.sigmoid(z)
                # First derivative of sigmoid
                gprime = a * (1 - a)
                # Second derivative of sigmoid
                gprime_2 = gprime * (1 - 2 * a)

                dz_prev = torch.matmul(dz, w.T)
                # Compute the first order derivative dz
                dz = gprime * dz_prev
                # Compute the second order derivative ddz
                ddz = gprime_2 * dz_prev**2 + gprime * torch.matmul(ddz, w.T)

            elif activation == 'relu' and i < len(weights) - 1:
                # Apply ReLU activation
                a = torch.relu(z)
                # First derivative of ReLU
                gprime = (z > 0).float()  # Derivative is 1 for z > 0, else 0
                # Second derivative of ReLU (zero everywhere except at 0)
                gprime_2 = torch.zeros_like(gprime)

                # Compute the first order derivative dz
                dz = gprime * torch.matmul(dz, w.T)
                # Compute the second order derivative ddz
                ddz = gprime_2 * dz_prev**2 + gprime * torch.matmul(ddz, w.T)

            else:
                # If no activation (for the last layer)
                a = z  # No activation applied to the last layer
                dz = torch.matmul(dz, w.T)
                ddz = torch.matmul(ddz, w.T)

            # Update dz_prev for the next layer
            dz_prev = dz
            
        return dz, ddz

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
        z = self.encoder(x)  # Initial latent state [shape: (batch_size, latent_dim)]
      
        # Prepare trajectory storage
        batch_size, latent_dim = z.size()
        z_traj = torch.zeros(batch_size, num_steps, latent_dim).to(z.device)

        # Iterate through time steps
        for step in range(num_steps):

            # Predict derivative (dz/dt) using SINDy layer
            z_dot_pred, _ = self.sindy_model(z)

            # Update latent state using Euler's method
            z = z + z_dot_pred * dt

            z_traj[:, step, :] = z  # Store the predicted next latent state

        return z_traj

    def get_recon(self, x ):
        """
        Reconstruct the input images 
        
        Args:
        - x: Input images (or data to reconstruct).
        - treatment: Observed treatment variable.
        - size: Observed potential outcome variable.

        Returns:
        - reconstructions: The reconstructed images from the decoder.
        """
        # Encode the input to get latent representations
        z = self.encoder(x)

        # Decode using the G-Constrained Decoder with observed attributes
        x_hat = self.decoder(z)
        
        return x_hat

    def get_ode(self, x, dx, ddx = None):
        # Encode the input to get the latent representation (z_loc, z_logscale)
        z = self.encoder(x)
        
        # Compute the true derivative of z_dot based wrt x_dot: dz/dx * dx/dt
        encoder_parameters = list(self.encoder.parameters())
        encoder_weight_list = [w for w in encoder_parameters if len(w.shape) == 2]
        encoder_biases_list = [b for b in encoder_parameters if len(b.shape) == 1]

        if self.order == 1:
            dz_true = self.compute_derivative(x.view(x.shape[0], -1), dx.view(dx.shape[0], -1), encoder_weight_list, encoder_biases_list, activation='sigmoid')
            dz_pred, sindy_l1_loss = self.sindy_model(z) 

            # Compute the dynamics using the SINDy model (get_dynamics expects z)
            dynamics_info = self.sindy_model.get_dynamics(z)
        
        elif self.order == 2:
            dz_true, ddz_true = self.compute_derivative_order2(x.view(x.shape[0], -1), dx.view(dx.shape[0], -1), ddx.view(ddx.shape[0], -1), encoder_weight_list, encoder_biases_list, activation='sigmoid') 
            ddz_pred, sindy_l1_loss = self.sindy_model(z, dz_true) 

            # Compute the dynamics using the SINDy model (get_dynamics expects z)
            dynamics_info = self.sindy_model.get_dynamics(z, dz_true)

        return dynamics_info

    def predict_v_hat(self, x):

        ### Encoder: Map input to latent space
        z = self.encoder(x)  # Latent mean and log-variance

        return z



if __name__ == '__main__':

    # Example Usage
    input_size = 784  # Input dimension
    latent_dim = 3  # Latent space dimension
    library_dim = 7

    autoencoder = Autoencoder(input_size, latent_dim, library_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)

    # Dummy data
    x = torch.rand(1024, input_size)  # number of timepoints, input size
    x_dot = torch.rand(1024, input_size)  # number of timepoints, input size
    treatment = torch.randint(0, 2, (1024, 1))  # Random treatment labels
    size = torch.rand(1024, 1)

    # Forward pass
    output = autoencoder(x, x_dot, treatment, size)
    # Print losses
    print(output)



