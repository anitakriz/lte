import torch
import torch.nn as nn
import numpy as np

######################### ResBlock ############################

class ResBlock(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, stride = 1, scale_factor = 2, upsample = False):
        super().__init__()
        
        self.upsample = upsample
        
        if upsample:
            self.up = nn.Upsample(scale_factor = scale_factor, mode = 'nearest')

        self.convblock = nn.Sequential( # main path
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

        if stride != 1 or in_channels != out_channels: # handle spatial size and varying channel dim.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU()
   

    def forward(self, x): 

        if self.upsample: # on decoder side
            x = self.up(x)

        out = self.convblock(x) # main path
        x = self.shortcut(x) # residual path 
        out += x # skip connect
        out = self.act(out)

        return out

######################### Encoder ############################

class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size = 3, stride = 1, padding = 1) # 1 -> 16 channels, 32^2 -> 32^2

        # 5 layers with sequence of resblock at each layer 
        self.resblocks1 = nn.Sequential( # 32^2 -> 16^2 spatial size
            ResBlock(16, 32, stride = 2), # downsample
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )

        self.resblocks2 = nn.Sequential(  # 16^2 -> 8^2 spatial size
            ResBlock(32, 64, stride = 2), # downsample
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.resblocks3 = nn.Sequential( # 8^2 -> 4^2 spatial size
            ResBlock(64, 128, stride = 2), # downsample
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        self.resblocks4 = nn.Sequential( # 4^2 -> 1^2 spatial size
            ResBlock(128, 256, stride = 4), # downsample
            ResBlock(256, 256),
            ResBlock(256, 256),
        )

        # putting all the layers together
        self.encoder = nn.Sequential(
            self.conv1,
            self.resblocks1,
            self.resblocks2,
            self.resblocks3,
            self.resblocks4,
            )

        # self.fc_attr = nn.Linear(attribute_dim, 1)  # Attribute embedding
        self.fc_mu = nn.Linear(256, 256)  # fc layer for mean
        self.fc_logscale = nn.Linear(256, 256)  # fc layer for log st.dev.
    
    def forward(self, x, treatment):
      
        treatment = treatment.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
        treatment = treatment.expand(-1, x.size(1), x.size(2))  # (batch_size, 1, H, W)
    
        # Concatenate treatment to the image
        x = torch.cat((x.unsqueeze(1), treatment.unsqueeze(1)), dim=1)  # Concatenate along the channel dimension (1 -> 2)
        x = self.encoder(x)
        x = x.view(x.size(0), -1) 
    
        mu = self.fc_mu(x)
        logscale = self.fc_logscale(x)

        return mu, logscale

######################### Decoder ############################

class Decoder(nn.Module):
    def __init__(self, num_channels,):
        super().__init__()

        self.fc = nn.Linear(num_channels, num_channels) 
        self.resblocks1 = ResBlock(num_channels, 128, upsample = True, scale_factor = 4)
        self.resblocks2 = nn.Sequential(
            ResBlock(128, 64, upsample = True),
            ResBlock(64, 64)
        )
        self.resblocks3 = nn.Sequential(
            ResBlock(64, 32, upsample = True),
            ResBlock(32, 32),
            ResBlock(32, 32)
        )
        self.resblocks4 = nn.Sequential(
            ResBlock(32, 16, upsample = True),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16)
        )

        self.decoder = nn.Sequential(
            self.resblocks1,
            self.resblocks2,
            self.resblocks3,
            self.resblocks4,
            )
        
        self.final_loc = nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1)
        self.final_logscale = nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1)
        
        
    def forward(self, z):

        x = self.fc(z)
        x = x.view(x.size(0), x.size(1), 1, 1)  
        x = self.decoder(x)
        loc = self.final_loc(x)
        logscale = self.final_logscale(x)
        
        return loc, logscale


######################### Mechanism Network ############################
class MechanismNetwork(nn.Module):

    def __init__(self, dim_ui, dim_pa = 0, output_type = "binary"):
        super().__init__()

        self.output_type = output_type # Binary, Categorical, Continuous
        self.mechanism = nn.Sequential(
                nn.Linear(dim_ui + dim_pa, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, eps, pa=None):

        x = torch.cat((eps, pa), dim=1) if pa is not None else eps
        output = self.mechanism(x)

        if self.output_type == "binary":
            output = torch.sigmoid(output)  
        elif self.output_type == "continuous":
            output = torch.relu(output)
        elif self.output_type == "categorical":
            output = torch.softmax(output, dim=1)

        return output


######################### G-Constrained Decoder ############################
class GConstrainedDecoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.enc_channel = args.latent_ch
        self.decoder = Decoder(num_channels = self.enc_channel + 3 ) #TODO may have to not hardcode 3 if we inc size of z[2]

        # mechanisms for prediction of the attributes
        self.f_T = MechanismNetwork(self.enc_channel//3, output_type = "binary") 
        self.f_C = MechanismNetwork(self.enc_channel//3, output_type = "continuous")
        self.f_PO = MechanismNetwork(self.enc_channel//3, dim_pa = 2, output_type = "continuous")

    def forward(self, z_loc, z_logscale, obs_attr = None, t = None, do = None):
        
        # Split the mean vector into exo. noise terms for (1) Treatment (2) Covariates (3) Potential Outcome
        eps = torch.split(z_loc, self.enc_channel // 3, dim = 1) 

        # Obtain individual noise variables for each term
        eps_1 = eps[0] # corr. to Treatment
        eps_2 = eps[1] # corr. to Covariates
        eps_3 = eps[2] # corr. to Potential Outcome

        # Predict Treatment from eps_1
        T_hat = self.f_T(eps_1)
        # Predict Covariates from eps_2
        C_hat = self.f_C(eps_2) #TODO, larger pred for covarites
        # Predict Potential Outcomes from eps_3, the Treatment, and Covariates
        PO_hat = self.f_PO(eps_3, pa = torch.cat((T_hat, C_hat), dim = 1))

        #collect all attributes for SINDy training
        v_hat = torch.cat((T_hat, C_hat, PO_hat), dim = 1)
        if self.training: 
            grad_v_hat = torch.autograd.grad(
                v_hat, z_loc, grad_outputs=torch.ones_like(v_hat), create_graph=True
            )[0]
        else:
            # Explicitly enable gradient computation for evaluation
            with torch.enable_grad():
                grad_v_hat = torch.autograd.grad(
                    v_hat, z_loc, grad_outputs=torch.ones_like(v_hat), create_graph=True
                )[0]
        # Compute the derivative of v_hat with respect to z_loc
     
        # Now grad_v_hat has the shape [bs, latent_dim] and represents the gradient of v_hat w.r.t z_loc

        # We want v_hat_dot to have the shape [500, 3], so we will sum/average along the feature dimension for each part of v_hat
        # Assuming feature_size is the number of features in T_hat, C_hat, or PO_hat (which will be the same for all three)

        # Step 1: Reshape the gradients to separate them by T_hat, C_hat, and PO_hat
        grad_T_hat = grad_v_hat[:, :T_hat.shape[1]]
        grad_C_hat = grad_v_hat[:, T_hat.shape[1]:T_hat.shape[1] + C_hat.shape[1]]
        grad_PO_hat = grad_v_hat[:, T_hat.shape[1] + C_hat.shape[1]:]

        # Step 2: Combine the gradients for each part to form a [500, 3] shape
        v_hat_dot = torch.stack([
            grad_T_hat.mean(dim=1),  # Derivative of T_hat w.r.t z_loc
            grad_C_hat.mean(dim=1),  # Derivative of C_hat w.r.t z_loc
            grad_PO_hat.mean(dim=1),  # Derivative of PO_hat w.r.t z_loc
        ], dim=1)

        if t is not None:
            z = reparameterize(z_loc, z_logscale + torch.log(torch.tensor(float(t), device=z_logscale.device)))  # Obtain Sampled Latent for Decoder Step
        else:
            z = reparameterize(z_loc, z_logscale) # Obtain Sampled Latent for Decoder Step

        if obs_attr is not None: #if we are given the Treatment and Potential Outcome in Advance (ie during training)

            T = obs_attr['Treatment'].unsqueeze(1) #Observed Treatment value
            C = C_hat # NOTE: assume we NEVER observe the covariate and it must be learned...not sure if htis is a godo strategy
            PO = obs_attr['Potential Outcome'].unsqueeze(1) 
            
            obs_v = torch.cat((T, C, PO), dim = 1)
            x_loc, x_logscale = self.decoder(torch.cat((z, obs_v), dim = 1))
            x_recon, _ = self.decoder(torch.cat((z_loc, obs_v), dim = 1))

        
        #if not observed treatmeent/ potential outcome, use predicted values
        else:
            x_loc, x_logscale = self.decoder(torch.cat((z, v_hat), dim = 1))
            x_recon, _ = self.decoder(torch.cat((z_loc, v_hat), dim = 1))

        return x_loc, x_logscale, x_recon, v_hat, obs_v, v_hat_dot


######################### SINDy Model: ODE Discovery ############################
class SINDy(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.library_dim = args.library_dim # number of terms in its library
        self.coefficients = nn.Parameter(torch.randn(args.library_dim, 3))  
        self.register_buffer('coefficient_mask', torch.ones(args.library_dim, 3))  #TODO: why is this a buffer - check if correct

        self.threshold = 1e-3 # NOTE: hyperparemter! sequential thresholding

    def forward(self, z):

        # Compute Library (based on z, what are the values of terms)
        theta = self.compute_library(z) 
        # Apply mask ==> enforces sparsity
        masked_coefficients = self.coefficients * self.coefficient_mask
        # Compute z_dot = Θ(z^T)Ξ
        z_dot_pred = torch.matmul(theta, masked_coefficients)
        l1_reg_loss = torch.mean(torch.abs(masked_coefficients))

        return z_dot_pred, l1_reg_loss
    
    def compute_library(self, z):
        #NOTE: this was written assuming that covariate is one value, not a vector!
        treatment = z[:, 0:1]
        covariate = z[:, 1:2]
        outcome = z[:, 2:3]

        # Build the library NOTE: this should be done based on your "upper bound" best guesstimate
        theta = torch.cat([
            torch.ones_like(outcome),
            outcome,
            outcome**2,
            outcome * covariate,
            outcome * treatment,
            outcome**2 * covariate,
            outcome**2 * treatment
        ], dim = 1)

        return theta

    def update_mask(self):
        # Apply sequential thresholding. 
 
        with torch.no_grad():
            self.coefficient_mask = (torch.abs(self.coefficients) > self.threshold).float()
    
    def get_dynamics(self, z=None):
        """
        Return the learned ODE (dynamics) by returning the coefficients with their corresponding library terms.
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

######################### VAE ############################
class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Models
        self.encoder = Encoder()
        self.gconstrained_decoder = GConstrainedDecoder(args)
        self.likelihood = DGaussNet()
        self.sindy_model = SINDy(args)

        # Random Variables
        self.register_buffer("p_loc", torch.zeros(1, args.latent_ch))
        self.register_buffer("p_logscale", torch.ones(1, args.latent_ch))
        
        # Hyperparameters
        self.beta = args.beta
        self.bs = args.bs
        self.t = args.timepoints
        self.delta_t = args.delta_t

    def forward(self, x, x_dot, treatment, size):
        """
        Forward pass of the Variational Autoencoder.
        
        Args:
        - x: Input data (e.g., images or trajectories).
        - x_dot: Observed derivatives of x.
        - treatment: Treatment variables associated with x.
        - size: Contextual information for reconstruction.
        
        Returns:
        - A dictionary of losses and metrics.
        """

        ### Encoder: Map input to latent space
        x.requires_grad_(True)
        z_loc, z_logscale = self.encoder(x, treatment)  # Latent mean and log-variance

        # Compute the gradient of the latent mean w.r.t. input
        z_loc_dot = torch.autograd.grad(
            z_loc.sum(), x, retain_graph=True, create_graph=True
        )[0]
        z_loc.requires_grad_(True)  # Allow gradient flow for downstream calculations

        ### Decoder: Reconstruct input and predict dynamics
        x_loc, x_logscale, x_recon, v_hat, obs_v, v_hat_dot = self.gconstrained_decoder(
            z_loc, z_logscale, obs_attr={'Treatment': treatment, 'Potential Outcome': size}
        )

        # Losses on treatment and potential outcomes
        treatment_loss = nn.BCEWithLogitsLoss()(v_hat[:, 0], obs_v[:, 0])  # Binary cross-entropy loss
        outcome_loss = nn.MSELoss()(v_hat[:, -1], obs_v[:, -1])  # Mean squared error loss

        ### Derivative Matching: Reconstruct x_dot
        # Reshape reconstructed trajectories for time steps
        x_recon = x_recon.view(self.bs // self.t, self.t, *x_recon.shape[2:])
        x_dot_pred = torch.diff(x_recon, dim=1) / self.delta_t  # Predicted derivatives

        # Remove initial derivatives that were initialized as 0
        remove_indices = torch.arange(0, self.bs, self.t)
        mask = torch.ones(self.bs, dtype=torch.bool)
        mask[remove_indices] = False
        x_dot = x_dot[mask].view(self.bs // self.t, self.t - 1, *x_recon.shape[2:])
        sindy_x_loss = nn.MSELoss()(x_dot, x_dot_pred)  # SINDy loss on image-level derivatives

        ### Likelihood and KL Divergence
        nll = self.likelihood.nll(x_loc, x_logscale, x)  # Negative log-likelihood
        kl = gaussian_kl(
            q_loc=z_loc, q_logscale=z_logscale,
            p_loc=self.p_loc.expand(x.size(0), -1),
            p_logscale=self.p_logscale.expand(x.size(0), -1)
        )
        kl = kl.sum(dim=-1) / np.prod(x.shape[1:])  # Normalize KL by input dimensions
        elbo = nll.mean() + self.beta * kl.mean()  # Evidence lower bound (ELBO)

        ### SINDy: Discover ODEs for latent variables
        v_hat_dot_pred, sindy_l1_loss = self.sindy_model(v_hat)  # Predict latent derivatives
        sindy_v_loss = nn.MSELoss()(v_hat_dot, v_hat_dot_pred)  # SINDy loss on latent-level derivatives

        ### Return Losses 
        return {
            'potential_outcome_loss': outcome_loss,
            'treatment_loss': treatment_loss,
            'kl': kl.mean(),
            'nll': nll.mean(),
            'elbo': elbo,
            'sindy_x_loss': sindy_x_loss.mean(),
            'sindy_v_loss': sindy_v_loss.mean(),
            'sindy_l1_loss': sindy_l1_loss.mean()
        }
    
    def get_recon(self, x, treatment, size):
        """
        Reconstruct the input images based on their latent representations.
        
        Args:
        - x: Input images (or data to reconstruct).
        - treatment: Observed treatment variable.
        - size: Observed potential outcome variable.

        Returns:
        - reconstructions: The reconstructed images from the decoder.
        """
        # Encode the input to get latent representations
        z_loc, z_logscale = self.encoder(x, treatment)

        # Decode using the G-Constrained Decoder with observed attributes
        _, _, x_recon, _, _, _ = self.gconstrained_decoder(
            z_loc, z_logscale, obs_attr={'Treatment': treatment, 'Potential Outcome': size}
        )
        
        return x_recon

    def sample(self, treatment, size, t):
        
        self.p_loc = self.p_loc.clone().detach().requires_grad_(True)
        self.p_logscale = self.p_logscale.clone().detach().requires_grad_(True)

        x_loc, x_logscale, _, _, _, _ = self.gconstrained_decoder(
            self.p_loc.repeat(treatment.shape[0], 1), self.p_logscale.repeat(treatment.shape[0], 1), 
            obs_attr={'Treatment': treatment, 'Potential Outcome': size}, t = t
        )

        return self.likelihood.sample(x_loc, x_logscale, return_loc = True)
    
    def get_ode(self, x, treatment, size):
        # Encode the input to get the latent representation (z_loc, z_logscale)
        z_loc, z_logscale = self.encoder(x, treatment)

        # Decoder: Reconstruct the input and predict the dynamics (v_hat)
        _, _, _, v_hat, _, _ = self.gconstrained_decoder(
            z_loc, z_logscale, obs_attr={'Treatment': treatment, 'Potential Outcome': size}
        )

        # Compute the dynamics using the SINDy model (get_dynamics expects z)
        dynamics_info = self.sindy_model.get_dynamics(v_hat)

        return dynamics_info
    
    def predict_next_latents(self, x, treatment, size, num_steps, dt=.05):
        """
        Predict the latent trajectories using learned ODEs (SINDy).

        Arguments:
            x (tensor): The input data at the initial time step [shape: (batch_size, ...)].
            treatment (tensor): Treatment or condition applied [shape: (batch_size, treatment_dim)].
            size (tensor): Contextual information (e.g., potential outcomes).
            num_steps (int): Number of time steps to predict.
            dt (float): Time step size (default: 1.0).

        Returns:
            z_traj (tensor): Predicted latent trajectory [shape: (batch_size, num_steps, latent_dim)].
        """

        # Encode the input to get the latent representation (z_loc, z_logscale)
        z_loc, z_logscale = self.encoder(x, treatment)

        # Decode the input and predict the dynamics (v_hat)
        _, _, _, v_hat, _, _ = self.gconstrained_decoder(
            z_loc, z_logscale, obs_attr={'Treatment': treatment, 'Potential Outcome': size}
        )

        # Prepare trajectory storage
        batch_size, latent_dim = v_hat.size()
        v_traj = torch.zeros(batch_size, num_steps, latent_dim).to(v_hat.device)

        # Initial latent state
        v_traj[:, 0, :] = v_hat

        # Iterate through time steps to predict the latent trajectory
        for step in range(1, num_steps):
            # Predict the derivative of the latent state (v_hat_dot_pred) using the SINDy model
            v_hat_dot_pred, _ = self.sindy_model(v_hat)

            # Update the latent state using Euler's method (or other integration methods)
            v_hat = v_hat + v_hat_dot_pred * dt  # Numerical integration (Euler method)

            # Store the predicted latent state for the current time step
            v_traj[:, step, :] = v_hat

        return v_traj

    def predict_v_hat(self, x, treatment, size):

        ### Encoder: Map input to latent space
        z_loc, z_logscale = self.encoder(x, treatment)  # Latent mean and log-variance

        ### Decoder: Reconstruct input and predict dynamics
        _, _,  _, v_hat, obs_v, _ = self.gconstrained_decoder(
            z_loc, z_logscale, obs_attr={'Treatment': treatment, 'Potential Outcome': size}
        )

        return v_hat, obs_v

######################### KL-Divergence Calc. ############################
def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    return (
        -0.5
        + p_logscale
        - q_logscale
        + 0.5
        * (q_logscale.exp().pow(2) + (q_loc - p_loc).pow(2))
        / p_logscale.exp().pow(2)
    )
######################### Likelihood Calc. ############################
       
class DGaussNet(nn.Module):
    def __init__(self):
        super().__init__()

    def approx_cdf(self, x):
        """Approximate CDF for Gaussian using a tanh-based formulation."""
        return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def nll(self, loc, logscale, x):
        """
        Compute the negative log-likelihood for normalized images in [0, 1].

        Args:
            loc (Tensor): Mean (B, C, H, W).
            logscale (Tensor): Log-std (B, C, H, W).
            x (Tensor): Ground truth image in [0, 1].

        Returns:
            Tensor: Negative log-likelihood per batch.
        """
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale.clamp(min=-9))  # Clamp logscale to avoid instability

        # Adjust bounds for the [0, 1] range
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)

        # Compute log-probabilities
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min

        # Handle edge cases for boundaries [0, 1]
        log_probs = torch.where(
            x < 0.001,  # Near the lower boundary
            log_cdf_plus,
            torch.where(
                x > 0.999,  # Near the upper boundary
                log_one_minus_cdf_min,
                torch.log(cdf_delta.clamp(min=1e-12))  # Normal case
            ),
        )

        return -1.0 * log_probs.mean(dim=(1, 2, 3))

    def sample(self, loc, logscale, return_loc=False, t=None):
        """
        Sample from the Gaussian distribution parameterized by loc and logscale.

        Args:
            loc (Tensor): Predicted mean (B, C, H, W).
            logscale (Tensor): Predicted log-std (B, C, H, W).
            return_loc (bool): If True, return loc directly without sampling.
            t (float, optional): Temperature adjustment for sampling.

        Returns:
            Tuple[Tensor, Tensor]: Sampled image clamped to [0, 1] and standard deviation.
        """
        if return_loc:
            x = loc  # Deterministic sampling: use mean directly
        else:
            logscale = logscale.clamp(min=-9)  # Clamp logscale to avoid instability
            
            # Apply temperature adjustment if provided
            if t is not None:
                logscale = logscale + torch.tensor(t).to(loc.device).log()
            
            # Sample from Gaussian
            x = sample_gaussian(loc, logscale)
        
        # Clamp samples to [0, 1] range
        x = torch.clamp(x, min=0.0, max=1.0)
        return x, torch.exp(logscale)  # Return samples and standard deviation




def reparameterize(mu, log_var):
    """
    Perform the reparameterization trick: z = mu + epsilon * std.
    """
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)  
    z = mu + epsilon * std
    return z