import argparse

# python train.py --dataset_dir /usr/local/data/anitakriz/ode/lte/dot/data/untreated/ --save_dir /usr/local/data/anitakriz/ode/lte/dot/exps/dot_untreated_2/ --device 0 --dataset DOT --hps dot_untreated

# Assume zeros_10 is an object with parameters as specified in your dataset (e.g., zeros_10.delta_t, etc.)
pendulum = type('', (), {})()  # Create an empty object for storing these values
pendulum.start_sindy = 1 
pendulum.delta_t = .02
pendulum.input_dim = 51
pendulum.latent_dim = 1
pendulum.num_ode = 1
pendulum.poly_order = 3
pendulum.order = 2
pendulum.include_sine = True
pendulum.supervised = False
pendulum.lr = 1e-4
pendulum.threshold = .1
pendulum.lambda_1 = 5e-4
pendulum.lambda_2 = 5e-5
pendulum.lambda_3 = 1e-5

# Assume zeros_10 is an object with parameters as specified in your dataset (e.g., zeros_10.delta_t, etc.)
zeros_10 = type('', (), {})()  # Create an empty object for storing these values
zeros_10.start_sindy = 1 
zeros_10.delta_t = .1
zeros_10.input_dim = 128
zeros_10.latent_dim = 1
zeros_10.num_ode = 1
zeros_10.poly_order = 3
zeros_10.order = 1
zeros_10.supervised = False
zeros_10.lr = 1e-4
zeros_10.threshold = .1
zeros_10.lambda_1 = 5e-4
zeros_10.lambda_2 = 5e-5
zeros_10.lambda_3 = 0


# Assume zeros_10 is an object with parameters as specified in your dataset (e.g., zeros_10.delta_t, etc.)
dot_untreated = type('', (), {})()  # Create an empty object for storing these values
dot_untreated.start_sindy = 1 
dot_untreated.delta_t = .02
dot_untreated.input_dim = 51
dot_untreated.latent_dim = 1
dot_untreated.num_ode = 1
dot_untreated.poly_order = 3
dot_untreated.order = 1
dot_untreated.supervised = False
dot_untreated.lr = 1e-4
dot_untreated.threshold = .1
dot_untreated.lambda_1 = 5e-4
dot_untreated.lambda_2 = 5e-5
dot_untreated.lambda_3 = 1e-6
dot_untreated.lambda_4 = 1

def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--hps', type=str, required=True, help='HPs to use')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset type in use. Options: (MNIST_UNTREATED, MNIST_TE, Pendulum, dot)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to where npz files of dataset are stored')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to where you want to save experiment')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--treatment', action='store_true', help='Treatment input so use SINDY-c')

    # Training arguments
    parser.add_argument('--num_models', type=int, default=10, help='Num Models')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--refinement_epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--delta_t', type=float, default=.01, help='Change in t between timepoints')
    parser.add_argument('--pred_timepoints', type=int, default=5, help='Number of timepoints to predict in the future')
    parser.add_argument('--input_dim', type=int, default=51, help='Input dimension')
    parser.add_argument('--latent_ch', type=int, default=256, help='Latent channels of Encoder')
    parser.add_argument('--latent_dim', type=int, default=1, help='Latent dimension')
    parser.add_argument('--num_ode', type=int, default=1, help='Number of ODE')
    parser.add_argument('--poly_order', type=int, default=3, help='Order of terms in library')
    parser.add_argument('--include_sine', action='store_true', help='Include sine in library')
    parser.add_argument('--order', type=int, default=1, help='Order of derivative')
    parser.add_argument('--supervised', action='store_true', help='Add supervised loss in latent space')
    parser.add_argument('--sindy_enabled', action='store_true', help='Add supervised loss in latent space')
    parser.add_argument('--threshold', type=float, default=.01, help='Threshold when masking coefficients')
    parser.add_argument("--viz_freq", type=int, default=100, help="Epochs per visualization.")

    # Loss weights
    parser.add_argument('--lambda_1', type=float, default=5e-4, help='Weight on x-level derivative loss')
    parser.add_argument('--lambda_2', type=float, default=5e-5, help='Weight on z-level derivative loss')
    parser.add_argument('--lambda_3', type=float, default=1e-6, help='Weight on regularization loss')
    parser.add_argument('--lambda_4', type=float, default= 1e-6, help='Weight on supervised loss')
    parser.add_argument('--beta', type=float, default=.5, help='Beta term in ELBO')

    # Evaluation and checkpoints
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency of evaluation')
    parser.add_argument('--best_loss', type=float, default=float('inf'), help='Initial best loss')

    # Parse arguments
    args = parser.parse_args()

    if args.hps == "zeros_10":
        specific_args = zeros_10  # Assign the pre-defined object

    elif args.hps == "dot_untreated":
        specific_args =  dot_untreated

    elif args.hps == "pendulum":
        specific_args =  pendulum
    # Overwrite the default arguments with zeros_10 values
    for key, value in vars(specific_args).items():
        setattr(args, key, value)  # Use setattr to assign values dynamically

    return args
