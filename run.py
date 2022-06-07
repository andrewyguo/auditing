import argparse
import time
from datetime import datetime
import numpy as np

# Local Imports
import attack
import audit
import audit_utils
import datasets
import train

parser = argparse.ArgumentParser(description='Handle arguments for running auditing attacks')

parser.add_argument('--alpha', default=100, help='1-confidence', type=int)
parser.add_argument('--attack_l2_norm', default=10, help='Size of poisoning data', type=int)
parser.add_argument('--attack_type', default="backdoor", help='Type of attack to be used. Choose one from: [clip_aware, backdoor]', type=str)
parser.add_argument('--batch_size', default=250, help='Batch size', type=int)
parser.add_argument('--dataset', default="fashion_mnist", help='Dataset to use when training models', type=str)
parser.add_argument('--epochs', default=32, help='Number of epochs used in training', type=int)
parser.add_argument('--eps_vals', default=[1], help='Analysis epsilon values to use to train models', nargs="+", type=int)
parser.add_argument('--l2_norm_clip', default=1.0, help='Clipping norm', type=float)
parser.add_argument('--learning_rate', default=0.15, help='Learning rate for training', type=int)
parser.add_argument('--load_weights', default=False, help='if True, use weights saved in init_weights.h5', type=bool)
parser.add_argument('--microbatches', default=250, help='Number of microbatches (must evenly divide batch_size)', type=int)
parser.add_argument('--seed', default=None, help='Seed for np.random', type=int)
parser.add_argument('--models', default=["lr"], help='Models to audit. Choose multiple options from: ', nargs="+", type=str)
parser.add_argument('--noise_multiplier', default=1.1, help='Ratio of the standard deviation to the clipping norm', type=float)
parser.add_argument('--num_trials', default=100, help='Number of trials for auditing', type=int)
parser.add_argument('--output_file', help='File to write output to', type=str)
parser.add_argument('--pois_ct', default=[1], help='Number of poisoning points. Can specify multiple values.', nargs="+", type=int)
parser.add_argument('--debug', action='store_true', help="Enable debug prints to console.")

if __name__ == '__main__':
    args = parser.parse_args()
    
    output_path = "output/{}".format(args.output_file) if args.output_file is not None else "output/auditing_output{}".format(datetime.now().strftime("_%Y_%m_%d_%H"))
    
    if args.seed is not None:
        np.random.seed(args.seed)

    # Get Training Data
    (train_x, train_y) = datasets.get_data(args.dataset)

    # Poison Data
    poisoned_data = attack.poison_data(train_x, train_y, args)
    
    for model_name in args.models: 
        for epsilon in args.eps_vals:
            for pois_ct in args.pois_ct:
                start_time = time.time()

                # Train Model and Infer Membership 
                poison_scores, unpois_scores = train.train_and_score(args, poisoned_data[pois_ct], poisoned_data["pois"], model_name, epsilon, poisoned_data["unpois"])

                # Audit and Save Results 
                results = audit.compute_results(poison_scores, unpois_scores, pois_ct)
                # results = (a, b, c, d)
                info = (model_name, epsilon) 
                audit_utils.save_results(results, output_path, start_time, args, info)
