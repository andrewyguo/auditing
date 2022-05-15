import numpy as np
import torch
from tqdm import tqdm

from .logistic_regression import train_lr


def membership_test(model, pois_x, pois_y):
  """Membership inference - detect poisoning."""
  probs = model(torch.Tensor(np.concatenate([pois_x, np.zeros_like(pois_x)])))
  probs = probs.numpy()

  return np.multiply(probs[0, :] - probs[1, :], pois_y).sum()

def train_and_score(args, poisoned_data, poisoned_sample, model_name, epsilon):

    training_algorithms = {
      "lr" : train_lr
    }
    
    (pois_x1, pois_y1), (pois_x2, pois_y2) = poisoned_data

    sample_x, sample_y = poisoned_sample

    poison_scores = []
    unpois_scores = []

    for i in tqdm(range(args.num_trials), desc="Auditing {} (Total Trials)".format(model_name)):
      
      # Train Model, Test for Membership
      model = training_algorithms[model_name](args, (pois_x1, pois_y1))
      poison_scores.append(membership_test(model, pois_x1, pois_y1))

      model = training_algorithms[model_name](args, (pois_x2, pois_y2))
      unpois_scores.append(membership_test(model, pois_x2, pois_y2))
    
    return poison_scores, unpois_scores
