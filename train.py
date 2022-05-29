import numpy as np
import torch
from tqdm import tqdm

from models.logistic_regression import train_lr


def membership_test(model, pois_sample_x, pois_sample_y):
  """Membership inference - detect poisoning."""

  with torch.no_grad():
    input = np.concatenate([pois_sample_x, np.zeros_like(pois_sample_x)])

    probs = model(torch.Tensor(input))
    probs = probs.numpy() 
    
    score = np.multiply(probs[0, :] - probs[1, :], pois_sample_y).sum()

  return score

def train_and_score(args, poisoned_data, poisoned_sample, model_name, epsilon):

    training_algorithms = {
      "lr" : train_lr
    }
    
    (pois_x1, pois_y), (pois_x2, unpois_y) = poisoned_data

    pois_sample_x, pois_sample_y = poisoned_sample

    poison_scores = []
    unpois_scores = []

    for _ in tqdm(range(args.num_trials), desc="Auditing {} (Total Trials)".format(model_name)):
      
      # Train Model, Test for Membership
      model = training_algorithms[model_name](args, (pois_x1, pois_y))
      p_score = membership_test(model, pois_sample_x, pois_sample_y)
      poison_scores.append(p_score)

      model = training_algorithms[model_name](args, (pois_x2, unpois_y))
      u_score = membership_test(model, pois_sample_x, pois_sample_y)
      unpois_scores.append(u_score)
    
      if args.debug:
        print("Poisoned Score:", p_score)
        print("Unpoisoned Score:", u_score)

    return poison_scores, unpois_scores
