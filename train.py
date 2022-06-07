import numpy as np
import torch
from tqdm import tqdm

from models.logistic_regression import train_lr, find_loss_lr
from attacks.backdoor import BackDoorAttack
from attacks.clip_aware import ClipAwareAttack

def train_and_score(args, poisoned_data, poisoned_sample, model_name, epsilon, unpoisoned_sample):

    training_algorithms = {
      "lr" : train_lr
    }

    membership_tests = {
      "backdoor" : BackDoorAttack.membership_test,
      "clip_aware" : ClipAwareAttack.membership_test
    }

    membership_test = membership_tests[args.attack_type]
    
    (pois_x1, pois_y), (pois_x2, unpois_y) = poisoned_data

    pois_sample_x, pois_sample_y = poisoned_sample
    _, unpois_sample_y = unpoisoned_sample

    poison_scores = []
    unpois_scores = []

    for _ in tqdm(range(args.num_trials), desc="Auditing {} (Total Trials)".format(model_name), colour="green"):
      
      # Train Model, Test for Membership
      p_model = training_algorithms[model_name](args, (pois_x1, pois_y))
      p_score = membership_test(p_model, pois_sample_x, pois_sample_y, args)
      poison_scores.append(p_score)

      u_model = training_algorithms[model_name](args, (pois_x2, unpois_y))
      u_score = membership_test(u_model, pois_sample_x, unpois_sample_y, args)
      unpois_scores.append(u_score)

      if args.debug: 
        # Compare Loss on Poisoned and Unpoisoned Sample 
        find_loss_lr(pois_sample_x, pois_sample_y, p_model, poisoned=True)
        find_loss_lr(pois_sample_x, unpois_sample_y, u_model, poisoned=False)

        print("Poisoned Score: {:.3f}".format(p_score))
        print("Unpoisoned Score: {:.3f}".format(u_score))

    return poison_scores, unpois_scores
