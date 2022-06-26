import numpy as np
from statsmodels.stats import proportion

def compute_results(poison_scores, unpois_scores, pois_ct,
                    alpha=0.05):
  """
  Searches over thresholds for the best epsilon lower bound and accuracy.
  poison_scores: list of scores from poisoned models
  unpois_scores: list of scores from unpoisoned models
  pois_ct: number of poison points
  alpha: confidence parameter
  threshold: if None, search over all thresholds, else use given threshold
  """

  all_thresholds = np.unique(poison_scores + unpois_scores)

  poison_arr = np.array(poison_scores)
  unpois_arr = np.array(unpois_scores)

  best_threshold, best_epsilon, best_acc = all_thresholds[0], 0, 0

  # Find best threshold
  for thresh in all_thresholds:
    epsilon, acc = compute_epsilon_and_acc(poison_arr, unpois_arr, thresh,
                                           alpha, pois_ct)
    if epsilon > best_epsilon:
      best_epsilon, best_threshold = epsilon, thresh
    best_acc = max(best_acc, acc)

  best_epsilon, best_acc = compute_epsilon_and_acc(poison_arr, unpois_arr, best_threshold,
                                           alpha, pois_ct)
  return best_threshold, best_epsilon, best_acc


def compute_epsilon_and_acc(poison_arr, unpois_arr, threshold, alpha, pois_ct):
  """For a given threshold, compute epsilon and accuracy."""
  poison_ct = (poison_arr > threshold).sum()
  unpois_ct = (unpois_arr > threshold).sum()

  # clopper_pearson uses alpha/2 budget on upper and lower
  # so total budget will be 2*alpha/2 = alpha
  p1, _ = proportion.proportion_confint(poison_ct, poison_arr.size,
                                        alpha, method='beta')
  _, p0 = proportion.proportion_confint(unpois_ct, unpois_arr.size,
                                        alpha, method='beta')
  # p0, p1 = p1, p0

  if (p1 <= 1e-5) or (p0 >= 1 - 1e-5):  # divide by zero issues
    return 0, 0

  if (p0 + p1) > 1:  # see Appendix A
    p0, p1 = (1 - p1), (1 - p0)

  epsilon = np.log(p1 / p0) / pois_ct
  acc = (p1 + (1 - p0)) / 2  # this is not necessarily the best accuracy

  return epsilon, acc