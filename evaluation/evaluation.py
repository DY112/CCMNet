"""
This code is modified from the C5 repository. [https://github.com/mahmoudnafifi/C5]
"""

import numpy as np
from scipy.io import loadmat as loadmat
from os.path import join
import argparse

def error_metrics(errors):
  percentiles = np.percentile(errors, [25, 50, 75, 95])
  mean = np.mean(errors)
  median = percentiles[1]
  tri = np.sum(percentiles[0:3] * np.array([1, 2, 1])) / 4
  b25 = np.mean(errors[np.argwhere(np.squeeze(errors) <= percentiles[0])])
  w25 = np.mean(errors[np.argwhere(np.squeeze(errors) >= percentiles[2])])
  w05 = np.mean(errors[np.argwhere(np.squeeze(errors) >= percentiles[3])])
  max = np.max(errors)

  return {
      'mean': mean,
      'median': median,
      'b25': b25,
      'w25': w25,
      'w05': w05,
      'max': max,
      'tri': tri
  }


def ang_error(a, b):
  a = a / np.expand_dims(np.sqrt(np.sum(a * a, axis=1)), axis=-1)
  b = b / np.expand_dims(np.sqrt(np.sum(b * b, axis=1)), axis=-1)
  cossim = np.sum(a * b, axis=-1)
  cossim[cossim > 1] = 1
  cossim[cossim < 0] = 0
  return (180 / np.pi) * np.arccos(cossim)


def test_evaluation(model_name):

  mean_errors = []
  median_errors = []
  best_errors = []
  worst_errors = []
  worst_errors_05 = []
  tri_errors = []
  max_errors = []

  gt = loadmat(join(model_name, f'gt.mat'))['gt']
  predicted = loadmat(join(model_name,
                            f'results.mat'))['predicted']

  error = ang_error(predicted, gt)
  metrics = error_metrics(error)
  mean_errors.append(metrics['mean'])
  median_errors.append(metrics['median'])
  tri_errors.append(metrics['tri'])
  best_errors.append(metrics['b25'])
  worst_errors.append(metrics['w25'])
  worst_errors_05.append(metrics['w05'])
  max_errors.append(metrics['max'])

  # print dataset_name
  print(f'{model_name}')

  print('Mean: %0.2f - std:%0.2f' %
        (np.mean(mean_errors), np.std(np.array(mean_errors))))
  print('Median: %0.2f - std:%0.2f' %
        (np.mean(median_errors), np.std(np.array(median_errors))))
  print('Best25: %0.2f - std:%0.2f' %
        (np.mean(best_errors), np.std(np.array(best_errors))))
  print('Worst25: %0.2f - std:%0.2f' %
        (np.mean(worst_errors), np.std(np.array(worst_errors))))
  print('Worst05: %0.2f - std:%0.2f' %
        (np.mean(worst_errors_05), np.std(np.array(worst_errors_05))))
  print('Tri: %0.2f - std:%0.2f' %
        (np.mean(tri_errors), np.std(np.array(tri_errors))))
  print('Max: %0.2f - std:%0.2f' %
        (np.mean(max_errors), np.std(np.array(max_errors))))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='test_Cube+', help='model name (path after ../results/)')
  args = parser.parse_args()
  
  model_path = join('..', 'results', args.model_name)
  test_evaluation(model_path)
