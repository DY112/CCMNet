import numpy as np
from scipy.io import loadmat
from os.path import join

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

def multiple_test_evaluation(file_paths):
    all_errors = []

    for file_path in file_paths:
        gt = loadmat(file_path)['gt']
        predicted = loadmat(file_path.replace('gt', 'results'))['predicted']
        error = ang_error(predicted, gt)
        all_errors.extend(error)

    all_errors = np.array(all_errors)
    metrics = error_metrics(all_errors)

    print('Mean: %0.2f - std:%0.2f' % (metrics['mean'], np.std(all_errors)))
    print('Median: %0.2f - std:%0.2f' % (metrics['median'], np.std(all_errors)))
    print('Best25: %0.2f - std:%0.2f' % (metrics['b25'], np.std(all_errors)))
    print('Worst25: %0.2f - std:%0.2f' % (metrics['w25'], np.std(all_errors)))
    print('Worst05: %0.2f - std:%0.2f' % (metrics['w05'], np.std(all_errors)))
    print('Tri: %0.2f - std:%0.2f' % (metrics['tri'], np.std(all_errors)))
    print('Max: %0.2f - std:%0.2f' % (metrics['max'], np.std(all_errors)))

if __name__ == '__main__':
    cams = [
        'Canon1DsMkIII',
        'Canon600D',
        'FujifilmXM1',
        'NikonD5200',
        'OlympusEPL6',
        'PanasonicGX1',
        'SamsungNX2000',
        'SonyA57',
    ]
    ckpt_path_templete = '../results/c5_m_1_b_10_cam_aug'
    # Example usage
    # file_paths = []

    # for cam in cams:
    #     file_paths.append(join(ckpt_path_templete.replace('cam', cam), 'gt.mat'))
    
    file_paths = ['250105-163845_ccm',
                  '250105-180441_ccm',
                  '250105-180506_ccm',
                  '250105-180536_ccm',
                  '250105-180602_ccm',
                  '250105-222708_ccm',
                  '250105-222731_ccm',
                  '250106-134343_ccm']
    
    file_paths = [join('../results', path, 'gt.mat') for path in file_paths]

    multiple_test_evaluation(file_paths)