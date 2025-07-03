"""
This code is modified from the C5 repository. [https://github.com/mahmoudnafifi/C5]
"""

import argparse
import logging
import os,sys
import numpy as np
import torch
sys.path.append('..')
from src.ccmnet.ccmnet import network as ccmnet
from scipy.io import savemat
from src import dataset
from torch.utils.data import DataLoader
from src.ops import uv_histogram_to_rgb_image
from torchvision.utils import save_image
from torchvision.utils import make_grid

# Gamma correction value, commonly used for sRGB display
GAMMA_CORRECTION_VALUE = 1.0 / 2.19921875

def test_net(net, device, dir_img, batch_size=64, input_size=64, test_subsets=None,
             model_name='CCMNet', load_hist=False,
             white_balance=False, files=None,
             save_output=True,
             visualize_gt=False, visualize_intermediate=False, add_color_bar=True):
  """ Tests CCMNet network.

  Args:
    net: network object (ccmnet.network).
    device: use 'cpu' or 'cuda' (string).
    dir_img: full path of testing set directory (string).
    batch_size: mini-batch size; default value is 64.
    input_size: Number of bins in histogram; default is 64.
    test_subsets: list of test subsets; default is None.
    model_name: Name of the trained model; default is 'CCMNet'.
    load_hist: boolean flag to load histograms from beginning (if exists in the
      image directory); default value is True.
    white_balance: boolean to perform a diagonal correction using the estimated
      illuminant color and save output images in harddisk. The saved images
      will be located in white_balanced_images/model_name/.; default is False.
    files: a list to override loading files located in dir_img; default is
      None.
    save_output: boolean flag to save the results in results/model_name/.;
      default is True.
    visualize_gt: boolean flag to visualize ground truth images.
    visualize_intermediate: boolean flag to visualize intermediate outputs like P, F_chroma, B.
    add_color_bar: boolean flag to add color bar to visualized images.
  """

  if files is None:
    files = dataset.Data.load_files(dir_img, test_subsets)
  files = dataset.Data.filter_files(files, test_subsets)

  batch_size = min(batch_size, len(files))

  test = dataset.Data(files, input_size=input_size, mode='testing',
                      load_hist=load_hist)
  test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                           num_workers=8, pin_memory=True)

  logging.info(f'''Starting testing:
        Model Name:            {model_name}
        Batch size:            {batch_size}
        Input size:            {input_size} x {input_size}
        Testing data:          {len(files)}
        Test subsets:          {test_subsets}
        White balance:         {white_balance}
        Save output:           {save_output}
        Device:                {device.type}
        Visualize GT:          {visualize_gt}
        Visualize Intermediate:{visualize_intermediate}
        Add Color Bar:         {add_color_bar}
    ''')

  # General flag, can be determined once
  is_any_visualization_globally_enabled = white_balance or visualize_gt or visualize_intermediate
  save_vis_dir = None # Initialize save_vis_dir

  if is_any_visualization_globally_enabled:
    save_vis_dir = os.path.join('../white_balanced_images', model_name)
    parent_visualization_base_dir = os.path.dirname(save_vis_dir)
    if not os.path.exists(parent_visualization_base_dir):
      os.makedirs(parent_visualization_base_dir, exist_ok=True)
    if not os.path.exists(save_vis_dir):
      os.makedirs(save_vis_dir, exist_ok=True)
      logging.info(f'Created visualization directory {save_vis_dir}')

  with torch.no_grad():
    results = np.zeros((len(test), 3))  # to store estimated illuminant values
    gt = np.zeros((len(test), 3))  # to store ground-truth illuminant colors
    filenames = []  # to store filenames
    index = 0

    for batch in test_loader:
      model_histogram = batch['model_input_histograms']
      model_histogram = model_histogram.to(device=device,
                                           dtype=torch.float32)
      file_names = batch['file_name']
      
      image_rgb_from_batch = None
      # Load image if any visualization option that requires the original RGB image is enabled
      if white_balance or visualize_gt or visualize_intermediate:
          if 'image_rgb' in batch: # Ensure the key exists
              image_rgb_from_batch = batch['image_rgb'].to(device=device, dtype=torch.float32)
          else:
              logging.warning("'image_rgb' not found in batch, cannot perform some visualizations.")

      histogram = batch['histogram']
      histo_npy = histogram.cpu().numpy()
      histogram = histogram.to(device=device, dtype=torch.float32)

      gt_ill = batch['gt_ill']
      gt_ill = gt_ill.to(device=device, dtype=torch.float32)

      cm1 = batch['cm1']
      cm1 = cm1.to(device=device, dtype=torch.float32)
      cm2 = batch['cm2']
      cm2 = cm2.to(device=device, dtype=torch.float32)

      predicted_ill, P, F, B, N_after_conv = net(histogram, model_in_N=model_histogram, cm1=cm1, cm2=cm2)
      P = P.detach().cpu().numpy()
      F = F.detach().cpu().numpy()
      B = B.detach().cpu().numpy()
      N_after_conv = N_after_conv.detach().cpu().numpy()
      P_before_sftmx = N_after_conv + B
      F_chroma = F[:, 0, :, :]
      F_edges = F[:, 1, :, :]

      # Visualization is performed if globally enabled
      if is_any_visualization_globally_enabled:
        if save_vis_dir is None:
            logging.error("save_vis_dir is None inside visualization block when it should have been set. Skipping visualization.")
            continue

        bs_for_visualization = 0
        if (white_balance or visualize_gt) and image_rgb_from_batch is not None:
            bs_for_visualization = image_rgb_from_batch.shape[0]
        elif predicted_ill is not None: 
            bs_for_visualization = predicted_ill.shape[0]
        
        if bs_for_visualization == 0:
            # If image_rgb_from_batch was None but predicted_ill is also None or empty, log and skip.
            # This can happen if a batch is problematic or empty.
            if predicted_ill is None or predicted_ill.shape[0] == 0 :
                logging.warning(f"Batch size for visualization is 0 (no image data or predictions) for files {file_names}. Skipping visualization for this batch.")
                continue # Skip to next batch if no items to visualize
            # If predicted_ill has items, bs_for_visualization should have been set from it.
            # This path indicates bs_for_visualization remained 0 unexpectedly.
            logging.warning(f"Could not determine a valid batch size for visualization for files {file_names}. Skipping visualization for this batch.")
            continue


        # 1. White-balanced image visualization
        if white_balance and image_rgb_from_batch is not None:
          corrected_image = image_rgb_from_batch.clone() # Clone to avoid modifying the original batch image
          for c in range(3):
            correction_ratio = predicted_ill[:, 1] / torch.clamp(predicted_ill[:, c], min=1e-6) # Avoid division by zero
            correction_ratio = correction_ratio.view(bs_for_visualization, 1, 1)
            # Ensure predicted_ill has enough elements for bs_for_visualization
            if c < predicted_ill.shape[1] and bs_for_visualization <= predicted_ill.shape[0]:
               corrected_image[:, c, :, :] = corrected_image[:, c, :, :] * correction_ratio[:bs_for_visualization]


          corrected_image = torch.pow(corrected_image, GAMMA_CORRECTION_VALUE)

          if add_color_bar:
            # Create color bar for predicted illuminant
            pred_color_bar = torch.zeros((bs_for_visualization, 3, corrected_image.shape[2], 20), device=device, dtype=torch.float32)
            max_val_pred = torch.max(predicted_ill, dim=1, keepdim=True)[0]
            # Ensure shapes match for division
            normalized_pred_ill = predicted_ill[:bs_for_visualization] / torch.clamp(max_val_pred[:bs_for_visualization], min=1e-6)
            for b_idx in range(bs_for_visualization):
              if b_idx < normalized_pred_ill.shape[0]: # Check bounds
                  pred_color_bar[b_idx, :, :, :] = normalized_pred_ill[b_idx].view(3, 1, 1)
            # Concatenate image and color bar
            corrected_image_to_save = torch.cat((corrected_image, pred_color_bar), dim=3)
          else:
            corrected_image_to_save = corrected_image

          for b_idx in range(bs_for_visualization):
            if b_idx >= corrected_image_to_save.shape[0] or b_idx >= predicted_ill.shape[0] or b_idx >= gt_ill.shape[0]: continue # Boundary check
            angular_error = torch.acos(torch.clamp(torch.sum(predicted_ill[b_idx] * gt_ill[b_idx]) /
               (torch.norm(predicted_ill[b_idx]) * torch.norm(gt_ill[b_idx])), -1.0, 1.0))
            angular_error_deg = torch.rad2deg(angular_error).item()
            filename_with_error = f"{os.path.splitext(file_names[b_idx])[0]}_wb_{angular_error_deg:.2f}.png"
            save_image(make_grid(corrected_image_to_save[b_idx, :, :, :], nrow=1), os.path.join(save_vis_dir, filename_with_error))

        # 2. Ground truth image visualization
        if visualize_gt and image_rgb_from_batch is not None:
          gt_corrected_image = image_rgb_from_batch.clone()
          for c in range(3):
            # Ensure gt_ill has enough elements for bs_for_visualization
            if c < gt_ill.shape[1] and bs_for_visualization <= gt_ill.shape[0]:
                gt_correction_ratio = gt_ill[:bs_for_visualization, 1] / torch.clamp(gt_ill[:bs_for_visualization, c], min=1e-6) # Avoid division by zero
                gt_correction_ratio = gt_correction_ratio.view(bs_for_visualization, 1, 1)
                gt_corrected_image[:, c, :, :] = gt_corrected_image[:, c, :, :] * gt_correction_ratio
          
          gt_corrected_image = torch.pow(gt_corrected_image, GAMMA_CORRECTION_VALUE)

          if add_color_bar:
            # Create color bar for ground truth illuminant
            gt_color_bar_tensor = torch.zeros((bs_for_visualization, 3, gt_corrected_image.shape[2], 20), device=device, dtype=torch.float32)
            max_val_gt = torch.max(gt_ill, dim=1, keepdim=True)[0]
            # Ensure shapes match for division
            normalized_gt_ill = gt_ill[:bs_for_visualization] / torch.clamp(max_val_gt[:bs_for_visualization], min=1e-6)
            for b_idx in range(bs_for_visualization):
               if b_idx < normalized_gt_ill.shape[0]: # Check bounds
                  gt_color_bar_tensor[b_idx, :, :, :] = normalized_gt_ill[b_idx].view(3, 1, 1)
            # Concatenate ground truth image and color bar
            gt_image_to_save = torch.cat((gt_corrected_image, gt_color_bar_tensor), dim=3)
          else:
            gt_image_to_save = gt_corrected_image

          for b_idx in range(bs_for_visualization):
            if b_idx >= gt_image_to_save.shape[0]: continue # Boundary check
            filename_gt = f"{os.path.splitext(file_names[b_idx])[0]}_gt.png"
            save_image(make_grid(gt_image_to_save[b_idx, :, :, :], nrow=1), os.path.join(save_vis_dir, filename_gt))

        # 3. Intermediate results visualization (P, F_chroma, F_edges, B, N_after_conv, P_before_sftmx)
        if visualize_intermediate:
          # These visualizations depend on the batch size of P, F, B etc. which should match predicted_ill.shape[0]
          # If bs_for_visualization was derived from predicted_ill, this is fine.
          # Or, use P.shape[0] directly for these loops if there's a mismatch possibility.
          num_intermediate_items = P.shape[0] # typically matches bs_for_visualization if from same batch processing
          for b_idx in range(num_intermediate_items):
            if b_idx >= len(file_names): continue # Ensure file_names has this index

            base_filename = os.path.splitext(file_names[b_idx])[0]
            uv_histogram_to_rgb_image(P[b_idx], save_path=os.path.join(save_vis_dir, f'{base_filename}_P.png'))
            uv_histogram_to_rgb_image(F_chroma[b_idx], save_path=os.path.join(save_vis_dir, f'{base_filename}_F_chroma.png'))
            uv_histogram_to_rgb_image(F_edges[b_idx], save_path=os.path.join(save_vis_dir, f'{base_filename}_F_edges.png'))
            uv_histogram_to_rgb_image(B[b_idx], save_path=os.path.join(save_vis_dir, f'{base_filename}_B.png'))
            uv_histogram_to_rgb_image(N_after_conv[b_idx], save_path=os.path.join(save_vis_dir, f'{base_filename}_N_after_conv.png'))
            uv_histogram_to_rgb_image(P_before_sftmx[b_idx], save_path=os.path.join(save_vis_dir, f'{base_filename}_P_before_sftmx.png'))

            # Save input histograms (original and edges)
            if b_idx < histo_npy.shape[0]: # Check bounds for histo_npy
                uv_histogram_to_rgb_image(histo_npy[b_idx][0], save_path=os.path.join(save_vis_dir, f'{base_filename}_hist_input.png'))
                if histo_npy[b_idx].shape[0] > 1: # Check if edge histogram exists
                    uv_histogram_to_rgb_image(histo_npy[b_idx][1], save_path=os.path.join(save_vis_dir, f'{base_filename}_hist_edges_input.png'))

        # 4. Save input raw image (original), scaled to max 1
        if (white_balance or visualize_gt or visualize_intermediate) and image_rgb_from_batch is not None:
          input_rgb_original = image_rgb_from_batch.clone()
          for b_idx in range(bs_for_visualization):
            if b_idx >= input_rgb_original.shape[0] or b_idx >= len(file_names): continue # Boundary check

            filename_raw = f"{os.path.splitext(file_names[b_idx])[0]}_raw_input.png"
            # Scale each image individually so that the maximum value is 1
            current_img_rgb = input_rgb_original[b_idx, :, :, :]
            max_val_img = torch.max(current_img_rgb)
            if max_val_img > 1e-6: # Avoid division by zero or near-zero for black images
                current_img_rgb = current_img_rgb / max_val_img
            else: # if max_val_img is 0 or very small, image is already black or near black
                current_img_rgb = torch.zeros_like(current_img_rgb)

            save_image(make_grid(current_img_rgb, nrow=1), os.path.join(save_vis_dir, filename_raw))


      L = len(predicted_ill)
      results[index:index + L, :] = predicted_ill.cpu().numpy()
      gt[index:index + L, :] = gt_ill.cpu().numpy()
      for f in file_names:
          filenames.append(f)
      index = index + L

    if save_output:
      save_dir = os.path.join('../results', model_name)
      if not os.path.exists(save_dir):
        if not os.path.exists('../results'):
          os.mkdir('../results')
        os.mkdir(save_dir)
        logging.info(f'Created results directory {save_dir}')
      savemat(os.path.join(save_dir, 'gt.mat'), {'gt': gt})
      savemat(os.path.join(save_dir, 'results.mat'), {'predicted': results})
      savemat(os.path.join(save_dir, 'filenames.mat'), {'filenames': filenames})

  logging.info('End of testing')


def get_args():
  parser = argparse.ArgumentParser(description='Test CCMNet.')

  parser.add_argument('-b', '--batch-size', metavar='B', type=int,
                      nargs='?', default=64,
                      help='Batch size', dest='batchsize')

  parser.add_argument('-s', '--input-size', dest='input_size', type=int,
                      default=64, help='Size of input (hist and image)')

  parser.add_argument('-ntrd', '--testing-dir-in', dest='in_tedir',
                      default='../../dataset/CCMNet/original_resized/',
                      help='Input testing image directory')

  parser.add_argument('-lh', '--load-hist', dest='load_hist',
                      type=bool, default=True,
                      help='Load histogram if exists')

  parser.add_argument('-wb', '--white-balance', type=bool,
                      default=False, help='save white-balanced image',
                      dest='white_balance')

  parser.add_argument('-n', '--model-name', dest='model_name',
                      default=None)

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  # list of test subsets
  parser.add_argument('-ts', '--test_subsets', dest='test_subsets', default=None,
                      nargs='+', help='List of test subsets')
  
  parser.add_argument('-netd', '--net-depth', dest='net_depth', default=4, type=int,
                      help='Depth of encoder network')
  
  parser.add_argument('-maxc', '--max-convdepth', dest='max_conv_depth', default=32, type=int, help='Max depth of conv layers')
  
  parser.add_argument('-cfefn', '--cfe-feature-num', dest='cfe_feature_num', default=8, type=int,
                      help='Channel size of camera fingerprint embedding (CFE) feature')
  
  # Visualization options
  parser.add_argument('--visualize-gt', action='store_true', default=False,
                      help='Visualize ground truth images')
  parser.add_argument('--visualize-intermediate', action='store_true', default=False,
                      help='Visualize intermediate outputs like P, F_chroma, B')
  parser.add_argument('--add-color-bar', action='store_true', dest='add_color_bar', default=False,
                      help='Add color bar to visualized images (default: False)')

  return parser.parse_args()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Testing CCMNet')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)
  logging.info(f'Using device {device}')

  net = ccmnet(input_size=args.input_size,
                cfe_feature_num=args.cfe_feature_num,
                net_depth=args.net_depth,
                max_conv_depth=args.max_conv_depth,
                device=device)

  model_path = os.path.join('../models', args.model_name + '.pth')
  net.load_state_dict(torch.load(model_path, map_location=device))
  logging.info(f'Model loaded from {model_path}')
  net.to(device=device)
  net.eval()
  test_net(net=net, device=device,
           test_subsets=args.test_subsets, dir_img=args.in_tedir,
           white_balance=args.white_balance,
           visualize_gt=args.visualize_gt,
           visualize_intermediate=args.visualize_intermediate,
           add_color_bar=args.add_color_bar,
           batch_size=args.batchsize, model_name=args.model_name,
           input_size=args.input_size,
           load_hist=args.load_hist)
