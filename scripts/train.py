"""
This code is modified from the C5 repository. [https://github.com/mahmoudnafifi/C5]
"""

import argparse
import logging
import os
import sys
sys.path.append('..')
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from src.ccmnet.ccmnet import network as ccmnet
import random
import math
from src import ops
import wandb, cv2
from src.ops import uv_histogram_to_rgb_image

from src import dataset
from torch.utils.data import DataLoader

torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)


def train_net(net, device, dir_img, val_dir_img=None, aug_dir_img=None, val_ratio=0.1,
              epochs=1000, batch_size=64, lr=0.001, l2reg=0.00001,
              grad_clip_value=0, increasing_batch_size=False, load_hist=True,
              train_subsets=None, test_subsets=None, exclude_cameras=None,
              chkpoint_period=10, smoothness_factor_F=0.15,
              smoothness_factor_B=0.02,
              optimizer_algo='Adam',
              aug_dir=None, input_size=64, validation_frequency=10,
              model_name='CCMNet', save_cp=True, visualize_training=False):
  """ Trains CCMNet network and saves the trained model in harddisk.

  Args:
    net: network object (ccmnet.network).
    device: use 'cpu' or 'cuda' (string).
    dir_img: full path of training set directory (string).
    val_dir_img: full path of validation set directory; if it is None (
      default), some images in training set will be used for validation.
    val_ratio: if val_dir_img is None, this variable set the ratio of the
      total number of training images to be used for validation.
    batch_size: mini-batch size; default value is 64.
    lr: learning rate; default value is 0.001.
    l2reg: L2 regularization factor; default value is 0.00001.
    grad_clip_value: threshold value for clipping gradients. If it is set to
      0 (default) clipping gradient is not applied.
    increasing_batch_size: boolean flag to use increasing batch size during
      training; default value is False.
    load_hist: boolean flag to load histograms from beginning (if exists in the
      image directory); default value is True.
    train_subsets: list of camera names to include in training set; default is
      None (all cameras are included).
    test_subsets: list of camera names to include in validation set; default is
      None (all cameras are included).
    exclude_cameras: list of *additional* camera names to exclude from training
      and augmentation data (IMX135 is always excluded by default); default is None.
    chkpoint_period: save a checkpoint every chkpoint_period epochs; default
      value is 10.
    smoothness_factor_F: smoothness regularization factor of convolutional
      filter F; default value is 0.15.
    smoothness_factor_B: smoothness regularization factor of bias B; default
      value is 0.02.
    optimizer_algo: Optimization algorithm: 'SGD' or 'Adam'; default is 'Adam'.
    aug_dir: full path of additional images (for augmentation). If it is None,
      only the images in the 'dir_img' will be used for training; default
      value is None.
    input_size: Number of bins in histogram; default is 64.
    validation_frequency: Number of epochs to validate the model; default
      value is 10.
    model_name: Name of the final trained model; default is 'CCMNet'.
    save_cp: boolean flag to save checkpoints during training; default is True.
    visualize_training: boolean flag to enable saving of local validation sample images; default is False.
  """

  dir_checkpoint = '../checkpoints_model/'  # check points directory

  # Determine cameras to exclude
  # IMX135 is always excluded.
  # User-provided `exclude_cameras` are additional exclusions.
  cameras_to_exclude_final = ['IMX135'] # Start with the default mandatory exclusion
  user_specified_exclusions_log_str = "None"

  if exclude_cameras: # Check if user provided additional cameras to exclude
    cameras_to_exclude_final.extend(exclude_cameras)
    # Remove duplicates if user accidentally included IMX135 or other duplicates
    cameras_to_exclude_final = sorted(list(set(cameras_to_exclude_final)))
    user_specified_exclusions_log_str = str(exclude_cameras)

  # check if there is additional images to use
  if aug_dir is not None:
    aug_files = []
    for aug_set in aug_dir:
      current_aug_files = dataset.Data.load_files(os.path.join(aug_dir_img,aug_set))
      if cameras_to_exclude_final: # Filter if the final list is not empty
        logging.info(f"Excluding cameras from augmentation set {aug_set}: {cameras_to_exclude_final}")
        current_aug_files = dataset.Data.exclude_files(current_aug_files, cameras_to_exclude_final)
      aug_files.extend(current_aug_files)
    random.shuffle(aug_files)
    augmentation = True
  else:
    augmentation = False

  input_files = dataset.Data.load_files(dir_img, train_subsets)
  if cameras_to_exclude_final: # Filter if the final list is not empty
    logging.info(f"Excluding cameras from main training data: {cameras_to_exclude_final}")
    input_files = dataset.Data.exclude_files(input_files, cameras_to_exclude_final)
  random.shuffle(input_files)

  if val_dir_img is not None:
    tr_files = input_files
    val_files = dataset.Data.load_files(val_dir_img, test_subsets)
    val_files = dataset.Data.filter_files(val_files, test_subsets)
  else:
    logging.info('Validation directory is not given. Using a part of '
                  'training data for validation.')
    assert (0 < val_ratio < 1)
    val_files = input_files[:math.ceil(len(input_files) * val_ratio)]
    tr_files = input_files[math.ceil(len(input_files) * val_ratio):]
  
  if aug_dir is not None:
    tr_files = tr_files + aug_files

  # smoothness Sobel filters
  u_variation = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
  u_variation = torch.tensor(
    u_variation, dtype=torch.float32).unsqueeze(0).expand(
    1, 1, 3, 3).to(device=device)
  u_variation.requires_grad = False

  v_variation = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
  v_variation = torch.tensor(
    v_variation, dtype=torch.float32).unsqueeze(0).expand(
    1, 1, 3, 3).to(device=device)
  v_variation.requires_grad = False

  val_batch_sz = min(len(val_files), batch_size)

  train = dataset.Data(tr_files, input_size=input_size,
                       mode='training', load_hist=load_hist)
  val = dataset.Data(val_files, input_size=input_size,
                     mode='testing', load_hist=load_hist)
  train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=16, pin_memory=True)
  val_loader = DataLoader(val, batch_size=val_batch_sz, shuffle=False,
                          num_workers=16, pin_memory=True, drop_last=True)

  global_step = 0

  logging.info(f'''Starting training:
        Model Name:            {model_name}
        Epochs:                {epochs}
        Batch size:            {batch_size}
        Input size:            {input_size} x {input_size}
        Skip connection:       True
        Learning rate:         {lr}
        L2 reg. weight:        {l2reg}
        Training data:         {len(train)}
        Train subsets:         {train_subsets}
        Test subsets:          {test_subsets}
        Augmentation:          {augmentation}
        Increasing batch size: {increasing_batch_size}
        Smoothness factor F:   {smoothness_factor_F}
        Smoothness factor B:   {smoothness_factor_B}
        Grad. clipping:        {grad_clip_value}
        Optimizer:             {optimizer_algo}
        Excluded cameras (always): IMX135
        User-specified additional exclusions: {user_specified_exclusions_log_str}
        Validation size:       {len(val)}
        Validation Frq.:       {validation_frequency}
        Checkpoints:           {save_cp}
        Device:                {device.type}
    ''')

  if optimizer_algo == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                           weight_decay=l2reg)
  elif optimizer_algo == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2reg)
  else:
    raise NotImplementedError

  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.5)

  curr_batch_size = batch_size

  if increasing_batch_size:
    max_batch_size = 128  # maximum number of mini-batch
    milestones = [20, 50]  # milestones (epochs) to duplicate curr_batch_size

  best_val_score = 1000

  for epoch in range(epochs):
    net.train()
      
    epoch_angular_loss = 0
    epoch_smoothness_loss = 0

    if increasing_batch_size and (epoch + 1) in milestones:
      if curr_batch_size < max_batch_size:
        curr_batch_size = min(int(curr_batch_size * 1.5), max_batch_size)

        # training data loader
        train = dataset.Data(tr_files, input_size=input_size,
                             mode='training', load_hist=True)
        train_loader = DataLoader(train, batch_size=curr_batch_size,
                                  shuffle=True, num_workers=16,
                                  pin_memory=True)

    with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}',
              unit='img') as pbar:
      for batch in train_loader:
        # input histogram batch
        histogram = batch['histogram']
        histogram = histogram.to(device=device, dtype=torch.float32)

        # model histogram(s)
        model_histogram = batch['model_input_histograms']
        model_histogram = model_histogram.to(device=device,
                                             dtype=torch.float32)

        # gt illuminant color batch
        gt = batch['gt_ill']
        gt = gt.to(device=device, dtype=torch.float32)

        # ColorMatrices
        cm1 = batch['cm1']
        cm1 = cm1.to(device=device, dtype=torch.float32)
        cm2 = batch['cm2']
        cm2 = cm2.to(device=device, dtype=torch.float32)

        predicted_ill, P, F, B, N_after_conv = net(histogram, model_in_N=model_histogram, cm1=cm1, cm2=cm2)

        if len(B.shape) == 2:
          B = torch.unsqueeze(B, dim=0)

        if len(F.shape) == 3:
          F = torch.unsqueeze(F, dim=0)

        loss = ops.angular_loss(predicted_ill, gt)
        py_loss = loss.item()

        # convert shrink angular error back to true angular error for printing
        try:
          py_loss = np.rad2deg(np.math.acos(np.math.cos(np.deg2rad(
            py_loss)) / 0.9999999))
        except:
          pass

        # decouple F into chroma and edge filters for visualization
        F_chroma = F[:, 0, :, :]
        F_edges = F[:, 1, :, :]

        # smoothing regularization for B
        s_loss_B = smoothness_factor_B * (torch.mean(
          torch.nn.functional.conv2d(
            torch.unsqueeze(B, dim=1), u_variation, stride=1) ** 2) +
                                          torch.mean(
                                            torch.nn.functional.conv2d(
                                              torch.unsqueeze(B, dim=1),
                                              v_variation,
                                              stride=1) ** 2))

        # smoothing regularization for F
        s_loss_F_chroma = (torch.mean(torch.nn.functional.conv2d(
          torch.unsqueeze(F_chroma, dim=1), u_variation, stride=1) ** 2) +
                           torch.mean(torch.nn.functional.conv2d(
                             torch.unsqueeze(F_chroma, dim=1), v_variation,
                             stride=1) ** 2))

        s_loss_F_edges = (torch.mean(torch.nn.functional.conv2d(
          torch.unsqueeze(F_edges, dim=1), u_variation, stride=1) ** 2) +
                          torch.mean(torch.nn.functional.conv2d(
                            torch.unsqueeze(F_edges, dim=1), v_variation,
                            stride=1) ** 2))

        s_loss_F = smoothness_factor_F * (s_loss_F_chroma +
                                          s_loss_F_edges)

        # final smoothing regularization
        smoothness_loss = s_loss_F + s_loss_B

        loss = loss + smoothness_loss

        epoch_smoothness_loss += smoothness_loss.item()
        epoch_angular_loss += py_loss

        optimizer.zero_grad()
        loss.backward()

        # log step loss using wandb
        wandb.log({'angular_loss': py_loss,
                   'smoothness_loss': smoothness_loss.item(),
                   'total_loss': loss.item()})

        if grad_clip_value > 0:
          torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_value)

        optimizer.step()

        pbar.update(np.ceil(histogram.shape[0]))

        pbar.set_postfix(**{'angular loss (batch)': py_loss},
                         **{'smoothness loss (batch)'
                            : smoothness_loss.item()})

        global_step += 1

    epoch_smoothness_loss = epoch_smoothness_loss / (len(train) /
                                                     curr_batch_size)
    epoch_angular_loss = epoch_angular_loss / (len(train) / curr_batch_size)

    logging.info(f'Epoch loss: angular = {epoch_angular_loss}, '
                 f'smoothness = {epoch_smoothness_loss}')
    
    # log epoch loss using wandb
    wandb.log({'epoch_angular_loss': epoch_angular_loss,
           'epoch_smoothness_loss': epoch_smoothness_loss,
           'epoch_total_loss': epoch_angular_loss + epoch_smoothness_loss,
           'epoch': epoch + 1,
           'lr': optimizer.param_groups[0]['lr']})

    scheduler.step()

    # model validation
    if (epoch + 1) % validation_frequency == 0:

      val_score = vald_net(net=net, loader=val_loader, device=device, model_name=args.model_name, epoch=epoch, visualize_training=visualize_training)

      logging.info('Validation loss: {}'.format(val_score))

      # save the best model
      if (val_score < best_val_score):
        wandb.run.summary['best_val_score'] = val_score
        wandb.alert(title='Best model saved!', text=f'Epoch {epoch+1} \nValidation loss: {best_val_score:.4f} -> {val_score:.4f}')
        logging.info('Best model saved!')

        best_val_score = val_score
        torch.save(net.state_dict(), '../models/' + f'{model_name}_best.pth')

    # save the last checkpoint
    if save_cp and (epoch + 1) % chkpoint_period == 0:
      if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')

      torch.save(net.state_dict(), dir_checkpoint +
                 f'{model_name}_{epoch + 1}.pth')
      logging.info(f'Checkpoint {epoch + 1} saved!')

  # save final trained model
  if not os.path.exists('../models'):
    os.mkdir('../models')
    logging.info('Created trained models directory')

  torch.save(net.state_dict(), '../models/' + f'{model_name}.pth')
  logging.info('Saved trained model!')
  logging.info('End of training')


def vald_net(net, loader, device='cuda', model_name=None, epoch=None, visualize_training=False):
  """ Evaluates using the validation set.

  Args:
    net: network object
    loader: dataloader of validation data
    device: 'cpu' or 'cuda'; default is 'cuda'
    model_name: Name of the model, used for visualization directory path.
    epoch: Current epoch number, used for naming visualized images.
    visualize_training: If True, saves intermediate outputs as images locally.

  Returns:
    val_loss: validation angular error
  """
  vis_dir = os.path.join('../vis_training', f'{model_name}')
  os.makedirs(vis_dir, exist_ok=True)

  net.eval()
  n_val = 0
  val_loss = 0

  with tqdm(total=len(loader), desc='Validation round', unit='batch',
            leave=False) as pbar:
    for i,batch in enumerate(loader):

      histogram = batch['histogram']
      histogram = histogram.to(device=device,
                               dtype=torch.float32)

      model_histogram = batch['model_input_histograms']
      model_histogram = model_histogram.to(device=device,
                                           dtype=torch.float32)

      gt = batch['gt_ill']
      gt = gt.to(device=device, dtype=torch.float32)

      cm1 = batch['cm1']
      cm1 = cm1.to(device=device, dtype=torch.float32)
      cm2 = batch['cm2']
      cm2 = cm2.to(device=device, dtype=torch.float32)

      with torch.no_grad():

        predicted_ill, P, F, B, heatmap_before_sftmx = net(histogram, model_in_N=model_histogram, cm1=cm1, cm2=cm2)
        if visualize_training and i == 0:
          rand_idx = random.randint(0, predicted_ill.shape[0] - 1)
          uv_histogram_to_rgb_image(P[rand_idx].cpu().numpy(), save_path=os.path.join(vis_dir, f'P_{epoch}.png'))
          uv_histogram_to_rgb_image(F[rand_idx, 0].cpu().numpy(), save_path=os.path.join(vis_dir, f'F_chroma_{epoch}.png'))
          uv_histogram_to_rgb_image(F[rand_idx, 1].cpu().numpy(), save_path=os.path.join(vis_dir, f'F_edges_{epoch}.png'))
          uv_histogram_to_rgb_image(B[rand_idx].cpu().numpy(), save_path=os.path.join(vis_dir, f'B_{epoch}.png'))
          uv_histogram_to_rgb_image(heatmap_before_sftmx[rand_idx].cpu().numpy(), save_path=os.path.join(vis_dir, f'heatmap_before_sftmx_{epoch}.png'))
        loss = ops.angular_loss(predicted_ill, gt)

        try:
          py_loss = np.rad2deg(np.math.acos(np.math.cos(np.deg2rad(
            loss.item())) / 0.9999999))
        except:
          py_loss = loss.item()

        py_loss = py_loss * predicted_ill.shape[0]
        n_val = n_val + predicted_ill.shape[0]
        val_loss = val_loss + py_loss

      pbar.update(np.ceil(histogram.shape[0]))

  net.train()
  val_loss = val_loss / n_val

  # log validation loss using wandb
  wandb.log({'val_loss': val_loss, 'epoch': epoch + 1})

  return val_loss


def get_args():
  """ Gets command-line arguments.

  Returns:
    Return command-line arguments as a set of attributes.
  """

  parser = argparse.ArgumentParser(description='Train CCMNet.')
  parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                      help='Number of epochs', dest='epochs')

  parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                      default=16, help='Batch size', dest='batch_size')

  parser.add_argument('-opt', '--optimizer', dest='optimizer', type=str,
                      default='Adam', help='Adam or SGD')

  parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float,
                      nargs='?', default=5e-4, help='Learning rate', dest='lr')

  parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float,
                      nargs='?', default=5e-4, help='L2 regularization '
                                                    'factor', dest='l2r')

  parser.add_argument('-l', '--load', dest='load', type=bool, default=False,
                      help='Load model from a .pth file')

  parser.add_argument('-ml', '--model-location', dest='model_location',
                      default=None)

  parser.add_argument('-vr', '--validation-ratio', dest='val_ratio',
                      type=float, default=0.1, help='Validation set ratio.')

  parser.add_argument('-vf', '--validation-frequency', dest='val_frq',
                      type=int, default=1, help='Validation frequency.')

  parser.add_argument('-s', '--input-size', dest='input_size', type=int,
                      default=64, help='Size of input histogram')

  parser.add_argument('-lh', '--load-hist', dest='load_hist',
                      type=bool, default=True, help='Load histogram if exists')

  parser.add_argument('-ibs', '--increasing-batch-size',
                      dest='increasing_batch_size', type=bool, default=False,
                      help='Increasing batch size.')

  parser.add_argument('-gc', '--grad-clip-value', dest='grad_clip_value',
                      type=float, default=0.5, help='Gradient clipping value; '
                                                  'if = 0, no clipping applied')

  parser.add_argument('-slf', '--smoothness-factor-F',
                      dest='smoothness_factor_F', type=float, default=0.15,
                      help='Smoothness regularization factor of conv filter')

  parser.add_argument('-slb', '--smoothness-factor-B',
                      dest='smoothness_factor_B', type=float, default=0.02,
                      help='Smoothness regularization factor of bias')

  parser.add_argument('-ntrd', '--training-dir-in', dest='in_trdir',
                      default='../../dataset/CCMNet/original_resized/',
                      help='Input training image directory')

  parser.add_argument('-nvld', '--validation-dir-in', dest='in_vldir',
                      default=None,
                      help='Input validation image directory; if is None, the '
                           'validation will be taken from the training data '
                           'based on the validation-ratio argument')
  
  parser.add_argument('-nagd', '--augmentation-dir-in', dest='in_augdir',
                      default='../../dataset/CCMNet/augmented_dataset/')

  parser.add_argument('-augd', '--augmentation-dir', dest='aug_dir',
                      default=None, nargs='+',
                      help='Directory include augmentation data.')

  parser.add_argument('-n', '--model-name', dest='model_name',
                      default='CCMNet')

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  # list of train subsets
  parser.add_argument('-tr' '--train_subsets', dest='train_subsets', default=None,
                      nargs='+', help='List of train subsets')

  # list of test subsets
  parser.add_argument('-ts', '--test_subsets', dest='test_subsets', default=None,
                      nargs='+', help='List of test subsets')
  
  parser.add_argument('-netd', '--net-depth', dest='net_depth', default=4, type=int,
                      help='Depth of encoder network')
  
  parser.add_argument('-maxc', '--max-convdepth', dest='max_conv_depth', default=32, type=int, help='Max depth of conv layers')
  
  parser.add_argument('-cfefn', '--cfe-feature-num', dest='cfe_feature_num', default=8, type=int,
                      help='Channel size of camera fingerprint embedding (CFE) feature')
  
  parser.add_argument('--visualize-training', dest='visualize_training', default=False, action='store_true',
                      help='Enable saving of local validation sample images during training.')

  parser.add_argument('-excld', '--exclude', dest='exclude_cameras', default=None,
                      nargs='+', help='List of *additional* camera names to exclude from training (IMX135 is always excluded by default).')

  return parser.parse_args()

def get_model_name(args):
  basename = 'ccm'
  model_name = f'{basename}'

  prefix = ops.get_current_datetime()
  model_name = f'{prefix}_{model_name}'

  return model_name

if __name__ == '__main__':

  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  # Uncomment below lines to make training totally reproducible. But it will slow down the training.
  # torch.use_deterministic_algorithms(True)
  # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Training CCMNet')
  args = get_args()
  args.model_name = get_model_name(args)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)

  logging.info(f'Using device {device}')

  net = ccmnet(input_size=args.input_size,
                   cfe_feature_num=args.cfe_feature_num,
                   net_depth=args.net_depth,
                   max_conv_depth=args.max_conv_depth,
                   device=device)
  
  if args.load:
    net.load_state_dict(
      torch.load(args.model_location, map_location=device)
    )
    logging.info(f'Model loaded from {args.model_location}')

  net.to(device=device)

  wandb.init(project='CCWB2', name=args.model_name, dir='../wandb')
  wandb.config.update(args)

  try:
    train_net(net=net, device=device, dir_img=args.in_trdir,
              val_dir_img=args.in_vldir, aug_dir_img=args.in_augdir, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr,
              train_subsets=args.train_subsets,
              test_subsets=args.test_subsets,
              exclude_cameras=args.exclude_cameras,
              smoothness_factor_F=args.smoothness_factor_F,
              smoothness_factor_B=args.smoothness_factor_B,
              l2reg=args.l2r,
              load_hist=args.load_hist,
              optimizer_algo=args.optimizer, aug_dir=args.aug_dir,
              increasing_batch_size=args.increasing_batch_size,
              grad_clip_value=args.grad_clip_value,
              chkpoint_period=args.val_frq,
              validation_frequency=args.val_frq, input_size=args.input_size,
              val_ratio=args.val_ratio, model_name=args.model_name,
              visualize_training=args.visualize_training)

  except KeyboardInterrupt:
    logging.info('Interrupted by user. Exiting...')
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
