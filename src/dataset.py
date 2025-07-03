"""
This code is modified from the C5 repository. [https://github.com/mahmoudnafifi/C5]
"""

from os.path import join
from os import listdir
from os import path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import json
from src import ops
import random
import argparse


class Data(Dataset):
  def __init__(self, imgfiles, mode='training', input_size=64,
               load_hist=True, additional_data_selection='random'):
    """ Data constructor

    Args:
      imgfiles: a list of full filenames to be used by the dataloader. If the
         mode is set to 'training', each filename in the list should have a
         metadata json file with a postfix '_metadata'. For example, if the
         filename is 'data/image1_sensorname_canon.png', the metadata file
         should be 'data/image1_sensorname_canon_metadata.json'. Each
         metadata file should contain a key named 'illuminant_color_raw' or
         'gt_ill' that contains the true rgb illuminant color.
      mode: 'training' or 'testing'. In the training mode, ground-truth
         illuminant information should be loaded; while for the testing mode it
         is an optional. Default is 'training'.
      input_size: histogram dimensions (number of bins).
      load_hist: boolean flat to load histogram file if it exists; default is
        true.

    Returns:
      Dataset loader object with the selected settings.
    """

    assert (mode == 'training' or mode == 'testing')
    assert (input_size % 2 == 0)
    self.imgfiles = imgfiles
    self.input_size = input_size
    self.image_size = [384, 256]  # width, height
    self.load_hist = load_hist  # load histogram if exists
    self.mode = mode
    self.from_rgb = ops.rgb_to_uv  # rgb to chroma conversion function
    self.to_rgb = ops.uv_to_rgb  # chroma to rgb conversion function
    self.hist_boundary = ops.get_hist_boundary()
    
    self.additional_data_selection = additional_data_selection
    self.fixed_additional_input = {}

    logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

  def __len__(self):
    return len(self.imgfiles)

  def __getitem__(self, i):
    """ Gets next data in the dataloader.

    Args:
      i: index of file in the dataloader.

    Returns:
      A dictionary of the following keys:
      - image_rgb:
      - file_name: filename (without the full path).
      - histogram: input histogram.
      - model_input_histograms: input histogram and the additional histograms
          to be fed to CCMNet network.
      - gt_ill: ground-truth illuminant color. If the dataloader's 'mode'
         variable was  set to 'testing' and the ground-truth illuminant
         information does not exist, it will contain an empty tensor.
    """

    img_file = self.imgfiles[i]

    in_img = ops.read_image(img_file)
    in_img = ops.resize_image(in_img, self.image_size)

    rgb_img = ops.to_tensor(in_img)  # for visualization

    # gets the ground-truth illuminant color
    with open(path.splitext(img_file)[
                0] + '_metadata.json', 'r') as metadata_file:
      metadata = json.load(metadata_file)

    if self.mode == 'training':
      assert ['illuminant_color_raw' in metadata.keys() or 'gt_ill' in
              metadata.keys()]
    if 'illuminant_color_raw' in metadata.keys():
      gt_ill = np.array(metadata['illuminant_color_raw'])
      gt_ill = torch.from_numpy(gt_ill)
    elif 'gt_ill' in metadata.keys():
      gt_ill = np.array(metadata['gt_ill'])
      gt_ill = torch.from_numpy(gt_ill)
    else:
      gt_ill = torch.tensor([])

    if 'cm1' in metadata.keys() and 'cm2' in metadata.keys():
      cm1 = np.array(metadata['cm1']).reshape(9)
      cm2 = np.array(metadata['cm2']).reshape(9)
      cm1 = torch.from_numpy(cm1)
      cm2 = torch.from_numpy(cm2)
    else:
      # same sahpe as cm1 and cm2
      cm1 = torch.zeros(9)
      cm2 = torch.zeros(9)
    
    if 'fm1' in metadata.keys() and 'fm2' in metadata.keys():
      fm1 = np.array(metadata['fm1']).reshape(9)
      fm2 = np.array(metadata['fm2']).reshape(9)
      fm1 = torch.from_numpy(fm1)
      fm2 = torch.from_numpy(fm2)
    else:
      # same sahpe as fm1 and fm2
      fm1 = torch.zeros(9)
      fm2 = torch.zeros(9)

    # computes histogram feature of rgb and edge images
    if self.input_size == 64:
      post_fix = ''
    else:
      post_fix = f'_{self.input_size}'

    if path.exists(path.splitext(img_file)[0] +
                   f'_histogram{post_fix}.npy') and self.load_hist:
      histogram = np.load(path.splitext(img_file)[0] +
                          f'_histogram{post_fix}.npy', allow_pickle=False)
    else:
      histogram = np.zeros((self.input_size, self.input_size, 2))
      valid_chroma_rgb, valid_colors_rgb = ops.get_hist_colors(
        in_img, self.from_rgb)
      histogram[:, :, 0] = ops.compute_histogram(
        valid_chroma_rgb, self.hist_boundary, self.input_size,
        rgb_input=valid_colors_rgb)

      edge_img = ops.compute_edges(in_img)
      valid_chroma_edges, valid_colors_edges = ops.get_hist_colors(
        edge_img, self.from_rgb)

      histogram[:, :, 1] = ops.compute_histogram(
        valid_chroma_edges, self.hist_boundary, self.input_size,
        rgb_input=valid_colors_edges)

      np.save(path.splitext(img_file)[0] + f'_histogram{post_fix}.npy',
              histogram)

    in_histogram = ops.to_tensor(histogram)

    # gets additional input data
    additional_histogram = histogram

    u_coord, v_coord = ops.get_uv_coord(self.input_size,
                                        tensor=False, normalize=True)
    u_coord = np.expand_dims(u_coord, axis=-1)
    v_coord = np.expand_dims(v_coord, axis=-1)

    additional_histogram = np.concatenate([additional_histogram, u_coord],
                                          axis=-1)
    additional_histogram = np.concatenate([additional_histogram, v_coord],
                                          axis=-1)
    additional_histogram = np.expand_dims(additional_histogram, axis=-1)

    additional_histogram = ops.to_tensor(additional_histogram, dims=4)

    # print("RGB Image shape: ", rgb_img.shape)                         # (3, 256, 384)
    # print("Histogram shape: ", in_histogram.shape)                    # (2, 64, 64)
    # print("Additional Histogram shape: ", additional_histogram.shape) # (data_num, 4, 64, 64)
    # print("Ground Truth Illuminant: ", gt_ill.shape)                  # (3,)
    # print("CM1: ", cm1.shape)
    # print("CM2: ", cm2.shape)
    # input()

    return {
              'image_rgb': rgb_img,
              'file_name': path.basename(img_file),
              'histogram': in_histogram,
              'model_input_histograms': additional_histogram,
              'gt_ill': gt_ill,
              'cm1': cm1,
              'cm2': cm2,
              'fm1': fm1,
              'fm2': fm2
            }

  @staticmethod
  def load_files(img_dir, sub_dirs=None):
    """ Loads filenames in a given image directory and its subdirectories.

    Args:
      img_dir: image directory. Note that if the dataloader's 'mode' variable
        was set to 'training', each filename in the list should have a
        metadata json file with a postfix '_metadata'. For example, if the
        filename is 'data/image1_sensorname_canon.png', the metadata file
        should be 'data/image1_sensorname_canon_metadata.json'. Each
        metadata file should contain a key named 'illuminant_color_raw' or
        'gt_ill' that contains the true rgb illuminant color.
      sub_dirs: an optional list of subdirectories within img_dir to load files from.
        If None, loads files directly from img_dir.

    Returns:
      imgfiles: a list of full filenames.
    """

    logging.info(f'Loading images information from {img_dir} and subdirectories...')
    imgfiles = []
    if sub_dirs is None:
      imgfiles = [join(img_dir, file) for file in listdir(img_dir)
                  if file.endswith('.png') or file.endswith('.PNG')]
      logging.info(f'Found {len(imgfiles)} images in {img_dir}')
      
    else:
      logging.info(f'Loading images from subdirectories: {sub_dirs}')
      for sub_dir in sub_dirs:
        full_dir = join(img_dir, sub_dir)
        sub_imgfiles = [join(full_dir, file) for file in listdir(full_dir)
                        if file.endswith('.png') or file.endswith('.PNG')]
        imgfiles += sub_imgfiles
        logging.info(f'Found {len(sub_imgfiles)} images in {sub_dir}')
    return imgfiles

  
  def get_rand_examples_from_sensor(self, current_file, files, target_number):
    """ Randomly selects additional filenames of images taken by the same
       sensor.

    Args:
      current_file: filename of the current image; this filename should be in
         the following format: 'a_sensorname_b.png', where a is image id (can
         contain any string) and b is camera model name. The function will
         randomly select additional images that have the same camera model
         name (i.e., b).
      files: filenames of images in the dataloader.
      target_number: number of the additional images.

    Returns:
      sensor_files: additional image filenames taken by the same camera model
         used to capture the image in current_file.
    """
    assert ('sensorname' in current_file)
    sensor_name = path.splitext(current_file)[0].split('sensorname_')[-1]
    sensor_files = [file for file in files if sensor_name in file]

    if self.additional_data_selection == 'fixed':
      if sensor_name not in self.fixed_additional_input:
        sensor_files.remove(current_file)
        random.shuffle(sensor_files)
        if len(sensor_files) < target_number:
          raise Exception('Cannot find enough training data from sensor:'
                          f'{sensor_name}')
        rand_idx = random.sample(range(len(sensor_files)), target_number)
        sensor_files = [sensor_files[i] for i in rand_idx]
        self.fixed_additional_input[sensor_name] = sensor_files
        # print(f'Fixed additional input for sensor {sensor_name}: {sensor_files}')
      else:
        sensor_files = self.fixed_additional_input[sensor_name]
    else:
      sensor_files.remove(current_file)
      random.shuffle(sensor_files)
      if len(sensor_files) < target_number:
          raise Exception('Cannot find enough training data from sensor:'
                          f'{sensor_name}')
      sensor_files = sensor_files[:target_number]

    return sensor_files
  
  @staticmethod
  def exclude_files(imgfiles, exclude_str_list):
    """ Excludes filenames that contain the exclude_str_list.

    Args:
      imgfiles: a list of full filenames.
      exclude_str_list: a list of strings to exclude from the filenames.

    Returns:
      imgfiles: a list of full filenames that do not contain the exclude_str_list.
    """
    initial_len = len(imgfiles)

    if exclude_str_list is None:
      return imgfiles

    if type(exclude_str_list) is not list:
      exclude_str_list = [exclude_str_list]

    for exclude_str in exclude_str_list:
      imgfiles = [file for file in imgfiles if exclude_str not in file]

    logging.info(f'Excluded {initial_len - len(imgfiles)} images with {exclude_str_list}')
    
    return imgfiles
  
  @staticmethod
  def filter_files(imgfiles, filter_str_list):
    """ Filters filenames that contain the filter_str_list.

    Args:
      imgfiles: a list of full filenames.
      filter_str_list: a list of strings to filter from the filenames.

    Returns:
      imgfiles: a list of full filenames that contain the filter_str_list.
    """
    if type(filter_str_list) is not list:
      filter_str_list = [filter_str_list]
    
    ret_files = []

    for filter_str in filter_str_list:
      ret_files += [file for file in imgfiles if filter_str in file]
    
    return ret_files