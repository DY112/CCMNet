"""
This code is modified from the C5 repository. [https://github.com/mahmoudnafifi/C5]
"""
import sys
sys.path.append('..')
sys.path.append('../..')
import torch.nn as nn
import torch
from src import ops
import math
from src.diff_hist import DifferentiableHistogram
from layers import *
import torch.fft as fft
import numpy as np
import colour

class network(nn.Module):
  def __init__(self, input_size=64, cfe_feature_num=8,
               net_depth=4, max_conv_depth=32, device='cuda'):
    super(network, self).__init__()

    assert (input_size - 2 ** math.ceil(math.log2(input_size)) == 0 and input_size >= 16)

    self.input_size = input_size
    self.device = device
    self.cfe_feature_num = cfe_feature_num
    self.u_coord, self.v_coord = ops.get_uv_coord(self.input_size, tensor=True, device=self.device)
    
    self.histogram_layer = DifferentiableHistogram(hist_boundary=(-0.5,1.5))
    self.net_depth = net_depth
    self.sigmoid_layer = nn.Sigmoid()
    self.tanh_layer = nn.Tanh()
    self.use_cfe = True

    self.data_num = 1

    encoder_in_channel_primary = 4
    encoder_in_channel_primary += self.cfe_feature_num
    
    initial_conv_depth_primary = min(2 * encoder_in_channel_primary, max_conv_depth)
    if encoder_in_channel_primary <= max_conv_depth/2:
        initial_conv_depth_primary = 2 * encoder_in_channel_primary
    else:
        initial_conv_depth_primary = max_conv_depth

    actual_encoder_depth = self.net_depth
    
    encoder_output_channels = min(initial_conv_depth_primary * (2 ** (actual_encoder_depth - 1)), max_conv_depth)
    bottleneck_in_channels = encoder_output_channels

    self.encoder = Encoder(in_channel=encoder_in_channel_primary, 
                         first_conv_depth=initial_conv_depth_primary, 
                         max_conv_depth=max_conv_depth,
                         data_num=self.data_num, 
                         depth=actual_encoder_depth, 
                         normalization=True, norm_type='BN')
    self.encoder.to(device=device)
    
    self.decoder_B = Decoder(output_channels=1, 
                          encoder_first_conv_depth=initial_conv_depth_primary,
                          encoder_max_conv_depth=max_conv_depth,
                          normalization=True,
                          norm_type='IN',
                          depth=actual_encoder_depth)
    self.decoder_B.to(device=device)

    self.decoder_F = Decoder(output_channels=2,
                          encoder_first_conv_depth=initial_conv_depth_primary,
                          encoder_max_conv_depth=max_conv_depth,
                          normalization=True,
                          norm_type='IN',
                          depth=actual_encoder_depth)
    self.decoder_F.to(device=device)

    self.bottleneck = DoubleConvBlock(
      in_depth=bottleneck_in_channels, 
      mid_depth=min(initial_conv_depth_primary * (2 ** actual_encoder_depth), max_conv_depth),
      out_depth=encoder_output_channels,
      pooling=False, normalization_block='Second', normalization=False, norm_type='IN')
    self.bottleneck.to(device=device)

    self.illum_hist_feature_encoder = CFEEncoder(size=self.input_size, n_layers=4, feat_ch=self.cfe_feature_num, max_depth=64,
                                                                norm_type='BN', normalization=True)
      
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, N, model_in_N, cm1, cm2):
    """ Forward function of CCMNet network

    Args:
      N: input histogram(s) (B x C_N x H x W, e.g., B x 2 x 64 x 64 for hist data without uv)
         Note: The original N included uv, model_in_N had it. Assuming N is just the core histogram here.
      model_in_N: input histogram(s) concatenated with uv coordinates and potentially CFE features.
                  Expected shape with current config: B x 1 x (4 or 4+cfe_feature_num) x H x W
      cm1: ColorMatrix1
      cm2: ColorMatrix2

    Returns:
      rgb: predicted illuminant rgb colors in the format b x 3.
      P: illuminant heat map.
      F: conv filter of the CCC model.
      B: bias of the CCC model.
      N_after_conv: Feature map after convolution with F, before adding B.
    """
    batch_size = N.shape[0]
    cm1_3x3 = torch.reshape(cm1, (batch_size, 3, 3))
    cm2_3x3 = torch.reshape(cm2, (batch_size, 3, 3))
    
    cfe_input_hist = self.get_illum_hist(cm1_3x3, cm2_3x3)
    cfe_features = self.illum_hist_feature_encoder(cfe_input_hist.unsqueeze(1))

    concat_features = cfe_features.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, N.shape[-2], N.shape[-1]) 
    model_in_N = torch.cat([model_in_N, concat_features], dim=2)

    latent, encoder_output = self.encoder(model_in_N)
    latent = self.bottleneck(latent)
    
    # Decoder B (Bias)
    B = self.decoder_B(latent, encoder_output)
    B = torch.squeeze(B)

    # Decoder F (Filter)
    F = self.decoder_F(latent, encoder_output)

    # Apply F and B to the original histogram N (which should be just the 2-channel histogram data)
    # N is expected to be B x 2 x H x W (chroma and edge histograms)
    if N.shape[1] > 2: # If N accidentally contains uv coordinates, take only first 2 channels
        N_hist_data = N[:, :2, :, :]
    else:
        N_hist_data = N
        
    N_fft = fft.rfft2(N_hist_data)
    F_fft = fft.rfft2(F)
    N_after_conv = fft.irfft2(N_fft * F_fft)

    N_after_conv = torch.sum(N_after_conv, dim=1)
    N_after_bias = N_after_conv + B

    N_after_bias = torch.clamp(N_after_bias, -100, 100)
    P = self.softmax(torch.reshape(N_after_bias, (batch_size, -1)))
    P = torch.reshape(P, N_after_bias.shape)

    u = torch.sum(P * self.u_coord, dim=[-1, -2])
    v = torch.sum(P * self.v_coord, dim=[-1, -2])
    u, v = ops.from_coord_to_uv(self.input_size, u, v)
    rgb = ops.uv_to_rgb(torch.stack([u, v], dim=1), tensor=True)
    
    return rgb, P, F, B, N_after_conv

  def get_illum_hist(self, cm1_3x3, cm2_3x3, colortemp_step=80):
    color_temps = torch.arange(2500, 7501, colortemp_step).to(self.device)
    xy_chromaticities = np.array([colour.temperature.CCT_to_xy(temp.item(), method="Kang 2002") for temp in color_temps])
    xy_chromaticities = torch.tensor(xy_chromaticities).to(self.device)
    z_chromaticities = 1 - xy_chromaticities[:, 0] - xy_chromaticities[:, 1]
    xyz_chromaticities = torch.stack([xy_chromaticities[:, 0], xy_chromaticities[:, 1], z_chromaticities], dim=1)
    xyz_chromaticities /= xyz_chromaticities[:, 1:2]
    xyz_chromaticities = xyz_chromaticities.float()

    batch_size = cm1_3x3.shape[0]
    num_temps = xyz_chromaticities.shape[0]

    g = torch.clip((1 / color_temps - 1 / 2856) / (1 / 6504 - 1 / 2856),0,1)
    g = g.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, 3, 3)

    cm1_expanded = cm1_3x3.unsqueeze(1).repeat(1, num_temps, 1, 1)
    cm2_expanded = cm2_3x3.unsqueeze(1).repeat(1, num_temps, 1, 1)

    cm = g * cm1_expanded + (1 - g) * cm2_expanded
    xyz_chromaticities = xyz_chromaticities.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)

    camera_rgbs = torch.einsum('bijk,bikl->bijl', cm, xyz_chromaticities).squeeze(-1)
    camera_rgbs = camera_rgbs.transpose(1, 2).unsqueeze(-1).float()

    histograms = self.histogram_layer(camera_rgbs).detach()

    return histograms

class Encoder(nn.Module):
  """ Encoder """

  def __init__(self, in_channel, first_conv_depth=48, max_conv_depth=32,
               data_num=1, normalization=False, norm_type='BN', depth=4):
    """ Encoder constructor

    Args:
      in_channel: number of channels of the primary input stream (e.g., original histogram + features).
      first_conv_depth: output channels produced by the first encoder layer for the primary stream.
      max_conv_depth: maximum output channels can be produced by any conv in the encoder.
      data_num: number of input streams (1).
      normalization: boolean flag to apply normalization in the encoder.
      norm_type: 'BN' or 'IN'.
      depth: number of encoder layers.

    Returns:
      Encoder object with the selected settings.
    """
    super().__init__()
    self.encoders = nn.ModuleList([])
    self.data_num = data_num
    self.encoder_depth = depth

    if self.data_num > 1:
      self.merge_layers = nn.ModuleList([])
      self.cross_pooling = CrossPooling() # Default merging strategy when data_num>1
    else:
      self.merge_layers = None
      self.cross_pooling = None

    for data_i in range(self.data_num):
      encoder_i = nn.ModuleList([])
      current_in_channel = in_channel
      if data_i > 0:
          current_in_channel = 4 
      
      current_first_conv_depth = first_conv_depth
      if data_i > 0:
          current_first_conv_depth = min(2 * current_in_channel, max_conv_depth)

      if self.data_num > 1:
        merge_layers_i = nn.ModuleList([])
      
      skip_connections = (data_i == 0) # Skip connections only for the primary stream (data_i=0)

      for block_j in range(self.encoder_depth):
        if block_j % 2 == 0 and normalization:
          norm_applied = normalization
        else:
          norm_applied = False

        if block_j == 0:
          block_in_depth = current_in_channel
          block_out_depth = min(current_first_conv_depth * (2 ** block_j), max_conv_depth)
        else:
          ref_first_conv_depth = first_conv_depth if data_i == 0 else current_first_conv_depth
          block_in_depth = min(ref_first_conv_depth * (2 ** (block_j - 1)), max_conv_depth)
          block_out_depth = min(ref_first_conv_depth * (2 ** block_j), max_conv_depth)

        double_conv_block = DoubleConvBlock(
          in_depth=block_in_depth,
          out_depth=block_out_depth,
          normalization=norm_applied, norm_type=norm_type,
          normalization_block='Second', return_before_pooling=skip_connections)
        encoder_i.append(double_conv_block)

        if self.data_num > 1 and block_j < self.encoder_depth - 1:
          merge_in_depth = 2 * block_out_depth
          merge_layer = ConvBlock(kernel=1,
                                  in_depth=merge_in_depth,
                                  conv_depth=block_out_depth,
                                  stride=1, padding=0,
                                  normalization=False,
                                  norm_type=norm_type, pooling=False)
          merge_layers_i.append(merge_layer)

      self.encoders.append(encoder_i)
      if self.data_num > 1:
        self.merge_layers.append(merge_layers_i)

  def forward(self, x):
    """ Forward function of Encoder module

    Args:
      x: input tensor. 
         If data_num=1: B x 1 x C x H x W (e.g. C = in_channel from __init__)
         If data_num=2: B x 2 x C_i x H x W (C_0 = in_channel, C_1 = 4 for illum_hist)
         The middle dimension (1 or 2) is the data_num (stream index).

    Returns:
      y: processed data by the encoder, which is the input to the bottleneck.
      skip_connection_data: a list of processed data by each encoder for
        u-net skip connections (only from the primary stream data_j=0).
    """
    skip_connection_data = [] # Only from primary stream (j=0)
    latent_x = [None] * self.data_num # Use a list to store latents for each stream

    for encoder_block_i in range(self.encoder_depth):
      stacked_latent_for_pooling = [] # Collect latents at this depth for pooling/merging
      for data_j in range(self.data_num):
        current_input = x[:, data_j, :, :, :] if encoder_block_i == 0 else latent_x[data_j]
        
        if data_j == 0: # Primary stream, may have skip connections
          processed_latent, latent_before_pooling = self.encoders[data_j][encoder_block_i](current_input)
          skip_connection_data.append(latent_before_pooling)
        else: # Secondary streams, no skip connections returned from here
          processed_latent = self.encoders[data_j][encoder_block_i](current_input)
        
        latent_x[data_j] = processed_latent
        if self.data_num > 1:
            stacked_latent_for_pooling.append(latent_x[data_j])

      if self.data_num > 1:
        stacked_tensor = torch.stack(stacked_latent_for_pooling, dim=-1)
        pooled_data = self.cross_pooling(stacked_tensor)
        
        if encoder_block_i < (self.encoder_depth - 1):
          for data_j in range(self.data_num):
            merge_layer_input = torch.cat([pooled_data, latent_x[data_j]], dim=1)
            latent_x[data_j] = self.merge_layers[data_j][encoder_block_i](merge_layer_input)
            
    if self.data_num == 1:
      y = latent_x[0]
    else:
      y = pooled_data

    skip_connection_data.reverse() # Match U-Net decoder order
    return y, skip_connection_data


class Decoder(nn.Module):
  """ Decoder """

  def __init__(self, output_channels, encoder_first_conv_depth=8,
               normalization=False, encoder_max_conv_depth=32,
               norm_type='IN', depth=4):
    """ Decoder constructor

    Args:
      output_channels: output channels of the last layer in the decoder.
      encoder_first_conv_depth: output channels produced by the first encoder
        layer; default is 8. This and 'encoder_max_conv_depth' variables are
        used to dynamically compute the output of each corresponding decoder
        layer.
      normalization: boolean flag to apply normalization in the decoder;
        default is false.
      encoder_max_conv_depth: maximum output channels can be produced by any cov
        in the encoder; default is 32. This and 'encoder_first_conv_depth'
        variables are used to dynamically compute the output of each
        corresponding decoder layer. This variable also is used to know the
        output of the bottleneck unite.
      norm_type: when 'normalization' is set to true, the value of this variable
        (i.e., norm_type) specifies which normalization process is applied.
        'BN' refers to batch normalization and 'IN' (default) refers to instance
        normalization.
      depth: number of encoder layers; default is 4.

    Returns:
      Decoder object with the selected settings.
    """
    super().__init__()
    self.decoder = nn.ModuleList([])
    self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
    self.final_block = ConvBlock(kernel=3, activation=False, in_depth=int(
      encoder_first_conv_depth / 2), conv_depth=output_channels,
                                 stride=1, padding=1)
    
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    for i in range(depth):
      mid_depth = int(min(encoder_first_conv_depth * 2 ** (depth - 1 - i),
                          encoder_max_conv_depth))
      out_depth = int(min(encoder_first_conv_depth * 2 ** (depth - 2 - i),
                          encoder_max_conv_depth))
      in_depth = 2 * mid_depth

      double_conv_block = DoubleConvBlock(
        in_depth=in_depth, out_depth=out_depth, mid_depth=mid_depth,
        normalization_block='Second',
        normalization=normalization, norm_type=norm_type, pooling=False)
      self.decoder.append(double_conv_block)

  def forward(self, x, encoder_output):
    """ Forward function of Decoder module

    Args:
      x: processed data by the bottleneck
      encoder_output: list of skipped data from the encoder layers

    Returns:
      tensor of one the CCC model components (i.e., F, B) emitted by the
        network.
    """
    for i, decoder_block in enumerate(self.decoder):
      x = self.upsampling(x)
      
      if i < len(encoder_output):
          x = torch.cat([encoder_output[i], x], dim=1)
      else:
          pass 
      
      x = decoder_block(x)

    return self.final_block(x)


class CFEEncoder(nn.Module):
  def __init__(self, size, n_layers, feat_ch, max_depth, norm_type='BN', normalization=True):
    super(CFEEncoder, self).__init__()
    self.size = size
    self.n_layers = n_layers
    self.feat_ch = feat_ch
    self.max_depth = max_depth
    self.norm_type = norm_type
    self.normalization = normalization

    layers = []
    in_depth = 1
    out_depth = 16

    for i in range(n_layers):
      if i > 0:
        in_depth = min(out_depth, max_depth)
        out_depth = min(out_depth * 2, max_depth)
      
      # DoubleConvBlock is expected to be available from src.modules.layers
      layers.append(DoubleConvBlock(
        in_depth=in_depth,
        out_depth=out_depth,
        normalization=self.normalization,
        norm_type=self.norm_type,
        normalization_block='Second',
        pooling=True
      ))

    self.encoder = nn.Sequential(*layers)
    
    final_spatial_size = size // (2 ** n_layers)
    flat_size = out_depth * (final_spatial_size ** 2)
    
    self.fc = nn.Sequential(
      nn.Linear(flat_size, flat_size//2),
      nn.LeakyReLU(inplace=True),
      nn.Linear(flat_size//2, feat_ch)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

if __name__ == "__main__":
  # Test the network with dummy data
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  input_size = 64
  batch_size = 2
  cfe_feat_num = 8
  net_depth = 4
  max_conv_depth = 32

  # Initialize the network
  net = network(input_size=input_size,
                cfe_feature_num=cfe_feat_num,
                net_depth=net_depth,
                max_conv_depth=max_conv_depth,
                device=device)
  net.to(device)
  net.eval()

  # Create dummy input data
  # N: B x C_N x H x W (e.g., B x 2 x 64 x 64 for hist data, actual channels might vary)
  # For this test, assuming N is the base histogram part and model_in_N is constructed with coordinates
  dummy_N = torch.randn(batch_size, 2, input_size, input_size).to(device)
  
  # model_in_N: B x 1 x C_model_in x H x W
  # C_model_in is 4 (u, v, N_ch1, N_ch2) if CFE is not concatenated here
  # So, initial model_in_N has 4 channels (2 for histogram, 2 for uv coordinates)
  uv_coords_u, uv_coords_v = ops.get_uv_coord(input_size, tensor=True, device=device)
  uv_coords_u = uv_coords_u.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
  uv_coords_v = uv_coords_v.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
  
  # Assuming N has 2 channels as per dummy_N and they are part of model_in_N
  # Constructing model_in_N similarly to how it might be prepared before CFE concat
  model_in_N_base = torch.cat([dummy_N.unsqueeze(1), uv_coords_u, uv_coords_v], dim=2) # B x 1 x 4 x H x W

  dummy_cm1 = torch.randn(batch_size, 9).to(device)
  dummy_cm2 = torch.randn(batch_size, 9).to(device)

  # Perform a forward pass
  print("Performing forward pass...")
  with torch.no_grad():
    rgb, P, F, B, N_after_conv = net(dummy_N, model_in_N_base, dummy_cm1, dummy_cm2)
  print("Forward pass completed.")

  # Print output shapes
  print(f"Output rgb shape: {rgb.shape}")         # Expected: B x 3
  print(f"Output P shape: {P.shape}")           # Expected: B x H x W (e.g., B x 64 x 64)
  print(f"Output F shape: {F.shape}")           # Expected: B x 2 x K x K (e.g., B x 2 x 1 x 1 or other kernel size)
  print(f"Output B shape: {B.shape}")           # Expected: B x H x W (e.g., B x 64 x 64)
  print(f"N_after_conv shape: {N_after_conv.shape}") # Expected: B x 1 x H x W

  print("Test finished.")