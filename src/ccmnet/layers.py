import sys, math
sys.path.extend(['../..', '..'])

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """ Conv layer block. """

    def __init__(self, kernel, in_depth, conv_depth, stride=1, padding=1,
                 normalization=False, norm_type='BN', pooling=False,
                 bias_initialization='zeros', activation=True, dilation=1,
                 return_before_pooling=False):
        """ 
        ConvBlock constructor.

        Args:
            kernel (int): kernel size
            in_depth (int): depth of input tensor
            conv_depth (int): number of out channels produced by the convolution
            stride (int): stide of the convolution; default is 1
            padding (int): zero-padding added to both sides before the convolution
            normalization (boolean): boolean flag to apply normalization after the conv; default is false
            norm_type (str): normalization operation; 'BN' for batch-norm (default), 'IN' for instance normalization
            pooling (bool): boolean flag to apply a 2 x 2 max-pooling with stride of 2
                            before returning the final result; default is false
            bias_initialization (str): bias initialization; 'zeros' (default) or 'ones'
            activation (bool): boolean flag to apply a leaky ReLU activation; default is true
            dilation (int): spacing between conv kernel elements; default is 1
            return_before_pooling (bool): boolean flag to return the tensor before
                                          applying max-pooling (if 'pooling' is true); default is false

        Returns:
            ConvBlock object with the selected settings.
        """

        super(ConvBlock, self).__init__()

        conv = torch.nn.Conv2d(in_depth, conv_depth, kernel, stride=stride,
                               dilation=dilation, padding=padding,
                               padding_mode='replicate')
        torch.nn.init.kaiming_normal_(conv.weight)  # He initialization
        if bias_initialization == 'ones':
            torch.nn.init.ones_(conv.bias)
        elif bias_initialization == 'zeros':
            torch.nn.init.zeros_(conv.bias)
        else:
            raise NotImplementedError

        if activation:
            self.activation = torch.nn.LeakyReLU(inplace=False)
        else:
            self.activation = None
        if normalization:
            if norm_type == 'BN':
                self.normalization = torch.nn.BatchNorm2d(conv_depth, affine=True)
            elif norm_type == 'IN':
                self.normalization = torch.nn.InstanceNorm2d(conv_depth,
                                                             affine=False)
            else:
                raise NotImplementedError
        else:
            self.normalization = None
        self.conv = conv
        if pooling:
            self.pooling = torch.nn.MaxPool2d(2, stride=2)
        else:
            self.pooling = None
        self.return_before_pooling = return_before_pooling

    def forward(self, x):
        """
        Forward function of ConvBlock module.

        Args:
          x: input tensor

        Returns:
          y: processed tensor
        """
        x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pooling is not None:
            y = self.pooling(x)
        else:
            y = x
        if self.return_before_pooling:
            return y, x
        else:
            return y


class DoubleConvBlock(nn.Module):
    """ Double conv layers block. """

    def __init__(self, in_depth, out_depth, mid_depth=None, kernel=3, stride=1,
                 padding=None, dilation=None, normalization=False, norm_type='BN',
                 pooling=True, return_before_pooling=False,
                 normalization_block='Both'):
        """
        DoubleConvBlock constructor.

        Args:
            in_depth (int): depth of input tensor
            out_depth (int): number of out channels produced by the second convolution
            mid_depth (int): number of out channels produced by the first convolution; default is mid_depth = out_depth
            kernel (int): kernel size; default is 3
            stride (int): stide of the convolution; default is 1.
            padding (list): zero-padding added to both sides before the convolution operations; default is [1, 1]
            dilation (list): spacing between elements of each conv kernel; default is [1, 1]
            normalization (bool): boolean flag to apply normalization after the conv; default is false
            norm_type (str): normalization operation; 'BN' for batch-norm (default), 'IN' for instance normalization
            pooling (bool): boolean flag to apply a 2 x 2 max-pooling with stride of 2 before returning the final result; default is false
            return_before_pooling (bool): boolean flag to return the tensor before applying max-pooling (if 'pooling' is true); default is false
            normalization_block (str): if normalization flag is set to true, this variable controls
                                        when to apply the normalization process

        Returns:
            DoubleConvBlock object with the selected settings.
        """

        super().__init__()
        if padding is None:
            padding = [1, 1]
        if dilation is None:
            dilation = [1, 1]
        if mid_depth is None:
            mid_depth = out_depth
        if normalization:
            if normalization_block == 'First':
                norm = [True, False]
            elif normalization_block == 'Second':
                norm = [False, True]
            elif normalization_block == 'Both':
                norm = [True, True]
            else:
                raise NotImplementedError
        else:
            norm = [False, False]
        self.double_conv_1 = ConvBlock(kernel=kernel, in_depth=in_depth,
                                       conv_depth=mid_depth, stride=stride,
                                       padding=padding[0], pooling=False,
                                       dilation=dilation[0],
                                       norm_type=norm_type,
                                       normalization=norm[0])
        self.double_conv_2 = ConvBlock(kernel=kernel, in_depth=mid_depth,
                                       conv_depth=out_depth, stride=stride,
                                       padding=padding[1], pooling=pooling,
                                       dilation=dilation[1],
                                       norm_type=norm_type,
                                       normalization=norm[1],
                                       return_before_pooling=
                                       return_before_pooling)

    def forward(self, x):
        """ Forward function of DoubleConvBlock module.

        Args:
          x: input tensor

        Returns:
          y: processed tensor
        """

        x = self.double_conv_1(x)
        return self.double_conv_2(x)
    

class Flatten(nn.Module):
  """ Flattening. """

  def forward(self, x):
    """ Forward function of Flatten module.

    Args:
      x: input tensor with a total number of values = n

    Returns:
      an batch x n vector
    """

    x = x.view(x.size()[0], -1)
    return x


class CrossPooling(nn.Module):
  """ Cross pooling. """

  def forward(self, x):
    """ Forward function of CrossPooling module.

    Args:
      x: a stack of (batch x channel x height x width) tensors on the last axis

    Returns:
      A (batch x channel x height x width) tensor after applying max-pooling over the last axis
    """

    x, _ = torch.max(x, dim=-1)
    return x

