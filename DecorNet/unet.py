# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

__all__ = ["UNet", "Unet"]


# @export("monai.networks.nets")
# @alias("Unet")
class UNet(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            if len(channels) > 3:
                return self._get_connection_block(down, up, subblock)
            else:
                return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


Unet = UNet

class TE_block_v2(nn.Module):
    def __init__(self,ch_in: int,kernel_size: int = 5,strides: int = 1,dropout=0.0,):
        super().__init__()
        # weight
        self.Wt = torch.nn.Parameter(torch.randn(1))
        self.Wt.requires_grad = True
        self.Ws = torch.nn.Parameter(torch.randn(1))
        self.Ws.requires_grad = True
        
        # conv_texture
        self.cont_texture = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size,stride=strides,padding=int(kernel_size/2),bias=False)
        lp_filter = np.ones((kernel_size,kernel_size))/(kernel_size**2)
        hp_filter = -lp_filter
        hp_filter[int(kernel_size/2),int(kernel_size/2)]+= 1
        conv_filter = np.zeros((ch_in,ch_in,kernel_size,kernel_size))/(kernel_size**2)
        
        for i in range(ch_in):
            conv_filter[i,i,:,:] = hp_filter
        self.cont_texture.weight = torch.nn.Parameter(torch.tensor(conv_filter).type(torch.float32))
#         self.cont_texture.weight.requires_grad = False
#         self.cont_texture.requires_grad = False
        self.act = nn.PReLU()
    
        # dimension reduction
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_in, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.PReLU(),)

#         print(f'self.Wt {self.Wt}')
#         print(f'self.Ws {self.Ws}')
#         print(f'self.cont_texture.weight {self.cont_texture.weight}')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t: torch.Tensor = self.cont_texture(x)
        x_t: torch.Tensor = self.act(x_t) 
        x_out: torch.Tensor = torch.cat((x_t,x),dim=1)   
        x_out: torch.Tensor = self.conv_1x1(x_out) 
        return x_out

class UNet_TEv2(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            
            if len(channels) > 3:
                self.TE = TE_block_v2(inc)
                return nn.Sequential(self.TE,self._get_connection_block(down, up, subblock))
            else:
                return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


from torch.nn import init
class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TE_block_v3(nn.Module):
    def __init__(self,ch_in: int,kernel_size: int = 5,strides: int = 1,dropout=0.0,):
        super().__init__()
        # weight
#         self.Wt = torch.nn.Parameter(torch.randn(1))
#         self.Wt.requires_grad = True
#         self.Ws = torch.nn.Parameter(torch.randn(1))
#         self.Ws.requires_grad = True
        
        # conv_texture
        self.conv_texture = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size,stride=strides,padding=int(kernel_size/2),bias=False)
        lp_filter = np.ones((kernel_size,kernel_size))/(kernel_size**2)
        hp_filter = -lp_filter
        hp_filter[int(kernel_size/2),int(kernel_size/2)]+= 1
        conv_filter = np.zeros((ch_in,ch_in,kernel_size,kernel_size))/(kernel_size**2)
        
        for i in range(ch_in):
            conv_filter[i,i,:,:] = hp_filter
        self.conv_texture.weight = torch.nn.Parameter(torch.tensor(conv_filter).type(torch.float32))
#         self.conv_texture.weight.requires_grad = False
#         self.conv_texture.requires_grad = False
        self.act = nn.PReLU()
    
        # SE block
        if ch_in>1:
            self.SEattention = SEAttention(channel = ch_in, reduction=8)
        else:
            self.SEattention = SEAttention(channel = ch_in, reduction=1)
            
        # dimension reduction
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_in, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.PReLU(),)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t: torch.Tensor = self.conv_texture(x)
        x_t: torch.Tensor = self.act(x_t) 
        x_t: torch.Tensor = self.SEattention(x_t)    
        x_out: torch.Tensor = torch.cat((x_t,x),dim=1)   
        x_out: torch.Tensor = self.conv_1x1(x_out) 
        return x_out

class UNet_TEv3(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            
            if len(channels) > 3:
                self.TE = TE_block_v3(inc)
                return nn.Sequential(self.TE,self._get_connection_block(down, up, subblock))
            else:
                return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

class UNet_TEv3_2(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            
            if len(channels) > 3:
                self.TE = TE_block_v3(inc,kernel_size = 7)
                return nn.Sequential(self.TE,self._get_connection_block(down, up, subblock))
            else:
                return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

class TE_block_v3_1(nn.Module):
    def __init__(self,ch_in: int,kernel_size: int = 5,strides: int = 1,dropout=0.0,):
        super().__init__()
        # weight
#         self.Wt = torch.nn.Parameter(torch.randn(1))
#         self.Wt.requires_grad = True
#         self.Ws = torch.nn.Parameter(torch.randn(1))
#         self.Ws.requires_grad = True
        
        # conv_texture
#         self.conv_texture = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size,stride=strides,padding=int(kernel_size/2),bias=False)
#         lp_filter = np.ones((kernel_size,kernel_size))/(kernel_size**2)
#         hp_filter = -lp_filter
#         hp_filter[int(kernel_size/2),int(kernel_size/2)]+= 1
#         conv_filter = np.zeros((ch_in,ch_in,kernel_size,kernel_size))/(kernel_size**2)

#         for i in range(ch_in):
#             conv_filter[i,i,:,:] = hp_filter
#         self.conv_texture.weight = torch.nn.Parameter(torch.tensor(conv_filter).type(torch.float32))
# #         self.conv_texture.weight.requires_grad = False
# #         self.conv_texture.requires_grad = False
#         self.act = nn.PReLU()

        # SE block
        if ch_in>1:
            self.SEattention = SEAttention(channel = ch_in, reduction=8)
        else:
            self.SEattention = SEAttention(channel = ch_in, reduction=1)
            
        # dimension reduction
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_in, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.PReLU(),)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_t: torch.Tensor = self.conv_texture(x)
#         x_t: torch.Tensor = self.act(x_t) 
        x_t: torch.Tensor = self.SEattention(x)    
        x_out: torch.Tensor = torch.cat((x_t,x),dim=1)   
        x_out: torch.Tensor = self.conv_1x1(x_out) 
        return x_out

class UNet_TEv3_1(nn.Module):
    '''
    Only attention no conv.
    '''
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            
            if len(channels) > 3:
                self.TE = TE_block_v3_1(inc)
                return nn.Sequential(self.TE,self._get_connection_block(down, up, subblock))
            else:
                return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
class UNet_5layers_SEattention(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)
        
        self.SE0 = SEAttention(channel = channels[0], reduction=8)
        self.SE1 = SEAttention(channel = channels[1], reduction=8)
        self.SE2 = SEAttention(channel = channels[2], reduction=8)
        self.SE3 = SEAttention(channel = channels[3], reduction=8)

        
    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        self.x1 = self._get_down_layer1(x)
        self.x1 = self.SE0(self.x1)
        
        self.x2 = self._get_down_layer2(self.x1) 
        self.x2 = self.SE1(self.x2)
        
        self.x3 = self._get_down_layer3(self.x2) 
        self.x3 = self.SE2(self.x3)
        
        self.x4 = self._get_down_layer4(self.x3) 
        self.x4 = self.SE3(self.x4)
        
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))


        return self.x9
class TE_block_v3_depthwise(nn.Module):
    def __init__(self,ch_in: int,kernel_size: int = 7,strides: int = 1,dropout=0.0,):
        super().__init__()
        # weight
#         self.Wt = torch.nn.Parameter(torch.randn(1))
#         self.Wt.requires_grad = True
#         self.Ws = torch.nn.Parameter(torch.randn(1))
#         self.Ws.requires_grad = True
        
        # depth-wise conv_texture
        self.conv_texture = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size,stride=strides,padding=int(kernel_size/2),bias=False, groups = ch_in)
        lp_filter = np.ones((kernel_size,kernel_size))/(kernel_size**2)
        hp_filter = -lp_filter
        hp_filter[int(kernel_size/2),int(kernel_size/2)]+= 1
        conv_filter = np.zeros((ch_in,1,kernel_size,kernel_size))/(kernel_size**2)
        
        for i in range(ch_in):
            conv_filter[i,0,:,:] = hp_filter
        self.conv_texture.weight = torch.nn.Parameter(torch.tensor(conv_filter).type(torch.float32))
        
        # point wise conv
        self.point_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=1)
        
        self.act = nn.PReLU()
    
        # SE block
        if ch_in>1:
            self.SEattention = SEAttention(channel = ch_in, reduction=8)
        else:
            self.SEattention = SEAttention(channel = ch_in, reduction=1)
            
        # dimension reduction
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_in, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.PReLU(),)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t: torch.Tensor = self.conv_texture(x)
        x_t: torch.Tensor = self.point_conv(x_t)
        x_t: torch.Tensor = self.act(x_t) 
        x_t: torch.Tensor = self.SEattention(x_t)    
        x_out: torch.Tensor = torch.cat((x_t,x),dim=1)   
        x_out: torch.Tensor = self.conv_1x1(x_out) 
        return x_out

class UNet_5layers_depth_wise(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
#         self._get_down_layer5 = self._get_down_layer(channels[3], channels[4], strides[4], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)
        
        self.TE1 = TE_block_v3_depthwise(self.in_channels)
        self.TE2 = TE_block_v3_depthwise(channels[0])

        
    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        self.x = self.TE1(x)
        self.x1 = self._get_down_layer1(self.x)
        
        self.x1 = self.TE2(self.x1)
        self.x2 = self._get_down_layer2(self.x1) 
        self.x3 = self._get_down_layer3(self.x2) 
        self.x4 = self._get_down_layer4(self.x3) 
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))


        return self.x9
class UNet_5layers(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
#         self._get_down_layer5 = self._get_down_layer(channels[3], channels[4], strides[4], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)

        
    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x: torch.Tensor, layer: str = None) -> torch.Tensor:
        
        self.x1 = self._get_down_layer1(x)
        self.x2 = self._get_down_layer2(self.x1) 
        self.x3 = self._get_down_layer3(self.x2) 
        self.x4 = self._get_down_layer4(self.x3) 
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))

        if not (layer is None):
            print(layer)
            h = eval(layer).register_hook(self.activations_hook)
            
        return self.x9

class UNet_5layersDIY(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
#         self._get_down_layer5 = self._get_down_layer(channels[3], channels[4], strides[4], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]+channels[7], self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]+channels[6], channels[7], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]+channels[5], channels[6], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[5], strides[3], is_top = False)

        
    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x: torch.Tensor, layer: str = None) -> torch.Tensor:
        
        self.x1 = self._get_down_layer1(x)
        self.x2 = self._get_down_layer2(self.x1) 
        self.x3 = self._get_down_layer3(self.x2) 
        self.x4 = self._get_down_layer4(self.x3) 
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))

        if not (layer is None):
            print(layer)
            h = eval(layer).register_hook(self.activations_hook)
            
        return self.x9

from torch.nn import init
class TextureAttention(nn.Module):

    def __init__(self, in_channel=512,out_channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, img,texture):
        b_img,c_img,_,_ = img.size()
        b, c, _, _ = texture.size()
        y_texture = self.avg_pool(texture).view(b, c)
        y_img = self.avg_pool(img).view(b_img, c_img)
        y = torch.cat((y_texture,y_img),axis = 1)
        y = self.fc(y).view(b, c, 1, 1)
        return texture * y.expand_as(texture)

class UNet_5layers_GLCMv1(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        in_channels_texture: Optional[int] = 10,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0]*2, channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)
        
        # GLCM block
        self._get_down_layer1_GLCM = self._get_down_layer(in_channels_texture, channels[0], strides[0], is_top=True)
        self.texture_attention = TextureAttention(channels[0]+channels[0],channels[0],8)
        
        
    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x ) -> torch.Tensor:
        
        '''
        Args:
            x[:,0,:,:]: input image
            x[:,1:,:,:]: input feature
        '''
        x_img = x[:,0:1,:,:]
        x_texture = x[:,1:,:,:]
        self.x1 = self._get_down_layer1(x_img)
        
        self.texture1 = self._get_down_layer1_GLCM(x_texture)
        self.texture1_attention = self.texture_attention(self.x1,self.texture1)
        self.x2_input = torch.cat((self.x1,self.texture1_attention),axis=1)
        
        self.x2 = self._get_down_layer2(self.x2_input) 
        self.x3 = self._get_down_layer3(self.x2) 
        self.x4 = self._get_down_layer4(self.x3) 
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))


        return self.x9

from torch.nn import init
class Global_context(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        self.y = self.fc(y).view(b, c, 1, 1)
        return x + self.y.expand_as(x)

class UNet_5layers_GC(nn.Module):
    
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.gradients = None

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)
        
        self.Gloval_context1 = Global_context(channel = channels[0], reduction=8)
        self.Gloval_context2 = Global_context(channel = channels[1], reduction=8)
        self.Gloval_context3 = Global_context(channel = channels[2], reduction=8)
        self.Gloval_context4 = Global_context(channel = channels[3], reduction=8)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x: torch.Tensor, layer: str = None) -> torch.Tensor:
        
        self.x1 = self._get_down_layer1(x)
        self.x1 = self.Gloval_context1(self.x1)
        
        self.x2 = self._get_down_layer2(self.x1) 
        self.x2 = self.Gloval_context2(self.x2)
        
        self.x3 = self._get_down_layer3(self.x2) 
        self.x3 = self.Gloval_context3(self.x3)
        
        self.x4 = self._get_down_layer4(self.x3) 
        self.x4 = self.Gloval_context4(self.x4)
        
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))
        
        if not (layer is None):
            print(layer)
            h = eval(layer).register_hook(self.activations_hook)

        return self.x9

class UNet_5layers_i2i4(nn.Module):
    '''
    combine idea 2 and idea 4
    '''
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)
        
        self.Gloval_context1 = Global_context(channel = channels[0], reduction=8)
        self.Gloval_context2 = Global_context(channel = channels[1], reduction=8)
        self.Gloval_context3 = Global_context(channel = channels[2], reduction=8)
        self.Gloval_context4 = Global_context(channel = channels[3], reduction=8)
        
        self.TE1 = TE_block_v3(self.in_channels)
        self.TE2 = TE_block_v3(channels[0])

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        self.xTE1 = self.TE1(x)
        self.x1 = self._get_down_layer1(self.xTE1)
        self.x1 = self.Gloval_context1(self.x1)
        
        self.xTE2 = self.TE2(self.x1)
        self.x2 = self._get_down_layer2(self.xTE2) 
        self.x2 = self.Gloval_context2(self.x2)
        
        self.x3 = self._get_down_layer3(self.x2) 
        self.x3 = self.Gloval_context3(self.x3)
        
        self.x4 = self._get_down_layer4(self.x3) 
        self.x4 = self.Gloval_context4(self.x4)
        
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))

        return self.x9

class TE_block_v4(nn.Module):
    '''
    Used depth-wise convolution
    '''
    def __init__(self,ch_in: int,kernel_size: int = 5,strides: int = 1,dropout=0.0,):
        super().__init__()
        # weight
        self.Wt = torch.nn.Parameter(torch.randn(1))
        self.Wt.requires_grad = True
        self.Ws = torch.nn.Parameter(torch.randn(1))
        self.Ws.requires_grad = True
        
        # conv_texture
        self.conv_texture = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size,stride=strides,padding=int(kernel_size/2),bias=False,groups = ch_in)
        lp_filter = np.ones((kernel_size,kernel_size))/(kernel_size**2)
        hp_filter = -lp_filter
        hp_filter[int(kernel_size/2),int(kernel_size/2)]+= 1
        conv_filter = np.zeros((ch_in,1,kernel_size,kernel_size))/(kernel_size**2)
        
        for i in range(ch_in):
            conv_filter[i,0,:,:] = hp_filter
        self.conv_texture.weight = torch.nn.Parameter(torch.tensor(conv_filter).type(torch.float32))
        self.act = nn.PReLU()
    
        # SE block
        if ch_in>=8:
            self.SEattention = SEAttention(channel = ch_in, reduction=8)
        else:
            self.SEattention = SEAttention(channel = ch_in, reduction=1)
            
        # dimension reduction
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_in, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.PReLU(),)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t: torch.Tensor = self.conv_texture(x)
        x_t: torch.Tensor = self.act(x_t) 
        x_t: torch.Tensor = self.SEattention(x_t)    
        x_out: torch.Tensor = torch.cat((x_t,x),dim=1)   
        x_out: torch.Tensor = self.conv_1x1(x_out) 
        return x_out

class TE_block_v5(nn.Module):
    '''
    Used depth-wise convolution and adding
    '''
    def __init__(self,ch_in: int,kernel_size: int = 5,strides: int = 1,dropout=0.0,):
        super().__init__()
        # weight
        self.Wt = torch.nn.Parameter(torch.randn(1))
        self.Wt.requires_grad = True
        self.Ws = torch.nn.Parameter(torch.randn(1))
        self.Ws.requires_grad = True
        
        # conv_texture
        self.conv_texture = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size,stride=strides,padding=int(kernel_size/2),bias=False, groups = ch_in)
        lp_filter = np.ones((kernel_size,kernel_size))/(kernel_size**2)
        hp_filter = -lp_filter
        hp_filter[int(kernel_size/2),int(kernel_size/2)]+= 1
        conv_filter = np.zeros((ch_in,1,kernel_size,kernel_size))/(kernel_size**2)
        
        for i in range(ch_in):
            conv_filter[i,0,:,:] = hp_filter
        self.conv_texture.weight = torch.nn.Parameter(torch.tensor(conv_filter).type(torch.float32))
        self.act = nn.PReLU()
    
        # SE block
        if ch_in>=8:
            self.SEattention = SEAttention(channel = ch_in, reduction=8)
        else:
            self.SEattention = SEAttention(channel = ch_in, reduction=1)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t: torch.Tensor = self.conv_texture(x)
        x_t: torch.Tensor = self.act(x_t) 
        x_t: torch.Tensor = self.SEattention(x_t)    
        return x_t+x

class UNet_5layers_i2i4_v2(nn.Module):
    '''
    Combine idea 2 and idea 4
    Used depth-wise convolution for texture enhancement block
    '''
    
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)
        
        self.Gloval_context1 = Global_context(channel = channels[0], reduction=8)
        self.Gloval_context2 = Global_context(channel = channels[1], reduction=8)
        self.Gloval_context3 = Global_context(channel = channels[2], reduction=8)
        self.Gloval_context4 = Global_context(channel = channels[3], reduction=8)
        
        self.TE1 = TE_block_v4(self.in_channels)
        self.TE2 = TE_block_v4(channels[0])

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        self.xTE1 = self.TE1(x)
        self.x1 = self._get_down_layer1(self.xTE1)
        self.x1 = self.Gloval_context1(self.x1)
        
        self.xTE2 = self.TE2(self.x1)
        self.x2 = self._get_down_layer2(self.xTE2) 
        self.x2 = self.Gloval_context2(self.x2)
        
        self.x3 = self._get_down_layer3(self.x2) 
        self.x3 = self.Gloval_context3(self.x3)
        
        self.x4 = self._get_down_layer4(self.x3) 
        self.x4 = self.Gloval_context4(self.x4)
        
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))

        return self.x9

class UNet_5layers_i2i4_v2_1(nn.Module):
    '''
    Combine idea 2 and idea 4
    Used depth-wise convolution for texture enhancement block
    Use adding instead of concatenation for texture enhancement block
    '''
    
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # build the block
        self._get_down_layer1 = self._get_down_layer(self.in_channels, channels[0], strides[0], is_top=True)
        self._get_down_layer2 = self._get_down_layer(channels[0], channels[1], strides[1], is_top=False)
        self._get_down_layer3 = self._get_down_layer(channels[1], channels[2], strides[2], is_top=False)
        self._get_down_layer4 = self._get_down_layer(channels[2], channels[3], strides[3], is_top=False)
        self._get_down_layer5 = self._get_bottom_layer(channels[3], channels[4])
        
        
        self._get_up_layer1 = self._get_up_layer(channels[0]*2, self.out_channels, strides[0], is_top = True)
        self._get_up_layer2 = self._get_up_layer(channels[1]*2, channels[0], strides[1], is_top = False)
        self._get_up_layer3 = self._get_up_layer(channels[2]*2, channels[1], strides[2], is_top = False)
        self._get_up_layer4 = self._get_up_layer(channels[3]+channels[4], channels[2], strides[3], is_top = False)
        
        self.Gloval_context1 = Global_context(channel = channels[0], reduction=8)
        self.Gloval_context2 = Global_context(channel = channels[1], reduction=8)
        self.Gloval_context3 = Global_context(channel = channels[2], reduction=8)
        self.Gloval_context4 = Global_context(channel = channels[3], reduction=8)
        
        self.TE1 = TE_block_v5(self.in_channels)
        self.TE2 = TE_block_v5(channels[0])

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        self.xTE1 = self.TE1(x)
        self.x1 = self._get_down_layer1(self.xTE1)
        self.x1 = self.Gloval_context1(self.x1)
        
        self.xTE2 = self.TE2(self.x1)
        self.x2 = self._get_down_layer2(self.xTE2) 
        self.x2 = self.Gloval_context2(self.x2)
        
        self.x3 = self._get_down_layer3(self.x2) 
        self.x3 = self.Gloval_context3(self.x3)
        
        self.x4 = self._get_down_layer4(self.x3) 
        self.x4 = self.Gloval_context4(self.x4)
        
        self.x5 = self._get_down_layer5(self.x4) 
        
        
        self.x6 = self._get_up_layer4(torch.cat([self.x5, self.x4], dim=1)) 
        self.x7 = self._get_up_layer3(torch.cat([self.x6, self.x3], dim=1))
        self.x8 = self._get_up_layer2(torch.cat([self.x7, self.x2], dim=1))
        self.x9 = self._get_up_layer1(torch.cat([self.x8, self.x1], dim=1))

        return self.x9
