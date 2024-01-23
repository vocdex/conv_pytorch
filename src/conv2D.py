import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from src import utils
import math
torch.manual_seed(0)


class CustomConv2D(nn.Module):
    """Custom 2D convolutional layer that only works with square kernels.
     Arguments:
     in_channels (int): Number of input channels.
     out_channels (int): Number of output channels produced by the convolution.
     kernel_size (int or tuple): Size of the convolutional kernel.
     stride (int or tuple, optional): Stride of the convolution. Default: 1
     padding (int or tuple, optional): Zero-padding added to the input. Default: 0
     dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
     bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    """
    def __init__(
                self, 
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Union[int, Tuple[int, int]] = 0,
                dilation: Union[int, Tuple[int, int]] = 1,
                bias: bool = True,
                ):
        super(CustomConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = utils.unpair(kernel_size)
        self.stride = utils.unpair(stride)
        self.padding = utils.unpair(padding)
        self.dilation = utils.unpair(dilation)
        self.bias = bias
        self.weight = nn.Parameter(
            torch.empty(
                (
                    self.out_channels,
                    self.in_channels * self.kernel_size * self.kernel_size,
                )
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros((self.out_channels)))
    
    def get_output_size(self, input_size:int) -> int:
        """
        Calculates the output size (i.e. feature map size) of the convolutional layer.
        The width and height are the same for all tensors.

        Arguments:
        input_size: Height or width of input tensor.
        """
        return (input_size + 2 * self.padding - self.dilation*(self.kernel_size-1)-1) // self.stride + 1

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            print(f'Input tensor has {input.dim()} dimensions. Adding a batch dimension.')
            input = input.unsqueeze(0)
        elif input.dim() ==2:
            print(f'Input tensor has {input.dim()} dimensions. Adding a batch and channel dimension.')
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() != 4:
            raise ValueError(f'Input tensor must have 2, 3 or 4 dimensions. Got {input.dim()} dimensions.')
        
        height, width = input.shape[-2:]

        input_unfold = F.unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        weight = self.weight.view(self.out_channels, -1)
        output = weight.matmul(input_unfold)
        output = F.fold(output, output_size=(self.get_output_size(height), self.get_output_size(width)), kernel_size=1)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return output
    
