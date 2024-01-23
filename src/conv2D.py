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
        kernel_size: Height or width of kernel tensor.
        padding: Padding size.
        stride: Stride size.
        """
        return (input_size + 2 * self.padding - self.dilation*(self.kernel_size-1)-1) // self.stride + 1

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 4:
            input = input.unsqueeze(0).unsqueeze(0)
            print(f'Input tensor has {input.dim()} dimensions. Adding 2 dimensions to match 4D tensor.')
        
        input_unfold = F.unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        weight = self.weight.view(self.out_channels, -1)
        output = weight.matmul(input_unfold)
        output = F.fold(output, output_size=(self.get_output_size(input.size(2)), self.get_output_size(input.size(3))), kernel_size=1)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return output
    

if __name__ == '__main__':
    from scipy.misc import face
    img = face(gray=True)
    img = torch.from_numpy(img).float()
    kernel_size = (2, 2)
    conv = CustomConv2D(in_channels=1, out_channels=2, kernel_size=kernel_size,padding=2)
    weights = conv.weight
    img = img.unsqueeze(0).unsqueeze(0)
    output_img1 = conv(img)
    conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=kernel_size,padding=2)
    output_img2 = conv2(img)

    import matplotlib.pyplot as plt
    # plot both images next to each other
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(output_img1[0, 0, :, :].detach().numpy(), cmap='gray')
    ax2.imshow(output_img2[0, 0, :, :].detach().numpy(), cmap='gray')
    plt.show()