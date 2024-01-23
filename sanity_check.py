import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.datasets import face
from src import CustomConv2D
torch.manual_seed(0)


def main():
    img = face() # koala face in RGB
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() # add batch dimension
    kernel_size = (2, 2)
    conv = CustomConv2D(in_channels=3, out_channels=3, kernel_size=kernel_size,padding=2)
    weights = conv.weight # save weights for PyTorch's Conv2D
    output = conv(img)

    # Compare with PyTorch's Conv2D
    conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, padding=2)
    conv2.weight = nn.Parameter(weights.view(conv2.weight.shape))
    output2 = conv2(img)
    print(f'Output shape (PyTorch): {output2.shape}')

    # Plot convolved image per channel for both implementations
    fig, axes = plt.subplots(2, 3)
    for ax, channel in zip(axes[0], ['Red', 'Green', 'Blue']):
        ax.set_title(channel)    
    for ax, channel in zip(axes[0], range(3)):
        ax.imshow(output[0, channel].detach().numpy())
    for ax, channel in zip(axes[1], range(3)):
        ax.imshow(output2[0, channel].detach().numpy())
    plt.show()

if __name__ == '__main__':
    main()