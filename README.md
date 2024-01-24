# conv_pytorch
- What I cannot create, I do not understand. - Richard Feynman

This repository contains my attempt at implementing a convolution operation in PyTorch.


Despite seeming straightforward in theory, its practical implementation presented its own set of challenges. To provide a transparent view of this learning process, I've included my initial, less polished attempts at implementing convolution, leading up to the more refined final version
## Convolution operation
At its core, the convolution operation is a series of element-wise matrix multiplications. We refer to this matrix as a "kernel". This kernel slides over the input image in strides, multiplying with each encountered matrix. To visualize, consider a convolution operation with a 3x3 kernel and a stride of 1 applied to a 6x6 input. Remember, images typically have 3 channels (RGB), so the actual input would be 6x6x3.
![convolution](/pics/conv_operation.gif) (Credits: [Stanford CS230](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#filter))

## Implementation
### How to get patches?
The kernel multiplies with patches of the input matrix, essentially sub-matrices that you observe in the image above. Obtaining these patches initially seemed straightforward: slide over the matrix with nested for-loops to gather all necessary patches. However, this method quickly revealed its inefficiency with a time complexity of O(n^2):
```python
def unfold(input: torch.Tensor,kernel_size, stride) -> torch.Tensor:
        """ Given a 2D tensor, unfold it to patches of size kernel_size."""
        dx, dy = kernel_size
        x = input.shape[0]
        y = input.shape[1]
        patches = []
        for i in range(0, x-dx+1,stride):
            for j in range(0, y-dy+1,stride):
                patches.append(input[i:i+dx,j:j+dy])
        return torch.stack(patches).view(dx*dy,-1).type(torch.float32)
```
With this approach, we end up with (x-dx+1) * (y-dy+1) patches, where dx and dy represent the kernel sizes in the x and y dimensions, respectively.
Expanding this to accommodate 4D tensors (batch_size, channels, x, y) necessitates an additional layer of for-loops, ballooning the time complexity to O(n^4) – a far cry from ideal:
```python
def batch_unfold(input: torch.Tensor,kernel_size, stride) -> torch.Tensor:
        """ Given a 4D tensor, unfold it to patches of size kernel_size."""
        dx, dy = kernel_size
        x = input.shape[2]
        y = input.shape[3]
        patches = []
        for i in range(0, x-dx+1,stride):
            for j in range(0, y-dy+1,stride):
                patches.append(input[:,:,i:i+dx,j:j+dy])
        return torch.stack(patches).view(input.shape[0],dx*dy,-1).type(torch.float32)
```
This embarrassingly slow approach led me to use torch.stack() to aggregate patches into a single tensor, which was kinda sluggish. A more efficient tensor operation solution undoubtedly exists, but for the moment, I've left it as is.

### Leveraging Built-in Functions
"Standing on the shoulders of giants," I turned to torch.nn.functional.unfold(), a function that exactly addresses my needs, and being implemented in C++, offers a significant speed advantage over my naive implementation. Here's the refined version:
```python
def unfold(input: torch.Tensor,kernel_size, stride) -> torch.Tensor:
        """ Given a 4D tensor, unfold it to patches of size kernel_size."""
        dx, dy = kernel_size
        return F.unfold(input, kernel_size, stride=stride)
```
## Unfold vs Fold
To understand how unfold and fold work, take a look at this figure:
![unfold_fold](/pics/unfold_fold.png) (Credits: [Stackoverflow](https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work))


## How to get feature map size?
The output size for feature maps is determined by the kernel size, stride, dilation, and padding. Here's the formula:
```python
def get_output_size(input_size, kernel_size, stride, padding, dilation):
    return (input_size + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
```
## How to get weight matrix shape?
Weight matrix is applied to each channel of the input matrix. So, the weight matrix shape is (output_channels, input_channels* kernel_size* kernel_size).
```python
def get_weight_shape(input_channels, output_channels, kernel_size):
    return (output_channels, input_channels*kernel_size*kernel_size)
```
## Quick sanity check
To make sure that our implementation is correct, we can compare it with PyTorch's implementation. Here's a quick check:
```bash
python sanity_check.py
```
This will give us the following per channel convolution output(first row is our implementation, second row is PyTorch's implementation):
![sanity_check](/pics/sanity_check.png)

## Tests
To run tests, simply run:
```bash
pytest tests/
```

## What's next?
- Implement transposed convolution
- Implement Fast Fourier convolution

## Recap
- PyTorch's built-in functions are fast
- Unfold and fold are inverses of each other
- Convolution is just a series of element-wise matrix multiplications
