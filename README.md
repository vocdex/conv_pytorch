# conv_pytorch
We think we understand how something works, but it is until we do implement it, we don't fully understand. Indeed, convolution was not that easy. I have to write couple of times to get it right. For full honesty, I am including my first "bad" attempts below, so you can see how I came to the final version.
## Convolution operation
Convolution operation is essentially element-wise matrix multiplication. We call this matrix a "kernel". This kernel slides over the input image with "stride"s and performs multiplication with each matrix.  
Here's a visualized example of convolution operation with 3x3 kernel and stride 1 applied to 6x6 input. Remember, images have 3 channels (RGB), so the input woudl actually be 6x6x3.
![convolution](/pics/conv_operation.gif) (Credits: [Stanford CS230](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#filter))

## Implementation
### How to get patches?
We need to multiply our kernel with patches of the input matrix. Patches are essentially sub-matrices of the input matrix that you see in the image above.
The first thing we need to do is to get matrix patches (the same size as the kernel) from input matrix. Just sliding over the entire matrix with nester for-loops would give us all the patches we need. This would require O(n^2) time complexity, which is not ideal:
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
So, in total, we should have (x-dx+1) * (y-dy+1) patches, where dx and dy are kernel sizes in x and y directions, respectively. 
As you can notice, this does not consider multiple channels, let alone batch size. So, we can extend this to 4D tensors (batch_size, channels, x, y) by adding another for-loop for batch size and channels. This would give us O(n^4) time complexity, which is not ideal at all. 
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
This is embarrassingly  slower than the unbatched version. I had to use torch.stack() to stack all the patches into a single tensor, which was very slow. I am sure there is a way to do this without using torch.stack() and being smart about tensor operations, but for now, I am going to leave it as it is.

Instead, "standing on the shoulders of giants", I used torch.nn.functional.unfold() function, which does exactly what I want. It is also implemented in C++, so it is much faster than my naive implementation. Here's the final version:
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
## Quick check
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