# conv_pytorch
We think we understand how something works, but it is until we do implement it, we don't fully understand. Indeed, convolution was not that easy. I have to write couple of times to get it right. For full honesty, I am including my first "bad" attempts below, so you can see how I came to the final version.
## Convolution operation
Convolution operation is essentially element-wise matrix multiplication. We call this matrix a "kernel". This kernel slides over the input image with "stride"s and performs multiplication with each matrix.  
Here's a visualized example of convolution operation with 3x3 kernel and stride 1 applied to 6x6 input. Remember, images have 3 channels (RGB), so the input woudl actually be 6x6x3.
![convolution](/pics/conv_operation.gif) (Credits: [Stanford CS230](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#filter))

## Implementation
### How to get patches?
The first thing we need to do is to get matrix patches (the same size as the kernel) from input matrix. Here's my naive implementation that runs in O(n^2) time due to nested for loops:
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
So, in total, we should have (x-dx+1) * (y-dy+1) patches, where dx and dy are kernel sizes in x and y directions respectively. 
As you can notice, this does not consider multiple channels, let alone batch size. So, I had to modify it to work with multiple channels and batches. Here's the final version:
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
This was embarrassingly (but not surprisingly) slower than unbatched 
version. I had to use torch.stack() to stack all the patches into a single tensor, which was very slow. I am sure there is a way to do this without using torch.stack() and being smart about tensor operations, but for now, I am going to leave it as it is.

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
