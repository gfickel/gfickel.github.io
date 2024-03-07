---
layout: post
title:  "Making your GPU go BRRR: Creating a CUDA Layer in PyTorch"
date:   2024-02-23 12:00:00 -0300
categories: jekyll update
---
I still remember the "dark ages" of research, when I was still doing my masters when it was common to find really impactful publications that provided no code. And yes, I've sent my fair share of emails to authors... Fortunately, this is no longer the norm, and even somewhat frowned upon. Caffe, Tensorflow, Keras, PyTorch, and even more deep learning frameworks really helped everyone to create way smaller, cleaner code, that was also easier to share.

Those frameworks are really incredible and allow us to quickly implement and test new ideas, however, they are not always the fastest way, even if they use CUDA down the line. This is definitely becoming a bottleneck. PyTorch 2 implemented a compile process to fuse layers to improve GPU usage, Flash Attention did the same by directly programming Attention in CUDA and achieved an even greater runtime improvement. And recently I've seen a trend to reimplement slower Python code in Rust [like this](https://github.com/huggingface/tokenizers) one from Hugging Face.

CUDA programming may seem intimidating, at least it was for me. I first learned circa 2010 and it was a really bad development experience, but by watching an [awesome video](https://www.youtube.com/watch?v=nOxKexn3iBo) by Jeremy Howard, I've learned that it is indeed possible to have a much better experience. The main idea is the following:

1. Implement the forward and backward pass in PyTorch. This gives access to an online debugger and the full functionality of Python, like Jupyter Notebooks.
2. Validate the implementation with gradcheck. This somewhat magic function runs your forward pass and does numerical derivation to validate your backward pass code.
3. Program the CUDA Kernel for forward and backward passes using Numba, directly in Python. This is the real thing, where we are dealing with CUDA threads and possibly memory management.
4. Ask [Chat-GPT](https://chat.openai.com/) to convert this code to C CUDA. Really, it works surprisingly well!
5. Use PyTorch internal functionality to compile this C CUDA to a Python module that you can use with torch tensors.
6. Use gradcheck again to verify that your CUDA written layer is 100% correct.

It may be a couple of hoops, but the ability to develop CUDA code in Python makes our lives so much easier. You have easier integration with debugers, and the iteration time between changes in code and running it is nearly instant, compared to the long time it takes to compile C CUDA. You may noticed that mentioned both forward and backward passes, and unfortunately, if we use CUDA for our backward pass, we can't rely on autograd to get this for us. But fortunately, we have this amazing function from PyTorch, [gradcheck](https://pytorch.org/docs/stable/notes/gradcheck.html), that will validate for us if our backpropagation is indeed correct. 

We need some kind of end goal, and for us, it will be the implementation of the Sigmoid activation inspired by [David Oniani](https://www.youtube.com/watch?v=oxC3T_-_Amw). You'll see that it has some interesting characteristics that will help us explore interesting (and important) aspects of creating a performant CUDA layer.

## 1. Forward and Backward passes in PyTorch

The idea here is to do two functions: one for the forward pass and the backward one. But first, let's remember the formula for the sigmoid and its derivative:

$$ \sigma(x) = \frac{1}{1+e^{-x}} $$

$$ \sigma^{'}(x) = \sigma(x)(1-\sigma(x)) $$

Those are not that complicated to implement, especially the derivative that only depends on the value of the sigmoid that we already computed on the forward pass. However, this sigmoid equation does present some numerical instabilities, so it is better to implement the following:

$$
\sigma(x)=\begin{cases}
\frac{1}{1+e^{-x}} & \text{ if } x>=0 \\ 
\frac{e^{x}}{1+e^{x}} & \text{ if } x<0 
\end{cases}
$$

With this in mind, we can generate the following Python code:

```python
def sigmoid_forward_torch(input):
    out_tensor = torch.empty_like(input)
    positive_mask = input >= 0
    out_tensor[positive_mask] = 1. / (1. + torch.exp(-input[positive_mask]))
    out_tensor[~positive_mask] = torch.exp(input[~positive_mask]) / (1. + torch.exp(input[~positive_mask]))
    
    return out_tensor

def sigmoid_backward_torch(input):
    return input * (1 - input)
```

Notice that I've used a variable called *positive_mask* to create an index to identify positive and negative input values. Other than that, the code is somewhat straightforward.


## 2. Check our Derivatives

Now that we have a Python code to do our forward and backward pass we can test if they are coherent with each other. In other words, we will use [gradcheck](https://pytorch.org/docs/stable/notes/gradcheck.html) from PyTorch to run a forward pass, compute numerically what the derivative should be, and check our backward pass result. But first, we must set it within its autograd format. It is not that complicated, and stays like this:

```python
class Sigmoid(torch.autograd.Function):
    """The Sigmoid activation function."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        out_tensor = torch.empty_like(input)
        positive_mask = input >= 0
        out_tensor[positive_mask] = 1. / (1. + torch.exp(-input[positive_mask]))
        out_tensor[~positive_mask] = torch.exp(input[~positive_mask]) / (1. + torch.exp(input[~positive_mask]))
        
        ctx.save_for_backward(out_tensor)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        (result,) = ctx.saved_tensors
        grad = result * (1 - result)
        return grad_output * grad
```

Notice that both on forward and backward pass we are dealing with an additional variable: ctx. This is our context, that we can use to save some data on our forward pass to use on backward. This is quite handy for our Sigmoid since the backward pass is a simple formula that uses the forward pass result, so we save it on the context for our backpropagation.

Finally, on the backward pass, we get the sigmoid result that we stored on ctx and use it to compute its derivative. But we have another input, that is the input derivative that is being propagated to our layer. So our final gradient is this derivative multiplied by our sigmoid derivative.

With this in hand, we can call the following function to check if everything is correct:

```python
sigmoid = Sigmoid.apply
data = torch.randn(4, dtype=torch.double, requires_grad=True)

if torch.autograd.gradcheck(sigmoid, data, eps=1e-8, atol=1e-7):
    print('gradcheck successful :D')
else:
    print('gradcheck unsuccessful :D')
```

If everything is correct we are ready to think about how to implement it in CUDA, otherwise, we can back up and check what we did wrong.


## 3. CUDA Implementation using Numba

(I will not dive into all the details on how CUDA works, but I suggest you check [this video](https://www.youtube.com/watch?v=nOxKexn3iBo) by Jeremy Howard to see a great explanation about it!)

The first thing we need to do is decide how we are going to model this in CUDA. I believe the most sensible approach is to use a single thread for each element on the input Tensor, for both forward and backward passes. And to finally implement it, we can use the [Numba library](https://numba.pydata.org/), which is a JIT compiler for Python with support for CUDA, SIMD, and even threading. But for our case, we are more interested in the CUDA dev environment, especially the [CUDA simulator](https://numba.pydata.org/numba-doc/latest/cuda/simulator.html#simulator).

To start, the first thing we must do is set NUMBA_ENABLE_CUDASIM='1' as an environment variable before we import Numba. Then we just need to add the @cuda.jit decorator on top of our CUDA kernel function and we are good to go!

Let's start with the following code for both the forward and backward passes:
```python
from numba import cuda
import torch

@cuda.jit
def sigmoid_forward(input, input_len, out):
    cbi,cbd,tid = cuda.blockIdx,cuda.blockDim,cuda.threadIdx
    idx = cbi.x * cbd.x + tid.x

    if idx >= input_len:
        return
    
    if input[idx] >= 0:
        res = 1. / ( 1. + math.exp(-input[idx]) )
    else:
        res = math.exp(input[idx]) / ( 1. + math.exp(input[idx]) )

    out[idx] = res

@cuda.jit
def sigmoid_backward(input, input_len, out):
    cbi,cbd,tid = cuda.blockIdx,cuda.blockDim,cuda.threadIdx
    idx = cbi.x * cbd.x + tid.x

    if idx >= input_len:
        return
    
    out[idx] = input[idx]*(1-input[idx])
```
There is a lot to unpack here, so let's start with the first lines. We are accessing ***cuda.blockIdx*** and ***cuda.threadIdx*** to get our block and thread indexes, and ***cuda.blockDim*** to know how many threads we have per block. And since we are using a single thread to compute a single value from our input tensor, we get our final index with

$$ idx = B_{index} * B_{size} + T_{index} $$,

where $$ B_{size} $$ is the number of threads per block, $$ B_{index} $$ and $$ T_{index} $$ are the block and thread indexes.

Having our current index, we must check if this index is within our input tensor size, and return without doing anything if it is not. Those cases will happen when the total number of threads, i.e. number of thread blocks times block size, is not exactly the same as the input size. 

If everything is correct, we will take the current value from input at location $$ idx $$ and calculate our Sigmoid. Nothing too fancy here. But we can test with the following code:

```python
def sigmoid_numba(input, fun, tw=16, gradcheck=False):
    (input_len,) = input.shape
    out = torch.zeros(input_len, dtype=torch.float32)
    out = out.contiguous().cuda()
    tpb = tw
    blocks = cdiv(input_len,tpb)
    fun[blocks, tpb](input, input_len, out) 
    return out
    
input = torch.as_tensor([0.3, -100000, 100000, 0.5, -0.5], dtype=torch.float32)
input = input.contiguous().cuda()

res = sigmoid_numba(input, sigmoid_forward, 1)
grad = sigmoid_numba(res, sigmoid_backward, 1)
```

I've created an auxiliary function called ***sigmoid_numba*** to encapsulate the important (and boring) code necessary to allocate our output tensor and calculate an appropriate number of threads per block and thread blocks. Those configurations have some upper limits depending on your CUDA GPU, and the optimal value for each also depends on the GPU version. But for now, we are just going with some numbers that somewhat seem right, and in the end, we can run a small benchmark to decide the best values for our particular GPU. And finally, notice that our input tensor is calling two functions: contiguous() and cuda(): [contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html#torch.Tensor.contiguous) makes sure that our tensor is contiguous in memory since we are accessing it like a single dimensional array; [cuda](https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda) returns a copy of our tensor in CUDA memory.

And that's it, with this code you are programming a CUDA kernel, but with the big difference that we can use a debugger and step to our code as we wish, and with a much smaller iteration time :). Notice that it is best to set $$ B_{size}=1 $$ when doing breakpoints since the debuggers usually don't work well with multiple threads calling a breakpoint at the same time.

This Numba CUDA development is way easier, and if we change our env variable to NUMBA_ENABLE_CUDASIM='0' we can run this code that Numba will compile it to CUDA for us, and we can see the performance that we should get. For some reason, the direct implementation in C CUDA is usually faster, with differences of 2x to be expected, but even then it should show us how fast our final implementation should be. Notice, however, that without CUDA Simulator enabled we will losse the ability to debug our code and use numpy/torch functions. You can check out [here](https://numba.pydata.org/numba-doc/latest/cuda/cudapysupported.html) what is supported. 


## 4. Calling out chat-GPT to Help Us

The Numba development is there to help us, but the final goal is to generate a C CUDA kernel that we can directly call on PyTorch. Fortunately, Chat-GPT is plenty capable of doing this! I've pasted the following query, followed by the Numba code: "Convert the following python code to C CUDA kernel. Also add a function that uses torch library to pass the input arguments, call the CUDA kernel, check for errors. The function must receive torch::Tensor as input and return the output as torch::Tensor."

And it gave me the something really close to this:

```cpp
#include <math.h>

__global__ void sigmoid_forward_cuda_kernel(const float* input, int input_len, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        float res;
        if (input[idx] >= 0) {
            res = 1. / (1. + expf(-input[idx]));
        } else {
            res = expf(input[idx]) / (1. + expf(input[idx]));
        }

        out[idx] = res;
    }
}

torch::Tensor sigmoid_forward_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    // Get the data pointers and sizes
    float* input_data_ptr = input.data_ptr<float>();
    int input_len = input.numel();

    // Allocate output tensor on GPU
    torch::Tensor out_tensor = torch::empty_like(input);

    // Get the data pointer for the output tensor
    float* out_data_ptr = out_tensor.data_ptr<float>();

    // Set block and grid dimensions
    int threads_per_block = 256; // You may adjust this based on your specific GPU capabilities
    int num_blocks = (input_len + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel
    sigmoid_forward_cuda_kernel<<<num_blocks, threads_per_block>>>(input_data_ptr, input_len, out_data_ptr);

    // Synchronize to ensure the kernel is done before proceeding
    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out_tensor;
}
```

Notice that in the query we've explicitly told chat-GPT to accept a torch::tensor as input and return another one as output. This makes our lives so much easier in the following steps.

The backward pass is quite similar, and you can check it on my [repo](https://github.com/gfickel/cuda-sigmoid).

## 5. Using PyTorch to Compile C CUDA

I really didn't know that PyTorch could do this, but if you have the dev files for CUDA and ninja build installed on your system you can pass the C CUDA code as a string and it will build it as a Python module for you. So first, to set things up we must have some auxiliary functions (thanks to Jeremy Howard), that you can check out [here](https://github.com/gfickel/cuda-sigmoid/blob/main/utils.py). The most important bit is a helper function to call [load_inline](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline) from PyTorch. It enables us to pass a C CUDA code as a string and compile it to a Python model containing the kernel as a Python function. It is quite amazing.

So, let's compile our C CUDA kernel! Here are the steps:

```python
cuda_src = FORWARD_PASS_CUDA_CODE_FROM_CHAT_GPT
fname = 'sigmoid_forward_cuda'
cpp_src = 'torch::Tensor sigmoid_forward_cuda(torch::Tensor input);'

module_forward = load_cuda(cuda_src, cpp_src, [fname])

input = torch.as_tensor([0.3, -100000, 100000, 0.5, -0.5], dtype=torch.float32)
input = input.contiguous().cuda()
res = module_forward.sigmoid_forward_cuda(input)
```
And that's it! But first, let's explain those lines a little bit. First, ***cuda_src*** is a Python string containing our code that was so gently translated to us by chat GPT. ***fname*** is the function name that we want to expose as a function in our compiled module, and ***cpp_src*** is the C++ code that is compiled with our CUDA kernel, and all it has is the declaration of our function. With all of this, we can finally call our helper ***load_cuda***, defined in our ***utils.py*** if you want to check it out, and it returns our new Python module with our ***sigmoid_forward_cuda*** function.

For the backward pass, it is mostly the same process, as expected. Here it is:

```python
cuda_src = BACKWARD_PASS_CUDA_CODE_FROM_CHAT_GPT
fname = 'sigmoid_backward_cuda'
cpp_src = 'torch::Tensor sigmoid_backward_cuda(torch::Tensor input);'

module_backward = load_cuda(cuda_src, cpp_src, [fname])

grad = module_backward.sigmoid_backward_cuda(res)
```

## 6. Check our Gradients Again

Great, we have both our forward and backward passes implemented in CUDA! However, are they correct? Did chat gpt make some silly mistake on the translation part? Well, at least we can check if the backward is indeed the correct derivation for the forward pass. Just like we did in step 2, we must call checkgradients. And to do this, first, we must adhere to the autograd interface, like this:

```python
class CUDASigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        result = module_forward.sigmoid_forward_cuda(data)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (result,) = ctx.saved_tensors
        grad = module_forward.sigmoid_backward_cuda(result)
        return grad_output * grad
```

Not that bad, if you ask me, and not that different from the one from step 2. And now, for the finale:

```python
sigmoid = CUDASigmoid.apply
data = torch.randn(4, dtype=torch.float, requires_grad=True)

# Changing eps and atol since we are dealing with float32
if torch.autograd.gradcheck(sigmoid, data, eps=1e-4, atol=1e-5):
    print('gradcheck successful :D')
else:
    print('gradcheck unsuccessful :D')
```
Wait, something is different. Our ***eps*** and ***atol*** are smaller, and our test data is float32 instead of float64. This final difference is indeed the key: gradcheck is made to work with float64, otherwise we will have some larger errors from our floating point operations. And since we've only implemented in float32 we must lower our error thresholds. It was indeed quite possible to use C++ templates and generate a double (i.e. float64) kernel also, but it was going to introduce some unnecessary complications for now.

With those caveats aside, our gradcheck should be passing and we are officially golden, our CUDA Sigmoid implementation is over!

## Conclusions

Uou, that was a long post. However, I tried to skim only the not-critical details and explain in greater detail the development pipeline. That is the key point that you should be taking from here: how to make CUDA development less sucky. And by using this PyTorch feature to compile CUDA code, we can even run CUDA kernels on Google Collabs!

In the end, I do believe that this is an interesting knowledge to have, and in this day and age of huge LLMs, being able to tackle some performance bottlenecks can have a great impact.