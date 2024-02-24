---
layout: post
title:  "Making your GPU go BRRR - Programing a PyTorch Layer in CUDA"
date:   2024-02-23 12:00:00 -0300
categories: jekyll update
---
I still remember the "dark ages" of research, when I was still doing my masters, that it was common to find really impactfull publications that provided no code. And yes, I've send my fair share of emails to authors... Fortunatelly, this is no longer the norm, and even somewhat frowned upon. Caffe, Tensorflow, Keras, PyTorch, and even more deeplearning frameworks really helped everyone to create way smaller, cleaner code, that were also easier to share.

Those frameworks are really incredible, and allow us to quickly implement and test new ideas, however they are not always the fastest way, even if they using CUDA down the line. A good example is the amazing work of Flash Attention, that implemented the Attention machanism (yeah, the one used on all LLMs right now) in CUDA, and achieved a great runtime improvement.

CUDA programming may seem intimidating, at least it was for me. I first learned circa 2010 and it was a really bad development experience, but by watching an [awesome video](https://www.youtube.com/watch?v=nOxKexn3iBo) by Jeremy Howard, I've learned that it is indeed possible to have a much better experience. And the main idea is the following:

1. Implement the forward and backward pass in PyTorch. This gives access to an online debugger and the full functionality of Python, like Jupyter Notebooks.
2. Validate the implementation with gradcheck. This somewhat magic function, runs your forward pass and do numerical derivation to validate your backward pass code.
3. Program the CUDA Kernel for forward and backward pass using Numba, directly in Python. This is the real thing, where we are dealing CUDA threads and possibly memory management.
4. Ask [Chat-GPT](https://chat.openai.com/) to convert this code to C CUDA. Really, it works surprisingly well!
5. Use PyTorch internal functionality to compile this C CUDA to a Python module that you can use with torch tensors.
6. Use gradcheck again to verify that your CUDA written layer is 100% correct.

It may be a couple of hoops, but the ability to develop CUDA code in Python makes our lives so much easier. You have easier integration with debugers, and the iteration time between changes in code and running it is nearly instant, compared to the long time it takes to compile C CUDA. And you may noticed that mentioned both forward and backward pass, and unfortunatly, if we use CUDA for our backward pass, we can't rely on autograd to get this for us. But fortunatly we have this amazing funtion from PyTorch, [gradcheck](https://pytorch.org/docs/stable/notes/gradcheck.html), that will validate for us if our backpropagation is indeed correct. 

So let's stop talking and lets start coding. For our example I've chosen the Sigmoid activation that have some interesting characteristics.

## Implement in PyTorch

The idea is 

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

```python
code
```
