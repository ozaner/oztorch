![OzTorch Logo](https://github.com/ozaner/oztorch/blob/main/oztorch-logo-dark.png)

--------------------------------------------------------------------------------

OzTorch aims to be a **reimplementation** of a **subset** of [PyTorch](https://pytorch.org/)'s Python API.

- [Why?](#why)
- [Subset of PyTorch?](#subset-of-pytorch)
- [Reimplementation of PyTorch?](#reimplementation-of-pytorch)
- [Phases](#phases)
  - [Phase 1 - Working Example](#phase-1---working-example)
  - [Phase 2 - Rust Backend](#phase-2---rust-backend)
  - [Phase 3 - CUDA support](#phase-3---cuda-support)

## Why?
Why reimplement the PyTorch API? If you've ever looked at the [PyTorch repo](https://github.com/pytorch/pytorch) its extremely complex, chock full of seemingly unrelated configuration files C++ bindings, legacy code, and other files that I, at least, simply can't make out.

OzTorch is a reimplementation of a subset of PyTorch (we will make that more concrete below) that serves to prove that it is not that hard to make a machine learning library in principle and, in particular, we will show that it is not that hard to make (the main parts of) PyTorch.

Of course, the resulting library won't be as fully-featured nor as performant as the original, but this is more of a learning exercise. We also address how better performance can be achieved in later phases of the project.

## Subset of PyTorch?
By 'subset of PyTorch' we mean only some functions and classes will be supported. Enough to at least create, train, and run linear models, logistic models, neural networks, and convolutional networks.

## Reimplementation of PyTorch?
By 'reimplementation of PyTorch' we mean that if we were to write some code using oztorch:
```python
import oztorch as torch

class TestModel(torch.nn.Module):
  def __init__(self):
    super(TestModel, self).__init__()

    self.linear1 = torch.nn.Linear(100, 200)
    self.activation = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(200, 10)
    self.softmax = torch.nn.Softmax()

  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.softmax(x)
    return x

testmodel = TestModel()
```

That same code should work if we replace the first line with:
```python
import torch
```

Note that because this is a `subset` of PyTorch, the reverse doesn't hold true in general. That is to day, some code written with PyTorch may not work with OzTorch (as it doesn't implement everything)

## Phases
OzTorch will be developed in 3 phases, each with increasing performance:

### Phase 1 - Working Example
A reimplementation of PyTorch, enough to make the models mentioned above, written purely in Python. This phase should be a complete package, and will most likely be much slower than PyTorch proper once benchmarks are run.

### Phase 2 - Rust Backend
In this phase, we replace the computationally expensive operations in the backend with Rust bindings, mirroring how PyTorch does the same with C++ bindings. We will then benchmark the performance and compare it to phase 1 as well as PyTorch proper.

### Phase 3 - CUDA support
In this phase we will, hopefully, further improve performance by implementing support for storing tensors and running operations on CUDA cores (i.e. Nvidia GPUs).