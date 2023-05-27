import oztorch as torch
# import torch


# a = torch.tensor([[[3.0,4],[5,7]],[[3,4],[5,7]]], requires_grad=True)
# print(a)
a = torch.tensor([5], requires_grad=True)
b = torch.Mul.apply(torch.tensor([2]),a) + torch.tensor([5])

print(b.size())
b.backward()
print(a.grad)

# a = torch.ones((5,), requires_grad=True)
# print("a", a.requires_grad)
# print("a", a.detach().requires_grad)

# a = torch.tensor([[3,4,5],[3,4,5],[3,4,5]])
# a = tensor([[[4,4],[62,5]],[[9,8],[432,34]]])
# a = torch.nn.Parameter(torch.ones((2,)))
# a.requires_grad_(False)

# d = torch.ones((2,))
# d.requires_grad = False
# print("HERE D", d.requires_grad)
# print("HERE D", d.is_leaf)
# # print("HERE D", d.grad_fn)
# print("HERE D2", d._storage)
# print("HERE D2", d)

# d.sum().backward()
# print(d.grad)


# b = torch.exp(2*d)
# b.sum().backward()

# print(a.grad)

# print(torch.tensor([[1,1],[1,1]], dtype=torch.float32))
# print(torch.empty((2,3), dtype=torch.float32))
# print(torch.Tensor(3,2,4))
# print(torch.Tensor(3,2,4).shape)

# def typingCheck(x: PyTensor) -> None:
#   print(x)

# typingCheck(2)




# a = tensor([[[9,0],[9,0]],[[9,0],[9,0]],[[9,0],[9,0]]])
# print(a)
# print(a.shape)
# print(a.stride())

# bases = [24, 60, 60]
# input = 623227
# output = []

# for base in reversed(bases):
#   output = [input % base] + output
#   input = input // base
# print(output)




# notes
# - bool's are ints
# - python int is always converted to int64 (unless otherwise specified)
# - python float is always converted to float32 (unless otherwise specified)
# - floats are rounded to 32bits (overflow = inf), ints overflow/underflow and crash if they don't fit in int64.

## todo:
#  - fix print of tensor (pretty print as well)