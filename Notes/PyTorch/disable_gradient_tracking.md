### Disabling gradient tracking

By default, all tensors with `requires_grad=True` are tracking their computational history and support gradient computation. However, there are some cases when we do not need to do that, for example, **when we have trained the model and just want to apply it to some input data, i.e. we only want to do forward computations through the network without updates**. We can stop tracking computations by surrounding our computation code with `torch.no_grad()` block:

```python

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w)+b
print(z.requires_grad)
print(z)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
print(z)

>>> True
>>> tensor([-2.7285,  2.3142, -4.3656], grad_fn=<AddBackward0>)
>>> False
>>> tensor([-2.7285,  2.3142, -4.3656])

```

Another way to achieve the same result is to use the `detach()` method on the tensor:

```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

>>> False
```


There are reasons you might want to disable gradient tracking:
* To mark some parameters in your neural network as frozen parameters. This is a very common scenario for finetuning a pretrained network
* To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.