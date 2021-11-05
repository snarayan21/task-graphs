import torch
from torch.autograd import grad

f01 = torch.tensor([1.0, 2.0], requires_grad=True)
f12 = torch.tensor([1.0, 2.0], requires_grad=True)

r1 = 3 * f01 ** 2
r1.retain_grad()
r2 = r1 + f12 ** 4
r2.retain_grad()
#r2_grads = grad(r2,[r1,r2], create_graph=True)
#print(r2_grads)
external_grad = torch.tensor([1.0, 1.0])
r2.backward(gradient=external_grad, create_graph=True)
print(f01.grad)
print(f12.grad)