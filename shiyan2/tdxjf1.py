import torch

x = torch.tensor([5.0],requires_grad=True)
lr = 0.1

for i in range(20):
    y = x**2+2*x+1
    y.backward()

    with torch.no_grad():
        x -= lr*x.grad

    x.grad.zero_()

print(x.item(),y.item())

