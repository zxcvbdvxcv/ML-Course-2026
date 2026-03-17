import torch
import matplotlib.pyplot as plt

loss = []
x = torch.tensor([5.0],requires_grad=True)

for i in range(20):
    y = x**2+2*x+1
    loss.append(y.item())

    y.backward()
    with torch.no_grad():
        x -= 0.1*x.grad
    x.grad.zero_()

plt.plot(loss,marker='o')
plt.title("Loss Curve")
plt.grid()
plt.show()
