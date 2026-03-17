import torch
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.linspace(-5,5,100)
y_vals = x_vals**2+2*x_vals+1

x_path = []
x = torch.tensor([5.0],requires_grad=True)

for i in range(10):
    y = x**2+2*x+1
    x_path.append(x.item())

    y.backward()
    with torch.no_grad():
        x -= 0.3*x.grad
    x.grad.zero_()

plt.plot(x_vals,y_vals)
plt.scatter(x_path,[xx**2+2*xx+1 for xx in x_path],color='r')
plt.title("Gradient Descent Path")
plt.grid()
plt.show()
