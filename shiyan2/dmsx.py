import torch
data = torch.tensor([1.,1.,0.,1.,0.])
p_mle = torch.mean(data)
alpha = 2
beta = 2
p_map = (torch.sum(data)+alpha-1)/(len(data)+alpha+beta-2)
print("MLE =",p_mle.item())
print("MAP =",p_map.item())
