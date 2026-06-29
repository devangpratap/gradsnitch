"""Live demo: a real torch run rigged to diverge (LR too high) + gradsnitch watching."""

import torch
import torch.nn as nn
import gradsnitch

torch.manual_seed(0)
X = torch.randn(512, 20)
w = torch.randn(20, 1)
y = X @ w + 0.1 * torch.randn(512, 1)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.55)  # deliberately too hot
lossfn = nn.MSELoss()

mon = gradsnitch.watch(model, opt, check_every=50)  # print errors live
for step in range(400):
    opt.zero_grad()
    loss = lossfn(model(X), y)
    loss.backward()
    mon.log(step, loss.item())  # one line: grabs grad_norm + lr for you
    opt.step()

print("\n=== gradsnitch report ===")
mon.report()
