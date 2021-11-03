from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from os.path import join
from sys import argv
from newgan.models import Config, gradient_penalty
from newgan.plotting import plot_train_hists

plt.switch_backend("Qt5Agg")


generator = torch.nn.Sequential(
    torch.nn.Linear(1, 128),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(128, 128),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(128, 1),
)
discriminator = torch.nn.Sequential(
    (torch.nn.Linear(1, 128)),
    torch.nn.LeakyReLU(0.2),
    (torch.nn.Linear(128, 128)),
    torch.nn.LeakyReLU(0.2),
    (torch.nn.Linear(128, 128)),
    torch.nn.LeakyReLU(0.2),
    (torch.nn.Linear(128, 1)),
)


config = Config(
    lr=0.0001,
    d_steps=5,
    g_steps=1,
    batch_size=64,
    epochs=1000,
    lambda_gp=10,
    show="--no-plot" not in argv,
)
optim_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.9))
optim_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.9))
scheduler_g = torch.optim.lr_scheduler.StepLR(
    optim_g, step_size=config.epochs // 5, gamma=0.95
)
scheduler_d = torch.optim.lr_scheduler.StepLR(
    optim_d, step_size=config.epochs // 5, gamma=0.9
)


pbar = trange(config.epochs)

# Generate some data
x = torch.randn(10000, 1).view(-1, 1) - 2
dataloader = DataLoader(
    x, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True
)
z = torch.randn(10000, 1)
with torch.no_grad():
    y = generator(z).numpy().flatten()


def train_step():
    for _ in range(config.d_steps):
        for real_data in dataloader:
            optim_d.zero_grad()
            pred_real = discriminator(real_data)
            pred_fake = discriminator(generator(torch.randn(config.batch_size, 1)))
            loss_d = -(pred_real.mean() - pred_fake.mean())
            loss_gp = gradient_penalty(discriminator, real_data, generator(real_data))
            loss = loss_d + config.lambda_gp * loss_gp
            loss.backward()
            w_distance = -loss_d
            optim_d.step()

    lr_d = optim_d.param_groups[0]["lr"]
    lr_g = optim_g.param_groups[0]["lr"]
    pbar.set_description_str(
        f"w_distance: {w_distance:.2e} | lrd:{lr_d:.1e} , lrg:{lr_g:.1e}"
    )
    # train generator
    optim_g.zero_grad()
    pred_fake = discriminator(generator(torch.randn(config.batch_size, 1)))
    loss_g = -pred_fake.mean()
    loss_g.backward()
    optim_g.step()
    scheduler_g.step()
    scheduler_d.step()
    with torch.no_grad():
        fake_data = generator(z).numpy().flatten()
    metrics = dict(w_distance=w_distance.item(), lr_d=lr_d, lr_g=lr_g)
    return metrics, fake_data


if __name__ == "__main__":
    if config.show:
        ani = plot_train_hists(x, y, config, train_step)
        plt.tight_layout()
        plt.show(block=False)
        filename = join(config.project_root, "plots/1d_wgan_gp10.mp4")
        ani.save(
            filename, fps=30, progress_callback=lambda i, n: pbar.update(1),
        )
    else:
        for _ in pbar:
            train_step()
