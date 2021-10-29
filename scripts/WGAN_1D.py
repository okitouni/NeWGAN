from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from os.path import join
from newgan.models import Config, gradient_penalty

plt.switch_backend("Qt5Agg")

generator = torch.nn.Sequential(
    torch.nn.Linear(1, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)
discriminator = torch.nn.Sequential(
    (torch.nn.Linear(1, 128)),
    torch.nn.ReLU(),
    (torch.nn.Linear(128, 128)),
    torch.nn.ReLU(),
    (torch.nn.Linear(128, 1)),
)


config = Config(lr=0.005, d_steps=1, g_steps=1, batch_size=64, epochs=1000, lambda_gp=10)
optim_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.9))
optim_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.9))
scheduler_g = torch.optim.lr_scheduler.StepLR(
    optim_g, step_size=config.epochs // 5, gamma=0.95
)
scheduler_d = torch.optim.lr_scheduler.StepLR(
    optim_d, step_size=config.epochs // 5, gamma=0.9
)


pbar = trange(config.epochs)
real_labels = torch.ones(config.batch_size, 1)
fake_labels = torch.zeros(config.batch_size, 1)

# Generate some data
x = torch.randn(10000, 1).view(-1, 1) - 2
dataloader = DataLoader(
    x, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True
)
z = torch.randn(10000, 1)
with torch.no_grad():
    y = generator(z)

fig, ax = plt.subplots()
n, _, bar_container = plt.hist(
    x.numpy(), bins=30, density=False, label="real", alpha=0.5
)
_, _, bar_container = plt.hist(
    y.numpy(), bins=30, density=False, label="fake", alpha=0.5,
)
text = ax.text(0.01, 0.9, "", transform=ax.transAxes, ha="left")

ax.set_ylim(0, max(n) * 1.1)
plt.legend()


def update(frame):
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
        f"Epoch {frame} | w_distance: {w_distance:.2e} | lrd:{lr_d:.1e} , lrg:{lr_g:.1e}"
    )
    # train generator
    optim_g.zero_grad()
    pred_fake = discriminator(generator(torch.randn(config.batch_size, 1)))
    loss_g = -pred_fake.mean()
    loss_g.backward()
    optim_g.step()
    scheduler_g.step()
    scheduler_d.step()

    # Plotting
    if frame > 0:
        fig.canvas.start_event_loop(0.001)
    with torch.no_grad():
        data = generator(z).numpy().flatten()
    text.set_text(
        "{}/{} | W distance {:.2e}".format(frame, config.epochs, w_distance.item())
    )
    n, bins = np.histogram(data, bins=30, density=False)
    for count, rect, coord in zip(n, bar_container.patches, bins):
        rect.set_height(count)
        rect.set_x(coord)
        rect.set_width(bins[1] - bins[0])
    # ax.set_ylim(0, max(n) * 1.1)
    return bar_container.patches + [text]


ani = FuncAnimation(
    fig, update, frames=config.epochs, interval=500, blit=True, repeat=False
)
plt.tight_layout()
plt.show(block=False)

filename = join(config.project_root, "plots/1d_wgan.mp4")
ani.save(
    filename, fps=30, progress_callback=lambda i, n: pbar.update(1),
)
