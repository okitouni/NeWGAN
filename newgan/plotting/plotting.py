from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_train_hists(x, y, config, train_step):
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
        metrics, fake_data = train_step()
        w_distance = metrics["w_distance"]
        # Plotting
        if frame > 0:
            plt.gcf().canvas.start_event_loop(0.001)
        text.set_text(
            "{}/{} | W distance {:.2e}".format(frame, config.epochs, w_distance)
        )
        n, bins = np.histogram(fake_data, bins=30, density=False)
        for count, rect, coord in zip(n, bar_container.patches, bins):
            rect.set_height(count)
            rect.set_x(coord)
            rect.set_width(bins[1] - bins[0])
        # ax.set_ylim(0, max(n) * 1.1)
        return bar_container.patches + [text]

    ani = FuncAnimation(
        fig, update, frames=config.epochs, interval=500, blit=True, repeat=False
    )
