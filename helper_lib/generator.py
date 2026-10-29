import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_samples(model, device="cpu", num_samples=10):
    """Sample random latent vectors, generate images, and plot them on a grid."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.z_dim, device=device)   # points in latent space
        imgs = model.generator(z).cpu() * 0.5 + 0.5                # rescale [-1, 1] -> [0, 1]

    ncols = min(num_samples, 5)
    nrows = math.ceil(num_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < num_samples:
            ax.imshow(imgs[i, 0], cmap="gray")

    plt.tight_layout()
    plt.show()
