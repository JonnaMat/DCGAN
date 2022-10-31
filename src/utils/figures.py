#pylint: disable=E0401:import-error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
from IPython.display import HTML

SQUARE_FIGSIZE = (10, 10)

def imshow_grid(batch, device, title: str):
    """Plot images in batch in an even grid."""
    plt.figure(figsize=SQUARE_FIGSIZE)
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(
            vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()

def losses(generator_losses, discriminator_losses):
    """Plot loss of generator and discriminator in one figure."""
    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(20,8))
    plt.title("Generator and Discriminator Training Loss", fontdict={"size":20})
    plt.plot(generator_losses,label="Generator")
    plt.plot(discriminator_losses,label="Discriminator")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel("Iteration", fontdict={"size":16})
    plt.ylabel("Loss",  fontdict={"size":16})
    plt.legend(fontsize=12)
    plt.show()

def real_vs_fakes(batch, device, img_list):
    """Plot real and fake images grid."""

    # Plot the real images
    plt.figure(figsize=SQUARE_FIGSIZE)
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images", fontdict={"size":20})
    plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images", fontdict={"size":20})
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

def progress(img_list):
    """Show animation of generator progress."""
    fig = plt.figure(figsize=SQUARE_FIGSIZE)
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())