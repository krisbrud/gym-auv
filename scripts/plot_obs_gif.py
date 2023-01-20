# %%
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageSequence

# import sys
# sys.path.append("/Library/TeX/texbin/")

filename = "policy_image_jan16-4.gif"

# plt.style.use("ggplot")

# Activate latex in matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ncols = 5
n_every = 50

# Open the GIF
with Image.open(filename) as im:
    # Get the total number of frames
    nframes = len(list(ImageSequence.Iterator(im)))
    # Calculate the number of rows
    nrows = math.ceil(nframes / (ncols * n_every))
    # print("nrows", nrows)
    # Create a figure with ncols columns and nrows rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    # Flatten the axes array to make it 1-dimensional
    # axes = axes.ravel()
    # Iterate over all frames
    frames = ImageSequence.all_frames(im)  


    for i in range(nframes):
        if i % n_every == 0:
            frame = frames[i]
            print(frame)
            i_row = i // (ncols * n_every)
            i_col = (i // n_every) % ncols
            # Plot each frame in the corresponding subplot
            ax = axes[i_row][i_col]
            ax.imshow(frame)
            ax.set_adjustable("box")
            ax.axis('off')
            title_offset = -0.1 
            ax.set_title(r"$$t = " + str(i) + r"\,[s]$$", y=title_offset)

    # Remove the empty subplots
    last_row = nrows - 1
    last_col = (nframes // n_every) % ncols

    if last_col != ncols - 1:
        for i in range(last_col + 1, ncols):
            axes[last_row][i].remove()

    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.savefig("policy_image_jan16-4.pdf", pad_inches=0, bbox_inches="tight") # , bbox_inches='tight', pad_inches=0)
    plt.show()
# %%
