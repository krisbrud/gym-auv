# %%
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageSequence

# import sys
# sys.path.append("/Library/TeX/texbin/")

filename = "reconstruction-jan16.gif"

# plt.style.use("ggplot")

# Activate latex in matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ncols = 5
n_every = 1 # 10

from dataclasses import dataclass

@dataclass
class ImageReconstruction:
    observation: Image
    reconstruction: Image
    

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
    print("len frames", len(frames))
    print(frames[0])


    # Get the width and height of the image
    im = frames[0]
    width, height = im.size

    # Calculate the width and height of the crop box
    crop_width = 64 
    crop_height = 64 

    # Initialize an empty list to hold the crop images
    cropped_images = [[] for _ in range(width)]
    reconstructions = [[] for _ in range(width)]

    # Iterate over the x and y coordinates
    for i, x in enumerate(range(0, width, crop_width)):
        print("i", i)
        obs = im.crop((x, 0, x + crop_width, crop_height))
        reconstr = im.crop((x, crop_height, x + crop_width, 2 * crop_height))
        reconstructions[i].append(ImageReconstruction(obs, reconstr))

        # for y in range(0, height, crop_height):
        #     # Use the crop method to create a new image
        #     cropped_im = im.crop((x, y, x + crop_width, y + crop_height))
        #     # Add the new image to the list
        #     cropped_images.append(cropped_im)

    

    for i in range(nframes):
        # if i % n_every == 0:
            # frame = frames[i]
            print("i", i)
            frame = reconstructions[0][i].observation
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

    # if last_col != ncols - 1:
    #     for i in range(last_col + 1, ncols):
    #         axes[last_row][i].remove()

    plt.subplots_adjust(wspace=0.1, hspace=0)
#     # plt.savefig("policy_image_jan16-4.pdf", pad_inches=0, bbox_inches="tight") # , bbox_inches='tight', pad_inches=0)
    plt.show()
# # %%

# %%
