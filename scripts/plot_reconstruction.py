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
class ObservationAndReconstruction:
    observation: Image
    reconstruction: Image
    
def get_observations_and_reconstructions(filename: str):
    with Image.open(filename) as im:
        nframes = len(list(ImageSequence.Iterator(im)))
        nrows = math.ceil(nframes / (ncols * n_every))
        # fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))

        frames = ImageSequence.all_frames(im)  

        # Get the width and height of the image
        width, height = im.size

        # Calculate the width and height of the crop box
        crop_width = 64 
        crop_height = 64 

        reconstructions = [[] for _ in range(width)]

        # Iterate over the x and y coordinates
        for frame in frames:
            for i, x in enumerate(range(0, width, crop_width)):
                print("i", i)
                # obs = im.crop((x, 0, x + crop_width, crop_height))
                # reconstr = im.crop((x, crop_height, x + crop_width, 2 * crop_height))
                obs = frame.crop((x, 0, x + crop_width, crop_height))
                reconstr = frame.crop((x, crop_height, x + crop_width, 2 * crop_height))

                reconstructions[i].append(ObservationAndReconstruction(obs, reconstr))

    return reconstructions
   
def plot_observations_and_reconstructions(reconstructions):
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

#     # Remove the empty subplots
#     last_row = nrows - 1
#     last_col = (nframes // n_every) % ncols

#     # if last_col != ncols - 1:
#     #     for i in range(last_col + 1, ncols):
#     #         axes[last_row][i].remove()

    plt.subplots_adjust(wspace=0.1, hspace=0)
# #     # plt.savefig("policy_image_jan16-4.pdf", pad_inches=0, bbox_inches="tight") # , bbox_inches='tight', pad_inches=0)
    plt.show()
# # # %%

# %%
