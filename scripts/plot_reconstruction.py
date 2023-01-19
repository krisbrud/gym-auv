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

        reconstructions = [[] for _ in range(width // crop_width)]

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
   
def plot_observations_and_reconstructions(obs_and_rec):
    """Takes a list of ObservationAndReconstruction objects and plots them in a grid."""
    nframes = len(obs_and_rec[0])
    for i in range(nframes):
        # if i % n_every == 0:
        # frame = frames[i]
        print("i", i)
        frame = obs_and_rec[0][i].observation
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
obs_and_rec = get_observations_and_reconstructions(filename)
# %%
# imagined_indices_to_plot = [5] + [i for i in range(9, 50, 5)]
# imagined_indices_to_plot
# TODO: Plot a single row
# %%

some_frame = obs_and_rec[0][0].observation

import matplotlib.gridspec as gridspec

def plot_observations_and_reconstructions(obs_and_rec, filename = None):
    observations = [o.observation for o in obs_and_rec]
    reconstructions = [o.reconstruction for o in obs_and_rec]

    fig = plt.figure(figsize=(10, 4))
    outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0, width_ratios=[5, 10])

    for i in range(2):
        # i = 0: left part, i = 1: right part

        cols = 5 if i % 2 == 0 else 10
        inner = gridspec.GridSpecFromSubplotSpec(2, cols,
                        subplot_spec=outer[i], wspace=0.0, hspace=-0.81 )

        for j in range(2 * cols):
            # Iterate over cols in parts, spanning 2 rows
            ax = plt.Subplot(fig, inner[j])
            if i == 0:
                # Left part
                if j < 5:
                    # Top part
                    list_to_take_from = observations
                    idx = j
                else:
                    # Bottom part
                    list_to_take_from = reconstructions
                    idx = j - 5
                
            else:
                # Right part (imagined)
                list_to_take_from = reconstructions
                if j % 10 == 0:
                    idx = 5
                    list_to_take_from = observations if j == 0 else reconstructions
                elif j < 10:
                    # Top part
                    list_to_take_from = observations
                    idx = (j % 10) * 5 + 4
                else:
                    # Bottom part
                    list_to_take_from = reconstructions
                    idx = (j % 10) * 5 + 4

                # if j < 10:
                #     # Top part
                #     list_to_take_from = observations
                #     # idx = 4 + 5 * j
                # else:
                #     # Bottom part
                #     list_to_take_from = reconstructions
                #     idx = (j - 10) * i + 5
                
            frame = list_to_take_from[idx]

            ax.imshow(frame)
            # t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i, j))
            t = ax.text(32, 90, r"$$t = " + str(idx + 1) + r"$$")

            x_offset_a = -100
            y_offset_a = 40
            if i == 0 and j == 0:
                obs_text = ax.text(x_offset_a, y_offset_a, "Observation")
                obs_text.set_ha('left')
            
            if i == 0 and j == 5:
                pred_text = ax.text(x_offset_a, y_offset_a, "Predictions")
                pred_text.set_ha('left')

            x_offset_b = 35 
            y_offset_b = -15
            big_fontsize = 20
            if i == 0 and j == 2:
                ctx_text = ax.text(x_offset_b, y_offset_b, "Context", fontsize=big_fontsize)
                ctx_text.set_ha('center')
            
            x_offset_c = 64 
            if i == 1 and j == 4:
                pred_text = ax.text(x_offset_c, y_offset_b, "Open loop prediction", fontsize=big_fontsize)
                pred_text.set_ha('center')


            t.set_ha('center')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    fig.show()


def plot_and_save_all(obs_and_rec_list):
    for i, obrec in enumerate(obs_and_rec_list):
        plot_observations_and_reconstructions(obrec, f"reconstruction-testfig-{i}.pdf")

# plot_observations_and_reconstructions(obs_and_rec[0], "testfig.pdf")
plot_and_save_all(obs_and_rec)
# %%
