# %% 
import numpy as np
import matplotlib.pyplot as plt

# def sensor_end(range, angle):
#     return np.array([range * np.cos(angle), range * np.sin(angle)])

def make_sensor_ends(sensor_range, angle):
    return np.array([sensor_range * np.cos(angle), sensor_range * np.sin(angle)]).T

sensor_range = 1
n_sensors = 180
sensor_starts = np.zeros((n_sensors, 2))
angles = np.deg2rad(np.linspace(0, 360, n_sensors, endpoint=False))

sensor_ends = make_sensor_ends(sensor_range, angles)
print(sensor_ends)
print(sensor_ends.shape)
# sensor_ends = [sensor_end(range, angle) for angle in angles]
# sensor_ends = [sensor_end(range, np.deg2rad(i * (360 / n_sensors))) for i in range(n_sensors)]
# some_end = sensor_end(range, np.deg2rad(3 * (360 / n_sensors)))

# plt.plot(sensor_starts, sensor_ends)
# %%

plt.style.use("ggplot")

import matplotlib.colors as mcolors
# hex_color = "#" + "f0f0f0"
# hex_color = "#" + "82aa8c"  # "f0f0f0"
# not_activated_color = mcolors.to_rgba(hex_color)
not_activated_color = (220 / 255, 220 / 255, 220 / 255, 1.0)
# colors = ["r"] * n_sensors
colors = [not_activated_color] * n_sensors

# activated_color_hex = "#8c4c4c"
activated_color_hex = "#ff5c5c" 

for i in range(0, 20):
    colors[i] = mcolors.to_rgba(activated_color_hex)

for (x1, y1), (x2, y2), color in zip(sensor_starts, sensor_ends, colors):
    # plt.plot([x1, x2], [y1, y2],  f"{color}-", color=color)
    plt.plot([x1, x2], [y1, y2], color=color)

# Turn of axes
plt.gca().set_axis_off()

# Set aspect ratio to 1
plt.gca().set_aspect("equal")

plt.savefig("test-lines.svg")
plt.show()
# %%
