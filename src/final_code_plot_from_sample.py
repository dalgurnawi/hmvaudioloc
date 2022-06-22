import tensorflow as tf
import tensorflow_io as tfio
# import glob
import datetime
from multiprocessing import Process, Queue
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import sys
import os
from itertools import cycle
import matplotlib.colorbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


def process(audio):
    audio = tf.cast(audio, tf.float32) / 32768.0
    spectrogram = tfio.audio.spectrogram(
        audio, nfft=1024, window=1024, stride=64)
    spectrogram = tfio.audio.melscale(
        spectrogram, rate=8000, mels=64, fmin=0, fmax=2000)
    spectrogram /= tf.math.reduce_max(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.image.resize(spectrogram, (64, 64))
    spectrogram = tf.transpose(spectrogram, perm=(1, 0, 2))
    spectrogram = spectrogram[::-1, :, :]
    return spectrogram


def map_sample_files(file_path):
    audio_tensor = tfio.audio.AudioIOTensor(file_path).to_tensor()
    audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
    spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
    spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
    full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
    return np.array([full_spectrogram])


def model_querry(file_path, direction_model, elevation_model):
    test_input = map_sample_files(file_path)
    # if len(test_input) == 0:
    #     return None
    direction_prediction = direction_model.predict(test_input)
    direction_prediction_probabilities = tf.math.top_k(direction_prediction, k=8)
    top_8_direction_scores = direction_prediction_probabilities.values.numpy()[0]
    direction_dict_class_entries = direction_prediction_probabilities.indices.numpy()[0]

    elevation_prediction = elevation_model.predict(test_input)
    elevation_prediction_probabilities = tf.math.top_k(elevation_prediction, k=3)
    top_3_elevation_scores = elevation_prediction_probabilities.values.numpy()[0]
    elevation_dict_class_entries = elevation_prediction_probabilities.indices.numpy()[0]
    return (top_8_direction_scores,
            direction_dict_class_entries,
            top_3_elevation_scores,
            elevation_dict_class_entries)


def cuboid_data(center, size=(1, 1, 1)):
    # code taken from
    # http://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    y = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
                  [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
                  [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
                  [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  # x coordinate of points in inside surface
    x = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
                  [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
                  [o[1], o[1], o[1], o[1], o[1]],  # y coordinate of points in outside surface
                  [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])  # y coordinate of points in inside surface
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],  # z coordinate of points in bottom surface
                  [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],  # z coordinate of points in upper surface
                  [o[2], o[2], o[2] + h, o[2] + h, o[2]],  # z coordinate of points in outside surface
                  [o[2], o[2], o[2] + h, o[2] + h, o[2]]])  # z coordinate of points in inside surface
    return x, y, z


def plot_cube_at(pos=(0,0,0), c="g", alpha=1.0, ax=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
        return ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=alpha)


def plotMatrix(ax, sector_dict, elevation_dict, cax, alpha=0.1):
    # plot a Matrix
    #x and y and z coordinates
    x = np.array(range(3))
    y = np.array(range(3))
    z = np.array(range(3))
    sector_0_down = sector_dict[0]*elevation_dict[0]
    sector_0_level = sector_dict[0]*elevation_dict[1]
    sector_0_up = sector_dict[0]*elevation_dict[2]
    sector_1_down = sector_dict[1]*elevation_dict[0]
    sector_1_level = sector_dict[1]*elevation_dict[1]
    sector_1_up = sector_dict[1]*elevation_dict[2]
    sector_2_down = sector_dict[2]*elevation_dict[0]
    sector_2_level = sector_dict[2]*elevation_dict[1]
    sector_2_up = sector_dict[2]*elevation_dict[2]
    sector_3_down = sector_dict[3]*elevation_dict[0]
    sector_3_level = sector_dict[3]*elevation_dict[1]
    sector_3_up = sector_dict[3]*elevation_dict[2]
    sector_4_down = sector_dict[4]*elevation_dict[0]
    sector_4_level = sector_dict[4]*elevation_dict[1]
    sector_4_up = sector_dict[4]*elevation_dict[2]
    sector_5_down = sector_dict[5]*elevation_dict[0]
    sector_5_level = sector_dict[5]*elevation_dict[1]
    sector_5_up = sector_dict[5]*elevation_dict[2]
    sector_6_down = sector_dict[6]*elevation_dict[0]
    sector_6_level = sector_dict[6]*elevation_dict[1]
    sector_6_up = sector_dict[6]*elevation_dict[2]
    sector_7_down = sector_dict[7]*elevation_dict[0]
    sector_7_level = sector_dict[7]*elevation_dict[1]
    sector_7_up = sector_dict[7]*elevation_dict[2]
    ludwig_down = 0
    ludwig_level = 0
    ludwig_up = 0
    data = np.array([[[sector_5_down, sector_5_level, sector_5_up],
                          [sector_4_down, sector_4_level, sector_4_up],
                          [sector_3_down, sector_3_level, sector_3_up]],
                         [[sector_6_down, sector_6_level, sector_6_up],
                          [ludwig_down, ludwig_level, ludwig_up],
                          [sector_2_down, sector_2_level, sector_2_up]],
                         [[sector_7_down, sector_7_level, sector_7_up],
                          [sector_0_down, sector_0_level, sector_0_up],
                          [sector_1_down, sector_1_level, sector_1_up]]])
    #data = np.round_(data*100, decimals=2)
    # print(data.mean())
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['white', 'yellow', '#e00b19'])
    norm = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=0.5)
    colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i,j,k])

    cubes = []
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    #if i >= 0.2 and j >= 0.2 and k >= 0.2:
                    if data[i,j,k] <= data.mean():
                        cube = plot_cube_at(pos=(xi, yi, zi), c= "white", alpha=data[i, j, k],  ax=ax)
                        cubes.append(cube)
                    elif data[i,j,k] >= data.mean():
                        cube = plot_cube_at(pos=(xi, yi, zi), c=colors(i,j,k), alpha=data.max(), ax=ax)
                        cubes.append(cube)

    if cax !=None:
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
        cbar.set_ticks(np.unique(data))
        # set the colorbar transparent as well
        cbar.solids.set(alpha=alpha)

    # draw point in the middle
    point = ax.scatter([1], [1], [1], color="g", s=1000)
    cubes.append(point)

    # draw arrow facing the front
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
    a = Arrow3D([1, 1], [1, 3], [1, 1], **arrow_prop_dict)
    arrow = ax.add_artist(a)
    cubes.append(arrow)
    return cubes


def update_plot(i, ):
    global ax_cb
    global ax
    global sector_dict
    global elevation_dict
    ax.clear()
    cubes = plotMatrix(ax, sector_dict, elevation_dict, ax_cb)
    return cubes


reconstructed_direction_model = tf.keras.models.load_model("retrained_complex_paper_code_4.model")
reconstructed_elevation_model = tf.keras.models.load_model("paper_code_7_elevation.model")

(sector_output_probs,
 sector_output_labels,
 elevation_output_probs,
 elevation_output_labels) = model_querry("all_outputs_complex/4_george_0_complex_level_sector_7.wav", reconstructed_direction_model,
                                         reconstructed_elevation_model)

sector_dict = dict(zip(sector_output_labels, sector_output_probs))
# print(sector_dict)
elevation_dict = dict(zip(elevation_output_labels, elevation_output_probs))
# print(elevation_dict)

print(sector_dict)
print(elevation_dict)

# plt.ion()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')
ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])  # axes for the colour bar
ax.set_aspect('auto')

ani = FuncAnimation(fig,
                    update_plot,
                    # fargs=(sector_dict, elevation_dict),
                    interval=500,
                    # blit=True,
                    )

# plt.draw()
# fig.canvas.draw()
# fig.canvas.flush_events()

plt.show()