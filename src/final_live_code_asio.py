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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# super impoertant int numbers of the mic devices
# you can use print(sd.query_devices()) to get the list with indices
# mic_nums = (50, 53)

# mic_nums = (0, 1)

device_num = 21
asio_channel_nums = ([0], [1])

# RATE = 48000
RATE = 44100
"""
chunk is basically the resolution of how often callback function is called

Below I am calling callback 10 times a second.
44100 is samples per second, so, 1/10th is so 4410 chunk is saved 10 times per second
seems to work better with smaller chunk size. keeping it at 1000th of RATE

"""
CHUNK = int(RATE / 1000)
CHANNELS = 1
sample_length = 1.0
# db_activation_threshold = -25


def recorder(start, device_num, channel_selector):
    sampling = False
    running = False
    my_iterator = cycle([1, 2])
    file_num = 0
    # counter = 0
    # db_activation_mean_list = []

    q = Queue()
    file = sf.SoundFile(f'./recordings_multimic/temp_{device_num}_{channel_selector[0]}_file_{file_num}.wav',
                        mode='w',
                        samplerate=RATE,
                        channels=CHANNELS)
    # print('Created', mic_num)

    def mic_processor(data, frames, time, status):
        if (status):
            # RIP recording :/
            print(status)  # Only status ever thrown is input overflow
            global running
            running = False
            sys.exit()
        q.put(data.copy())

    while not running:
        if datetime.datetime.now() >= start:
            asio_in = sd.AsioSettings(channel_selectors=channel_selector)
            # asio_out = sd.AsioSettings(channel_selectors=[12, 13])
            extra_settings = asio_in
            stream = sd.InputStream(samplerate=RATE,
                                    blocksize=CHUNK,
                                    device=device_num,
                                    channels=CHANNELS,
                                    callback=mic_processor,
                                    extra_settings=extra_settings)
            stream.start()
            # print('Microphone', mic_num, 'recording')
            # time.sleep(10)
            running = True

    while running:
        try:
            if not sampling:
                s_time = time.time()
                # time.sleep(1)
                sampling = True
            # print(f'delta time of sampling: {time.time() - s_time}')
            # print("chunk passed mic: ", mic_num)

            # indata = q.get()

            # weightedData = weighting.weight_signal(indata, RATE, 'A')
            # dBa = 20 * np.log10(rms_flat(weightedData))
            # original = 20 * np.log10(rms_flat(indata))
            # # if len(db_activation_mean_list) < 10:
            # #     db_activation_mean_list.append(dBa)
            # # else:
            # #     db_activation_mean_list.pop(0)
            # #     db_activation_mean_list.append(dBa)
            # #     db_activation_threshold = np.mean(db_activation_mean_list) + 10
            # print(f'mic{mic_num}: o:{original}dB w:{dBa}dB')
            # if dBa > db_activation_threshold and not sampling:
            #     print("sample recording started")
            #     s_time = time.time()
            #     sampling = True
            if sampling and (time.time() - s_time) < sample_length:
                # print("writing sample queue")
                file.write(q.get())
            if sampling and (time.time() - s_time) > sample_length:
                print("flushing queue to file")
                file.flush()
                file.close()
                # counter += 1

                file = sf.SoundFile(f'./recordings_multimic/temp_{device_num}_{channel_selector[0]}_file_{file_num}.wav',
                                    mode='w',
                                    samplerate=RATE,
                                    channels=CHANNELS)
                file_num = next(my_iterator)
                # file = sf.SoundFile(f'./recordings_multimic/{file_name}_{mic_num}.wav', mode='w', samplerate=RATE,
                #                     channels=CHANNELS)
                # q.empty()
                sampling = False
                # time.sleep(1)

        except KeyboardInterrupt:
            stream.stop()
            running = False


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


def map_sample_files(device_num, channel_selectors, file_check_num):
    # file_num = next(my_iterator)
    try:
        audio_tensor_1 = tfio.audio.AudioIOTensor(f"./recordings_multimic/temp_{device_num}_{channel_selectors[0][0]}_file_{file_check_num}.wav").to_tensor()
        audio_tensor_2 = tfio.audio.AudioIOTensor(f"./recordings_multimic/temp_{device_num}_{channel_selectors[1][0]}_file_{file_check_num}.wav").to_tensor()
    except tf.errors.InvalidArgumentError or tf.errors.OutOfRangeError or tf.errors.NotFoundError or tf.errors.NotFoundError:
        return
    spectrogram_1 = process(audio_tensor_1.numpy().squeeze().astype("float32"))
    spectrogram_2 = process(audio_tensor_2.numpy().squeeze().astype("float32"))
    full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
    return np.array([full_spectrogram])


def model_querry(direction_model, elevation_model, device_num, channel_selectors, file_check_num):
    test_input = map_sample_files(device_num, channel_selectors, file_check_num)
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
                        cube = plot_cube_at(pos=(xi, yi, zi), c= "white", alpha=0,  ax=ax)
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
    return cubes


def update_plot(i, ):
    global ax_cb
    global ax
    global sector_dict
    global elevation_dict
    ax.clear()
    cubes = plotMatrix(ax, sector_dict, elevation_dict, ax_cb)
    return cubes


if __name__ == '__main__':
    start = datetime.timedelta(seconds=5) + datetime.datetime.now()

    reconstructed_direction_model = tf.keras.models.load_model("retrained_complex_paper_code_7.model")
    reconstructed_elevation_model = tf.keras.models.load_model("paper_code_7_elevation.model")

    sector_output_probs = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sector_output_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    elevation_output_probs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    elevation_output_labels = np.array([0, 1, 2])
    sector_dict = dict(zip(sector_output_labels, sector_output_probs))
    elevation_dict = dict(zip(elevation_output_labels, elevation_output_probs))

    p1 = Process(target=recorder, args=(start, device_num, asio_channel_nums[0]))
    p2 = Process(target=recorder, args=(start, device_num, asio_channel_nums[1]))

    p1.start()
    p2.start()

    our_iterator = cycle([1, 2])
    file_check_num = 0
    time.sleep(7)

    plt.ion()

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

    while True:
        try:
            # time.sleep(0.0001)
            if not p1.is_alive() or not p2.is_alive():
                # print('Something went wrong with one of the processes, gotta exit :/')
                p1.terminate()
                p2.terminate()
                sys.exit()

        except KeyboardInterrupt:
            print('Writing to file')
            time.sleep(0.5)
            while p1.is_alive() or p2.is_alive():
                time.sleep(0.1)
            break

        try:
            time.sleep(0.1)
            try:
                (sector_output_probs, sector_output_labels, elevation_output_probs, elevation_output_labels) = model_querry(reconstructed_direction_model, reconstructed_elevation_model, device_num, asio_channel_nums, file_check_num)
            except ValueError:
                sector_dict = sector_dict
                elevation_dict = elevation_dict

            file_check_num = next(our_iterator)

            sector_dict = dict(zip(sector_output_labels, sector_output_probs))
            # print(sector_dict)
            elevation_dict = dict(zip(elevation_output_labels, elevation_output_probs))
            # print(elevation_dict)

            plt.draw()
            fig.canvas.draw()
            fig.canvas.flush_events()

        except tf.errors.InvalidArgumentError:
            continue
