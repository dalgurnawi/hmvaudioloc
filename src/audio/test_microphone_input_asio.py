#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


device = 1
channels = [1]
asio_channel = [0]
window = 200
interval = 30
blocksize = 10
samplerate = 44100 # if errors check 48000
downsample = 10


asio_in = sd.AsioSettings(channel_selectors=asio_channel)
mapping = [c - 1 for c in channels]  # Channel numbers start with 1
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::downsample, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines


try:
    if samplerate is None:
        device_info = sd.query_devices(device, 'input')
        samplerate = device_info['default_samplerate']

    length = int(window * samplerate / (1000 * downsample))
    plotdata = np.zeros((length, len(channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(channels) > 1:
        ax.legend(['channel {}'.format(c) for c in channels],
                  loc='lower left', ncol=len(channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)

    ax.tick_params(bottom=False,
                   top=False,
                   labelbottom=False,
                   right=False,
                   left=False,
                   labelleft=False)

    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=device,
        channels=channels,
        samplerate=samplerate,
        callback=audio_callback,
        extra_settings=asio_in)

    ani = FuncAnimation(fig,
                        update_plot,
                        interval=interval,
                        blit=True)

    with stream:
        plt.show()

except KeyboardInterrupt:
    sys.exit()

