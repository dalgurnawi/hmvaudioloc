import glob
import os
import pickle

from pydub import AudioSegment
from tqdm import tqdm


def concat_waves():
    file_list = []
    duration_list = []
    file_name_list = glob.glob("recordings/*.wav")
    for audio_file in tqdm(file_name_list):
        sound = AudioSegment.from_wav(audio_file)
        duration_list.append((len(sound)))
        file_list.append(sound)

    with open("duration_list.dat", "wb") as fp:  # Pickling
        pickle.dump(duration_list, fp)

    with open("names_list.dat", "wb") as np:  # Pickling
        pickle.dump(file_name_list, np)

    combined_sounds = sum(file_list)
    combined_sounds.export("output.wav", format="wav")
