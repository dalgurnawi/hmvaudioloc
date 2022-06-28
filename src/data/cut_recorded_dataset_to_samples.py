import os
import pickle

from pydub import AudioSegment
from tqdm import tqdm


def cut_waves(input_file_name="output.wav"):
    with open("duration_list.dat", "rb") as fp:  # Unpickling
        duration_list = pickle.load(fp)

    with open("names_list.dat", "rb") as np:  # Unpickling
        file_name_list = pickle.load(np)

    folder_name = "all_outputs"
    suffix = input_file_name[:-4]
    # folder_name = input_file_name[:-4]
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass

    start_duration = 0
    counter = 0

    for duration in tqdm(duration_list):
        a = AudioSegment.from_wav(input_file_name)
        slice_of_file = a[start_duration:start_duration + duration]
        file_name = file_name_list[counter].split('\\')[1]
        slice_of_file.export(f"./{folder_name}/{file_name[:-4]}_{suffix}.wav", format="wav")
        start_duration += duration

        counter += 1