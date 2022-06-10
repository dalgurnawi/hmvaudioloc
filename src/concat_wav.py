import glob
import os
import pickle

from pydub import AudioSegment
from tqdm import tqdm


def concat_waves():
    file_list = []
    duration_list = []
    file_name_list = glob.glob("recordings/*.wav")
    for audio_file in file_name_list:
        sound = AudioSegment.from_wav(audio_file)
        duration_list.append((len(sound)))
        file_list.append(sound)

    with open("duration_list.dat", "wb") as fp:  # Pickling
        pickle.dump(duration_list, fp)

    with open("names_list.dat", "wb") as np:  # Pickling
        pickle.dump(file_name_list, np)

    combined_sounds = sum(file_list)
    combined_sounds.export("output.wav", format="wav")


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


# list_of_all_outputs = ["level_sector_0.wav", "45up_sector_0.wav", "45down_sector_0.wav",
#                         "level_sector_1.wav", "45up_sector_1.wav", "45down_sector_1.wav",
#                         "level_sector_2.wav", "45up_sector_2.wav", "45down_sector_2.wav",
#                         "level_sector_3.wav", "45up_sector_3.wav", "45down_sector_3.wav",
#                         "level_sector_4.wav", "45up_sector_4.wav", "45down_sector_4.wav",
#                         "level_sector_5.wav", "45up_sector_5.wav", "45down_sector_5.wav",
#                         "level_sector_6.wav", "45up_sector_6.wav", "45down_sector_6.wav",
#                         "level_sector_7.wav", "45up_sector_7.wav", "45down_sector_7.wav",
#                        ]

list_of_all_outputs = ["level_sector_6.wav"]
for afile in tqdm(list_of_all_outputs):
    cut_waves(input_file_name=afile)


