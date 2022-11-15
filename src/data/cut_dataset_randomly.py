import os
from tqdm import tqdm
from pydub import AudioSegment
import random
import glob

sample_folder = 'sound_samples_complex'
input_file_name_list = glob.glob(f'{sample_folder}/*.wav')
output_folder_name = sample_folder + "_cut"

try:
    os.mkdir(output_folder_name)
except FileExistsError:
    pass

min_sample_length = 600
max_sample_length = 1500

for input_file_name in tqdm(input_file_name_list):
    start_duration = 0
    counter = 0
    suffix = input_file_name.split('/')[-1][:-4]
    a = AudioSegment.from_wav(input_file_name)
    max_samples = len(a) / min_sample_length
    if max_samples < 0:
        max_samples = 1
        max_sample_length = min_sample_length

    for sample in tqdm(range(int(max_samples))):
        duration = random.randint(min_sample_length, max_sample_length)
        curr_end_dur = start_duration + duration
        if curr_end_dur > len(a):
            break
        slice_of_file = a[start_duration:curr_end_dur]
        slice_of_file.export(f"./{output_folder_name}/{suffix}_{counter}.wav", format="wav")
        start_duration += duration

        counter += 1

