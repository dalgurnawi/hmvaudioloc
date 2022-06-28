import pickle

with open("duration_list.dat", "rb") as fp:  # Unpickling
    duration_list = pickle.load(fp)

with open("names_list.dat", "rb") as np:  # Unpickling
    file_name_list = pickle.load(np)

print(len(duration_list))
print(len(file_name_list))