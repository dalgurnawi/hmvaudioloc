import gdown

def download_augmented_spoken_mnist_dataset():
    file_id = 'TAKE ID FROM SHAREABLE LINK'
    destination = 'DESTINATION FILE ON YOUR DISK'
    gdown.download(file_id, destination, quiet=False, fuzzy=True)


def download_real_world_spoken_mnist_dataset():
    file_id = 'TAKE ID FROM SHAREABLE LINK'
    destination = 'DESTINATION FILE ON YOUR DISK'
    gdown.download(file_id, destination, quiet=False, fuzzy=True)


def download_real_world_complex_dataset():
    file_id = 'TAKE ID FROM SHAREABLE LINK'
    destination = 'DESTINATION FILE ON YOUR DISK'
    gdown.download(file_id, destination, quiet=False, fuzzy=True)


def download_real_world_spoken_test_dataset():
    file_id = 'TAKE ID FROM SHAREABLE LINK'
    destination = 'DESTINATION FILE ON YOUR DISK'
    gdown.download(file_id, destination, quiet=False, fuzzy=True)

# if __name__ == "__main__":
#     file_id = 'https://drive.google.com/file/d/1ijM6j1wytmEg-YWzxahteIokQ4ISBZtk/view?usp=sharing'
#     destination = '../../data/datasets/test.rar'
#     # download_file_from_google_drive(file_id, destination)
#     gdown.download(file_id, destination, quiet=False, fuzzy=True)
