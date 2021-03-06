import tensorflow as tf
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_test_dataset():
    pass


def test_model(model):
    pass


def test_all_trained_models(suffix):
    dirname = os.path.dirname(__file__)
    for model in glob.glob(os.path.join(dirname, "../../data/pickled_models/*" + suffix)):
        test_model(model)


