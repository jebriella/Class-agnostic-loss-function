from load_models import load_models
from data_loader import data_loader
from feature_setup import feature_setup
from my_vote import my_vote
import numpy as np

models = load_models()

path = "C:/full_batch/batch_1"
patients, images, masks = data_loader(path)

subt = np.load("subt.npy")
div = np.load("div.npy")

for i in range(len(patients)):
    print(patients[i])
    data = feature_setup(images[i], masks[i], subt, div)
    my_vote(patients[i], masks[i], data, models, path)
