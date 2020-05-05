from U_Net_2D import model_unet
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from metrics import dice_coef, loss_dl_ce, dice_class1, dice_class2, dice_class3, b_cross
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import multi_gpu_model
from DataGenerator import DataGenerator
import math
import os

batch_s = 8
n_train = 35296 #35321
n_val = 8800 #8831

train_img_path = 'left+ax/images'
train_mask_path = 'left+ax/masks'

val_img_path = 'left+ax/val/images'
val_mask_path = 'left+ax/val/masks'

train_list = os.listdir(train_img_path)
val_list = os.listdir(val_img_path)

train_gen = DataGenerator(train_img_path, train_mask_path, train_list, batch_size = batch_s)
val_gen = DataGenerator(val_img_path, val_mask_path, val_list, batch_size = batch_s)

model = model_unet(img_size = [256,256], base = 128, nr_classes = 3, dept = 4, batch_norm = True)#, s_dropout = 0.4)

n_epochs = 18

#model = multi_gpu_model(model, gpus=2)

model.compile(loss=loss_dl_ce, optimizer = Adam(lr=0.001), metrics=[dice_coef, dice_class1, dice_class2, dice_class3])#, precision, recall])

History = model.fit_generator(train_gen, 
                              epochs = n_epochs, 
                              steps_per_epoch = n_train/batch_s,
                              validation_data = val_gen,
                              validation_steps = n_val/batch_s,
                              max_queue_size = 2)
                              #workers = 10,
                              #use_multiprocessing = True)

#History = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data = (X_val, y_val))

plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot(np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.show()