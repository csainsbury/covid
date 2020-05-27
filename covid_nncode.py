import tensorflow as tf
import numpy as np
from numpy.random import seed
seed(1)
import keras
import sklearn
from keras.layers import Input, Embedding, Reshape, merge, Dropout, Dense, LSTM, core, Activation
from keras.layers import TimeDistributed, Flatten, concatenate, Bidirectional, Concatenate, Conv1D, MaxPooling1D, Conv2D
from keras.utils import np_utils
from keras.engine import Model
from keras.models import Sequential
from keras import layers, optimizers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import random
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import impyute as impy
import my_callbacks
import os
random.seed(42)
tf.random.set_seed(2)
#
sc = StandardScaler()



#
n_samples = 100000
kfold_splits = 4
n_epochs = 40


## data prep ##
# load main data input from R
COVID_data = pd.read_csv('/Users/ryanjmulholland/documents/Data/covsynth.csv')
COVID_data = COVID_data.values

## define X and y
unique_IDs = COVID_data[:, 0]
X = COVID_data[:, 1:(COVID_data.shape[1] - 1)]
y = COVID_data[:, (COVID_data.shape[1] - 1)]

# define kfold splits - split unique IDs to prevent peeking
from sklearn.model_selection import KFold
kf = KFold(n_splits= kfold_splits)
kf.get_n_splits(unique_IDs)
print(kf)

cvscores, aucs, losses, val_losses, acc_ = [], [], [], [], []
#
for index, (train_indices, val_indices) in enumerate(kf.split(unique_IDs)):
    print("Training on fold " + str(index+1) + "/" + str(kfold_splits) + "...")
    print("TRAIN:", train_indices, "TEST:", val_indices)
#
    # ID list from indices:
    train_IDs = unique_IDs[train_indices]
    val_IDs = unique_IDs[val_indices]
#
    # generate sets from intersections
    intersection_train = np.isin(unique_IDs, train_IDs)
    intersection_val = np.isin(unique_IDs, val_IDs)
#
    # Generate batches from indices
    X_train, X_val = X[intersection_train], X[intersection_val]
    y_train, y_val = y[intersection_train], y[intersection_val]
#
    # generate class weight
    cw = np.round(y_train.shape[0] / np.sum(y_train), 0)
#
    # save out files with linkids for output analysis
    np.savetxt("/Users/ryanjmulholland/documents/Data/y_train_" + str(index) +  "_.csv", y_train, delimiter=",")
    np.savetxt("/Users/ryanjmulholland/documents/Data/X_val_" + str(index) +  "_.csv", X_val[:, 0], delimiter=",")
    np.savetxt("/Users/ryanjmulholland/documents/Data/y_val_" + str(index) +  "_.csv", y_val, delimiter=",")
#
    # retain and then remove ID cols
    X_train_IDs, X_val_IDs = unique_IDs[intersection_train], unique_IDs[intersection_val]
#   X_train, X_val = X_train[: , 1:X_train.shape[1]], X_val[:, 1:X_val.shape[1]]
#   y_train, y_val = y_train[:, 0], y_val[:, 0]
#
#   from sklearn.preprocessing import StandardScaler
    scalers = {}
    scalers = StandardScaler()
    X_train = scalers.fit_transform(X_train)
    X_val = scalers.transform(X_val)
#
## FCL setup
    dense_N = 36
    dropout_n = 0.8
    n_batch_size = 1024 # 64
#
    # a = input the hour 0 dataset (2-dimensional: IDs, timesteps)
    time_point_data = Input(shape = (len(X_train[1]), ), dtype='float32', name = 'time_point_data')
#
    fcl_process = Dense(dense_N)(time_point_data)
    fcl_process = Dense(dense_N)(fcl_process)
    fcl_process = Dropout(dropout_n)(fcl_process)
    fcl_process = Dense(dense_N)(fcl_process)
    fcl_process = Dropout(dropout_n)(fcl_process)
#
    main_output = Dense(1, activation = 'sigmoid')(fcl_process)
#
    model = Model(inputs=[time_point_data], outputs=main_output)
#
    print(model.summary())
#
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
    model.compile(optimizer = adam, loss='binary_crossentropy', metrics=['accuracy'])
#
    class_weight = {0: 1.,
                    1: cw, # 1: 20.
                    }
#
    histories = my_callbacks.Histories()
    history = model.fit([X_train], y_train, epochs=n_epochs, batch_size=n_batch_size, validation_data = ([[X_val], y_val]),  class_weight=class_weight, callbacks = [histories])
    model.save("model.h5")
#
    scores = model.evaluate([X_val], y_val, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
#
    auc_values = histories.aucs
    aucs.append(auc_values)
#
    loss_values = history.history['loss']
    losses.append(loss_values)
#
    val_loss_values = history.history['val_loss']
    val_losses.append(val_loss_values)
#
    acc_values = history.history['accuracy']
    acc_.append(acc_values)
#
    y_pred_kfoldValidationSet_asNumber = model.predict([X_val])
    y_preds = y_pred_kfoldValidationSet_asNumber
    val_output = np.column_stack((X_val_IDs, y_pred_kfoldValidationSet_asNumber))
    np.savetxt("/Users/ryanjmulholland/documents/Data/raw_prediction" + str(index) +  "_.csv", val_output, delimiter=",")

# calculate average AUC per epoch
auc_array = np.asarray(aucs)
print(auc_array)
np.savetxt("/Users/ryanjmulholland/documents/Data/aucs_array.csv", auc_array, delimiter=",")
average_aucs = np.mean(auc_array, axis = 0)
print("average_aucs")
print(average_aucs)
np.savetxt("/Users/ryanjmulholland/documents/Data/average_aucs.csv", average_aucs, delimiter=",")

# calculate average loss per epoch
loss_array = np.asarray(losses)
#print(loss_array)
average_losses = np.mean(loss_array, axis = 0)

# calculate average val_loss per epoch
val_loss_array = np.asarray(val_losses)
#print(val_loss_array)
average_val_losses = np.mean(val_loss_array, axis = 0)

# calculate accuracy per epoch
acc_array = np.asarray(acc_)
average_acc = np.mean(acc_array, axis = 0)

# plot losses
import matplotlib
matplotlib.use('Agg') # ensure that matplotlib doesn't try and call a display

import matplotlib.pyplot as plt
loss = average_losses
val_loss = average_val_losses
acc_ = average_acc
print(average_acc)
#acc = history.history['acc']
#val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.legend()
plt.savefig('/Users/ryanjmulholland/documents/Data/loss_valLoss_2param.png', dpi = 300)
plt.clf()

plt.plot(epochs, acc_, 'b', label = 'accuracy per epoch')
plt.legend()
plt.savefig('/Users/ryanjmulholland/documents/Data/accuracyPerEpoch.png', dpi = 300)
plt.clf()

# plot AUROC - use the averaged AUC per epoch value
auc_p = average_aucs
auc_p_losses = average_losses

# plot epoch vs auroc
epochs = range(1, len(auc_p) + 1)
plt.plot(epochs, auc_p, 'bo', label = 'epoch vs average auc per epoch')
plt.legend()
plt.savefig('/Users/ryanjmulholland/documents/Data/auc_plot_2param.png', dpi = 300)
plt.clf()

# plot epoch vs auroc
epochs = range(1, len(auc_p) + 1)
plt.plot(epochs, auc_p_losses, 'bo', label = 'epoch vs average loss per epoch')
plt.legend()
plt.savefig('/Users/ryanjmulholland/documents/Data/auc_loss_plot_2param.png', dpi = 300)
plt.clf()

# plot loss vs auc
plt.plot(auc_p_losses, auc_p, 'bo', label = 'loss vs auc')
plt.legend()
plt.savefig('/Users/ryanjmulholland/documents/Data/aucVSloss_plot_2param.png', dpi = 300)
plt.clf()