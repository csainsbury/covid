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
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import argmax
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import random
import pydot
import graphviz
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
n_epochs = 100

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
    n_batch_size = 2048 # 64
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

#fit the model on the COVID dataset
def fit_model(X_train, y_train):
#define model
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
    histories = my_callbacks.Histories()
    #model fit
    model.fit([X_train], y_train, epochs=n_epochs, batch_size=n_batch_size, validation_data = ([[X_val], y_val]),  class_weight=class_weight, callbacks = [histories])
    return model

#fit and save the models
n_members = 5
for i in range(n_members):
    #fit model
    model = fit_model(X_train, y_train)
    # save models
    filename = '/Users/ryanjmulholland/documents/nnmodels/model_' + str(i + 1) + '.h5'
    model.save(filename)

# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = '/Users/ryanjmulholland/documents/nnmodels/model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='sigmoid')(merge)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # fit model
    model.fit(X, inputy, epochs=300, verbose=0)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)

# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# define ensemble model
stacked_model = define_stacked_model(members)


# fit stacked model on test dataset
fit_stacked_model(stacked_model, X_val, y_val)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, X_val)
yhat = argmax(yhat, axis=1)
acc = accuracy_score(y_val, yhat)
print('Stacked Test Accuracy: %.3f' % acc)

model.save('stacked_model.h5')

def get_model():
    return load_model('stacked_model.h5')

clf_nn = KerasClassifier(build_fn = get_model, epochs=1, batch_size =10)

#XGBoost Model
clf_XGB = XGBClassifier(n_estimators=100,
                        objective= 'binary:logistic',seed=1)

#RF Model
clf_rf = RandomForestClassifier(n_estimators=100)

clfs = [clf_nn, clf_XGB, clf_rf]

# Creating train and test sets for blending
dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_val.shape[0], len(clfs)))

#define kfold
kfold = model_selection.KFold(n_splits=kfold_splits)

for i, clf in enumerate(clfs):
    scores = model_selection.cross_val_score(clf, X_train, y_train,
    cv=kf, scoring='accuracy')
    print("##### Base Model %0.0f #####" % i)
    print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),
    scores.std()))
    clf.fit(X_train, y_train)
    print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict
    (X_train), y_train)))

    dataset_blend_train[:, i] = clf.predict_proba(X_train)[:, 1]
    dataset_blend_test[:, i] = clf.predict_proba(X_val)[:, 1]
    print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict
                                                            (X_val), y_val)))


print("##### Meta Model #####")
clf = LogisticRegression()
scores = model_selection.cross_val_score(clf, dataset_blend_train, y_train, cv=kfold, scoring='accuracy')
clf.fit(dataset_blend_train, y_train)
print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_train), y_train)))
print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_test), y_val)))