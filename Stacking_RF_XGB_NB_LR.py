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
from sklearn.ensemble import StackingClassifier
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
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
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
import warnings
warnings.filterwarnings("ignore")
random.seed(42)
tf.random.set_seed(2)
#
sc = StandardScaler()

#
kfold_splits = 5
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


# Logistic regression
clf_lr = LogisticRegression(C =1.5, solver='sag', fit_intercept=True)

# Naive Bayes Model
clf_nb = GaussianNB()

# XGBoost Model
clf_XGB = XGBClassifier(n_estimators=500,
                        objective= 'binary:logistic', learning_rate=0.1, max_depth=3, max_features="log2", seed=1)

# RF Model
clf_rf = RandomForestClassifier(n_estimators=500, max_features=0.25, criterion="entropy")

clfs = [clf_lr, clf_nb, clf_XGB, clf_rf]

#Create train and test datasets for stacking
dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_val.shape[0], len(clfs)))

#define kfold
kfold = model_selection.StratifiedKFold(n_splits=5)


result_table = pd.DataFrame(columns=['clfs', 'fpr', 'tpr', 'auc'])

#Base model performance
for i, clf in enumerate(clfs):
    scores = model_selection.cross_val_score(clf, X_train, y_train,
    cv=kfold, scoring='accuracy')
    print("##### Base Model %0.0f #####" % i)
    print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),
    scores.std()))
    clf.fit(X_train, y_train)
    print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict
    (X_train), y_train)))

    dataset_blend_train[:, i] = clf.predict_proba(X_train)[:, 1]
    dataset_blend_test[:, i] = clf.predict_proba(X_val)[:, 1]

    y_pred = clf.predict(X_val)
    print("Test Accuracy: %0.2f " % (metrics.accuracy_score(y_pred, y_val)))
    conf=(confusion_matrix(y_pred, y_val))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf)

    display = disp.plot()
    name = 'conf' + str(i + 1) + '.svg'
    plt.savefig(name)
    y_proba = clf.predict_proba(X_val)[::, 1]

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    result_table = result_table.append({'clfs': clf.__class__.__name__,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)

# base models
estimators = [
    ('lr', clf_lr),
    ('nb', clf_nb),
    ('xgb', clf_XGB),
    ('rf', clf_rf),
]

# Generate and fit stacking ensemble
stclf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

scores = model_selection.cross_val_score(stclf, dataset_blend_train, y_train, cv=kfold, scoring='accuracy')
stclf.fit(dataset_blend_train, y_train)

#Stacking ensemble performance
print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("Train Accuracy: %0.2f " % (metrics.accuracy_score(stclf.predict(dataset_blend_train), y_train)))
print("Test Accuracy: %0.2f " % (metrics.accuracy_score(stclf.predict(dataset_blend_test), y_val)))
y_pred = stclf.predict(dataset_blend_test)
conf = confusion_matrix(y_pred, y_val)
disp = ConfusionMatrixDisplay(confusion_matrix=conf)
display = disp.plot()
plt.savefig('stackedconf.svg')
y_proba = stclf.predict_proba(dataset_blend_test)[::,1]
fpr, tpr, _ = roc_curve(y_val, y_proba)
auc = roc_auc_score(y_val, y_proba)
result_table = result_table.append({'clfs': stclf.__class__.__name__,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('clfs', inplace=True)


fig = plt.figure(figsize=(8, 6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))


# Plot AUC Curves
plt.plot([0, 1], [0, 1], color='purple', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("1-Specificity", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("Sensitivity", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')

plt.savefig('auc.svg')
plt.show()


#Calibration curve

proba = stclf.predict_proba(dataset_blend_test)


ax = plt.gca()

ax.set_xlim([-0.1,1.1])
ax.set_ylim([-0.1,1.1])

ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")

fraction_of_positives, mean_predicted_value = calibration_curve(dataset_blend_test, proba, n_bins=20)
ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Stacked Ensemble Calibration")

plt.suptitle('Stacked Ensemble Calibration curve and Brier Loss', size=14)
plt.savefig('calib.svg')
plt.show()
