#!/usr/bin/env python
# coding: utf-8

import datetime
import dash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf

from umap import UMAP

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score, cross_val_predict

# Fix random seed for reproductibility
seed = 42
np.random.seed(seed)

# Function to plot losses
def plot_losses(training_loss, validation_loss):
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title('Model losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

# Function to plot the confusion matrix
# As from: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
def plot_confusion_matrix(cf_matrix):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n" for v1, v2 in
              zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cf_matrix, xticklabels=[0,1], yticklabels=[0,1], annot=labels, fmt='', cmap='Blues')
    plt.title("Confusion matrix")
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.show()

# Create a normalizer to be further used in pipelines
normalizer = Normalizer()

# Pipeline for normalization and UMAP embedding 
umap = UMAP(random_state=seed)
umap_pipeline = Pipeline([("normalization", normalizer), ("umap", umap)])

# Function to get UMAP embedding
def umap_embed(df):
    return umap_pipeline.fit_transform(df)

# Function to plot umap
def plot_umap(X, y):
    umap = umap_embed(X)
    fig = px.scatter(umap, x=0, y=1, color=y)
    fig.show()

# Function to plot 2 plots, one colored based on the actual labels, the other based on the predicted ones
def plot_test(test_set, true_label, predicted_label):
    # Data visualization in 2D, with points' color based on the real labels
    fig = px.scatter(test_set, x=0, y=1, color=true_label)
    fig.show()

    # Data visualization in 2D, with points' color based on the predicted labels
    fig = px.scatter(test_set, x=0, y=1, color=predicted_label)
    fig.show() 

# Function to plot the more comprehensive prediction data umap embedded
def plot_comprehensive_predictions_umap(df):
    hover_columns = df.columns[~df.columns.isin(['umap_0', 'umap_1',
                                                'geoeffective', 'prediction_type',
                                                'predicted'])]
    fig = px.scatter(df, x='umap_0', y='umap_1', color='prediction_type', 
                     color_discrete_sequence=["blue", "yellow", "green", "red"],
                     hover_data=df[hover_columns])
    fig.show()    