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
import sys  
sys.path.insert(0, '../')
import common_plotting_functions

from umap import UMAP

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score, cross_val_predict

# Fix random seed for reproductibility
seed = 42
np.random.seed(seed)

# Function to label data based on threshold
def label_geoeffectiveness_with_threshold(df, threshold):
    df['geoeffective'] = df.dst.apply(lambda x: 1 if x <= threshold else 0)

# Function to fit a model with tensorboard plots
def weighted_fit_with_validation_plot(model, X, y, batch_size, epochs, weights):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model on the actual trainining test
    model = model
    history = model.fit(X, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      sample_weight=weights,  
                      verbose=0,
                      validation_split=0.2,
                      callbacks=[tensorboard_callback])

# Function to fit a model with tensorboard plots
def weighted_fit_with_plot(model, X, y, batch_size, epochs, weights):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model on the actual trainining test
    model = model
    history = model.fit(X, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      sample_weight=weights,  
                      verbose=0,
                      callbacks=[tensorboard_callback])

# Function to fit a model with tensorboard plots with naming convention
def weighted_fit_with_validation_plot_and_name(model, X, y, batch_size, epochs, weights, naming_convention):
    log_dir = "logs/fit/" + naming_convention + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model on the actual trainining test
    model = model
    history = model.fit(X, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      sample_weight=weights,  
                      verbose=0,
                      validation_split=0.2,
                      callbacks=[tensorboard_callback])



# Function to fit a model with tensorboard plots with naming convention
def weighted_fit_with_plot_and_name(model, X, y, batch_size, epochs, weights, naming_convention):
    log_dir = "logs/fit/" + naming_convention + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model on the actual trainining test
    model = model
    history = model.fit(X, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      sample_weight=weights,  
                      verbose=0,
                      callbacks=[tensorboard_callback])

# Without using weights
def fit_with_plot(model, X, y, batch_size, epochs):
    weights = np.ones(X.shape[0])
    weighted_fit_with_plot(model, X, y, batch_size, epochs, weights)

# Implement a transformer (from continuous predictions to labels)
get_labels = lambda x: 1 if x >= 0.5 else 0 
labeler = np.vectorize(get_labels)

# Function to get prediction labels
def predict_labels(test_data):
    return labeler(model.predict(test_data))
    
# Function to plot confusion matrix and print evaluation metrics
def evaluate_performance(actual, predicted):
    # Create confusion matrix
    cm = confusion_matrix(actual, predicted)
    # Plot confusion matrix
    common_plotting_functions.plot_confusion_matrix(cm)
    # Compute metrics
    accuracy = accuracy_score(actual, predicted)
    recall = recall_score(actual, predicted)
    precision = precision_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    # Print metrics
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)      
    print("Precision: ", precision) 
    print("F1 score: ", f1)
    print("Specificity: ", specificity)
    # Return metrics
    return (accuracy, recall, precision)


# Create a normalizer to be further used in pipelines
normalizer = Normalizer()

# Pipeline for normalization and UMAP embedding 
umap = UMAP(random_state=seed)
umap_pipeline = Pipeline([("normalization", normalizer), ("umap", umap)])

# Function to get UMAP embedding
def umap_embed(df):
    return umap_pipeline.fit_transform(df)

# Function to plot 2 plots, one colored based on the actual labels, the other based on the predicted ones
def plot_test(test_set, true_label, predicted_label):
    # Data visualization in 2D, with points' color based on the real labels
    fig = px.scatter(test_set, x=0, y=1, color=true_label)
    fig.show()

    # Data visualization in 2D, with points' color based on the predicted labels
    fig = px.scatter(test_set, x=0, y=1, color=predicted_label)
    fig.show() 



# Get prediction label for binary classification
def get_binary_prediction_type(predicted, actual):
    if (predicted == 1 and actual == 1):
        return 'TP'
    if (predicted == 1 and actual == 0):
        return 'FP'
    if (predicted == 0 and actual == 0):
        return 'TN'
    if (predicted == 0 and actual == 1):
        return 'FN'
    return 'Unkonwn'

# Function to create a more comprehensive prediction data frame
def get_predictions_df(numeric_columns_used, test_examples, test_labels, predicted_labels):
    # Concatenate all given column
    predictions_df = pd.concat([test_examples, test_labels], axis=1)
    predictions_df['predicted'] = predicted_labels
    # Label the predictions with their type
    predictions_df['prediction_type'] = predictions_df.apply(lambda row: get_binary_prediction_type(row['predicted'], 
                                                                          row['geoeffective']), 
                                                                          axis=1)
                                                                          
    # Create the UMAP embedding
    umap_embedding = umap_embed(test_examples[numeric_columns_used])
    # Add UMAP values to the dataframe
    predictions_df['umap_0'] = umap_embedding[:, 0]
    predictions_df['umap_1'] = umap_embedding[:, 1]
    
    return predictions_df
    
# Function to plot the more comprehensive prediction data umap embedded
def plot_comprehensive_predictions_umap(df):
    hover_columns = df.columns[~df.columns.isin(['umap_0', 'umap_1',
                                                'geoeffective', 'prediction_type',
                                                'predicted'])]
    fig = px.scatter(df, x='umap_0', y='umap_1', color='prediction_type', 
                     color_discrete_sequence=["blue", "yellow", "green", "red"],
                     hover_data=df[hover_columns])
    fig.show()    

    
# Function to cross validate the classifier
def perform_cross_validation(model, X, y):
    scoring = ['accuracy', 'recall', 'precision', 'f1']
    return cross_validate(model, 
                         X, 
                         y, 
                         scoring=scoring,
                         return_train_score=True, 
                         cv=5)
                         
# Function to cross validate the classifier, with weights 
def perform_cross_validation(model, X, y):
    scoring = ['accuracy', 'recall', 'precision', 'f1']
    return cross_validate(model, 
                         X, 
                         y, 
                         scoring=scoring,
                         return_train_score=True, 
                         cv=5)
                         
# Function to cross validate the classifier, with weights 
def perform_weighted_cross_validation(model, weights, X, y):
    # Set training weights
    fit_params = {
            'sample_weight': weights
    }
    scoring = ['accuracy', 'recall', 'precision', 'f1']
    return cross_validate(model, 
                         X, 
                         y, 
                         fit_params=fit_params, 
                         scoring=scoring, 
                         return_train_score=True,
                         cv=5)
                         
# Function to print mean CV metrics (recall & precision) for train/test
def print_cv_metrics(cv_results):
    # Mean cross-validation scores - train folds
    print('---TRAIN---')
    print('Mean recall: ', np.mean(cv_results['train_recall']))
    print('Mean precision: ', np.mean(cv_results['train_precision']))
    print('Mean accuracy: ', np.mean(cv_results['train_accuracy']))
    print('Mean F1: ', np.mean(cv_results['train_f1']))
    # Mean cross-validation scores - test folds
    print('---TEST---') 
    print('Mean recall: ', np.mean(cv_results['test_recall']))
    print('Mean precision: ', np.mean(cv_results['test_precision']))
    print('Mean accuracy: ', np.mean(cv_results['test_accuracy']))
    print('Mean F1: ', np.mean(cv_results['test_f1']))  

# Function to print mean CV metrics (recall & precision) for test only
def print_test_cv_metrics(cv_results):
    # Mean cross-validation scores - test folds
    print('---TEST---') 
    print('Mean recall: ', np.mean(cv_results['test_recall']))
    print('Mean precision: ', np.mean(cv_results['test_precision']))
    print('Mean accuracy: ', np.mean(cv_results['test_accuracy']))
    print('Mean F1: ', np.mean(cv_results['test_f1']))  
