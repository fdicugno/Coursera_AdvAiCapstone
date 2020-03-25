# Constants

from IPython.display import display, HTML

import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf



def printAsTable(df, limit = None):
    '''
    Displays a Spark dataframe as HTML table

    :param  df:   the Spark dataframe to print
    :param  limit:   the number of records to print. Default None
    '''
    if limit is None :
        return HTML(df.toPandas().to_html())
    else :
        return HTML(df.limit(limit).toPandas().to_html())

def writeParquet(df = None, filename = None):
    '''
    Writes the dataframe {df} in in {filename} in Parquet format

    :param df:  the spark dataframe
    :param filename:    the destination filename
    '''
    print('Storing to parquet file {}...'.format(filename))
    df.write.parquet(filename)
    print('Complete!')


def readParquet(spark = None, filename = None):
    '''
    Restores a spark dataframe from a parquet file
    '''
    print('Restoring {}...'.format(filename))
    df = spark.read.parquet(filename)
    print('Complete!')

    return df

#https://runawayhorse001.github.io/LearningApacheSpark/classification.html



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def set_random_seed(seed):
    '''
    Sets the random seed for numpy and tensorflow

    :param seed: the random seed
    '''
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def save_feature_cols(filename, corr_features_cols, features_cols, dl_features_cols):
    write_json( { FEATURES_CONFIG_CORRELATION: corr_features_cols, FEATURES_CONFIG_STANDARD: features_cols, FEATURES_CONFIG_DEEP_LEARNIN: dl_features_cols }, filename )

def write_json( data, filename ):
    print('writing ' + filename)
    with open(filename, 'w') as f:
        json.dump(data, f)
    print('Complete!')

def read_feature_cols(filename, selected=None):
    '''
    reads the list of the features cols from a json file and returns the three name lists: FEATURES_CONFIG_CORRELATION, FEATURES_CONFIG_STANDARD, FEATURES_CONFIG_DEEP_LEARNIN

    :param filename: the name of the JSON file
    :param selected: the name of the list to be selected. If None then all the three lists are returned. Default: None
    '''
    json_data = read_json(filename)

    if selected is not None:
        return json_data[selected]
    else:
        return json_data[FEATURES_CONFIG_CORRELATION], json_data[FEATURES_CONFIG_STANDARD], json_data[FEATURES_CONFIG_DEEP_LEARNIN] 

def read_json( filename ):
    print('Reading ' + filename)

    data = None
    with open(filename) as json_file:
        data = json.load(json_file)
    
    return data

######## CONSTANTS ########

#Filenames
PAYMENTS_CSV_FILENAME       = 'data/PS_20174392719_1491204439457_log.csv'
PAYMENTS_PQT_FILENAME       = 'data/payments.v2.parquet'
PAYMENTS_ENC_PQT_FILENAME   = 'data/payments_enc.v6.parquet'

FEATURES_CONFIG_FILENAME    = PAYMENTS_ENC_PQT_FILENAME + '.json'

FEATURES_CONFIG_CORRELATION  = 'corr_features_cols'
FEATURES_CONFIG_STANDARD     = 'features_cols'
FEATURES_CONFIG_DEEP_LEARNIN = 'dl_features_cols'

FEATURES_CONFIG_IDX_SUFFIX   = '_indexed'

AUTOENC_X_TRAIN             = 'data/autoencoder.x_train.v2.npy'
AUTOENC_Y_TRAIN             = 'data/autoencoder.y_train.v2.npy'
AUTOENC_X_TEST              = 'data/autoencoder.x_test.v2.npy'
AUTOENC_Y_TEST              = 'data/autoencoder.y_test.v2.npy'
AUTOENC_X_ALL              = 'data/autoencoder.x_all.v2.npy'
AUTOENC_Y_ALL              = 'data/autoencoder.y_all.v2.npy'


IFOREST_X_TRAIN             = 'data/iforest.x_train.v2.npy'
IFOREST_Y_TRAIN             = 'data/iforest.y_train.v2.npy'
IFOREST_X_TEST              = 'data/iforest.x_test.v2.npy'
IFOREST_Y_TEST              = 'data/iforest.y_test.v2.npy'

MODEL_RF_TUNED_BEST         = 'models/random_forest_best_20200305.model'
MODEL_IFOREST_TUNED_BEST    = 'models/iforest_best.model'

RANDOM_SEED                 = 202002

RANDOM_SEED_RANDOM_FOREST   = 2694033904862417032

