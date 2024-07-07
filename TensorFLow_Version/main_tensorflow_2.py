import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from data_prep_tf import train_data_pipe, test_data_pipe, z_score
from model_builider_tf import TransLOB
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

if __name__ == '__main__':
    print('Started running...')

    # Check if GPU is available
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    # Set GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('GPU memory growth set to True')
    else:
        print("No GPU found")

    # Load and split df
    df = pd.read_csv('/content/drive/My Drive/LOBseries_100ms.csv').drop(columns=['Unnamed: 0'])
    print('Loaded df', df.shape)

    # Set params
    # Prompt the user for input and store it in a variable
    n_dim = float(input("Please enter n_dim: "))
    print("You entered the float:", n_dim)
    
    k = int(input("Please enter k: "))
    print("You entered the int:", k)
    
    a = float(input("Please enter a: "))
    print("You entered the float:", a)
    
    df['mid_price'] = (df['p1_a'] + df['p1_b']) / 2.0
    df['label'] = 1
    df['future_price'] = df['mid_price'].shift(-k)
    df.loc[df['future_price'] > df['mid_price'] * (1+a), 'label'] = 2
    df.loc[df['future_price'] < df['mid_price'] * (1-a), 'label'] = 0
    print('done labeling')
    print(df['label'].value_counts())
