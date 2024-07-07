import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, roc_auc_score
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
    n_dim = int(input("Please enter n_dim: "))
    print("You entered the int:", n_dim)
    
    k = int(input("Please enter k: "))
    print("You entered the int:", k)
    
    window_size = int(input("Please enter window_size: "))
    print("You entered the int:", window_size)
    
    a = float(input("Please enter a: "))
    print("You entered the float:", a)
    
    df['mid_price'] = (df['p1_a'] + df['p1_b']) / 2.0
    df['label'] = 1
    df['future_price'] = df['mid_price'].shift(-k)
    df.loc[df['future_price'] > df['mid_price'] * (1+a), 'label'] = 2
    df.loc[df['future_price'] < df['mid_price'] * (1-a), 'label'] = 0
    print('done labeling')
    print(df['label'].value_counts())

    val_point = int(len(df) * 2 / 3)
    test_point = val_point + 150000
    
    df_train = df.iloc[:val_point, :n_dim]
    df_val = df.iloc[val_point:test_point, :n_dim]
    df_test = df.iloc[test_point:, :n_dim]

    
    # Scale the data (Not neccessary if we do it with pct change)
    
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_train_scaled = pd.DataFrame(scaler.transform(df_train))
    df_val_scaled = pd.DataFrame(scaler.transform(df_val))
    df_test_scaled = pd.DataFrame(scaler.transform(df_test))
    print('Done Scaling')

    ## Split into timeseries data every one dataset

    # Train Data
    X_train = []
    y_train = []
    for t in range(0,len(df) - window_size): 
        X_train.append(df_train_scaled.iloc[t:t+window_size, :n_dim])
        y_train.append(df_train.loc[t+window_size,'label'])

    N = len(X_train) #Number of total series sized T

    X_train = np.array(X_train).reshape(N,window_size,n_dim)
    y_train = np.array(y_train)
    print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)

    # Validation data
    X_val = []
    y_val = []
    for t in range(0, len(df_val) - window_size):
        X_val.append(df_val_scaled.iloc[t:t + window_size, :n_dim])
        y_val.append(df_val.loc[t + window_size, 'label'])
    
    N_val = len(X_val)  # Number of total series sized T for validation
    
    X_val = np.array(X_val).reshape(N_val, window_size, n_dim)
    y_val = np.array(y_val)
    print('X_val shape:', X_val.shape, 'y_val shape:', y_val.shape)
    
    # Test data
    X_test = []
    y_test = []
    for t in range(0, len(df_test) - window_size):
        X_test.append(df_test_scaled.iloc[t:t + window_size, :n_dim])
        y_test.append(df_test.loc[t + window_size, 'label'])
    
    N_test = len(X_test)  # Number of total series sized T for test
    
    X_test = np.array(X_test).reshape(N_test, window_size, n_dim)
    y_test = np.array(y_test)
    print('X_test shape:', X_test.shape, 'y_test shape:', y_test.shape)


    # Set params
    epochs = int(input("Please enter epochs: "))
    print("You entered the int:", epochs)
    
    lr = float(input("Please enter lr: "))
    print("You entered the float:", lr)

    # Create and compile the model
    model = TransLOB(window_size, n_dim) 
    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            name="Adam",
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Fit the model
    r = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Finally test the model on test data
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)

    roc_auc = roc_auc_score(true_classes, predictions[:, 1])
    print(f'ROC AUC Score: {roc_auc:.2f}')
    
    # Save the model
    model.save('/content/drive/My Drive/my_model.h5')
