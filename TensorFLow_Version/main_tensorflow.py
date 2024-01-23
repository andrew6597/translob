import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data_prep_tf import train_data_pipe,test_data_pipe
from model_builider_tf import TransLOB
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == '__main__':
    print('Started running...')
    # Set params
    n_dim = 40
    last_train = 50000
    first_test = last_train + 10000
    last_test = first_test + 10000

    # Load and split df
    df = pd.read_csv('LOBseries_100ms.csv').drop(columns=['Unnamed: 0'])
    print('Loaded df', df.shape)

    df_train = df.iloc[:last_train, :n_dim]
    df_val = df.iloc[last_train:first_test, :n_dim]
    df_test = df.iloc[first_test: last_test, :n_dim]

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_train_scaled = pd.DataFrame(scaler.transform(df_train))
    df_val_scaled = pd.DataFrame(scaler.transform(df_val))
    df_test_scaled = pd.DataFrame(scaler.transform(df_test))

    # Load the data as tensorflow datasets for efficiency
    tf_dataset_train = tf.data.Dataset.from_tensor_slices(df_train_scaled)
    tf_dataset_val = tf.data.Dataset.from_tensor_slices(df_val_scaled)
    tf_dataset_test = tf.data.Dataset.from_tensor_slices(df_test_scaled)

    # Set params
    window_size = 100
    batch_size = 32
    k = 300
    epochs = 3
    lr = 0.001

    # Set the data pipeline for preprocessing before training
    ds_train = train_data_pipe(tf_dataset_train, window_size, batch_size, k)
    ds_val = test_data_pipe(tf_dataset_val, window_size, batch_size, k)
    ds_test = test_data_pipe(tf_dataset_test, window_size, batch_size, k)

    # Create and compile the model
    model = TransLOB(window_size,n_dim)

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
    r = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_val)

    import matplotlib.pyplot as plt

    plt.plot(r.history['loss'], label ='loss')
    plt.plot(r.history['val_loss'], label = 'val_loss')
    plt.legend()
    plt.show()

    # Finally test the model on test data
    predictions = model.predict(ds_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = []

    for _, labels in ds_test:
        for label in labels:
            true_classes.append(label.numpy())
    true_classes = np.array(true_classes)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')