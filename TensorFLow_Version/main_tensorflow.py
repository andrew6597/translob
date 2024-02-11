import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from data_prep_tf import train_data_pipe, test_data_pipe
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

    # Set params
    n_dim = 40
    # first_train = 1495000
    # print(f'Dataset starts at time {first_train}')
    # last_train = first_train + 150000
    # We will keep 20,000 rows for validation and the next 20,000 for test
    # first_test = last_train + 20000
    # last_test = first_test + 20000

    # Load and split df
    df = pd.read_csv('/content/drive/My Drive/LOBseries_100ms.csv').drop(columns=['Unnamed: 0'])
    print('Loaded df', df.shape)

    # df_train = df.iloc[first_train:last_train, :n_dim]
    # df_val = df.iloc[last_train:first_test, :n_dim]
    # df_test = df.iloc[first_test: last_test, :n_dim]
    val_point = int(len(df) * 2 / 3)
    test_point = val_point + 150000

    df_train = df.iloc[:val_point, :n_dim]
    df_val = df.iloc[val_point:test_point, :n_dim]
    df_test = df.iloc[test_point:, :n_dim]

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
    batch_size = 16
    k = 100
    epochs = 1
    lr = 0.0001

    # Set the data pipeline for preprocessing before training
    ds_train, proportions = train_data_pipe(tf_dataset_train, window_size, batch_size, k, val_point)
    ds_val = test_data_pipe(tf_dataset_val, window_size, batch_size, k)
    ds_test = test_data_pipe(tf_dataset_test, window_size, batch_size, k)

    # Create and compile the model
    model = TransLOB(window_size, n_dim)

    train_class_weights = [(1 / proportions[0]), (1 / proportions[1]),(1 / proportions[2])]

    def custom_sparse_crossentropy_loss(y_true, y_pred):
        # Convert class weights to tensor
        weights = tf.constant(train_class_weights)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = loss_fn(y_true, y_pred, sample_weight=weights)
        return loss


    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            name="Adam",
        ),
        loss=custom_sparse_crossentropy_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Fit the model
    r = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_val)

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

    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)

    model.save('/content/drive/My Drive/my_model.h5')
