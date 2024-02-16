import tensorflow as tf
import numpy as np

def z_score(df):
    means = df.expanding().mean().shift()  # Calculate expanding mean and shift by one
    sums = df.expanding().sum().shift()
    stds = (sums - means)/np.array(df.index)
    normalized_df = (df.iloc[1:,:] - means.iloc[1:,:]) / stds.iloc[1:,:]
    return normalized_df.reset_index(drop = True, inplace = True)

def get_mid_price(x):
   return (x[0] + x[2])/ 2.0

def generate_labels(current_mid_price, future_mid_prices, alpha = 0.00001):

    mean_future_mid_prices = tf.reduce_mean(future_mid_prices)

    if current_mid_price >= 0:
        condition_up = tf.greater(mean_future_mid_prices, current_mid_price + alpha * current_mid_price)
        condition_down = tf.less(mean_future_mid_prices, current_mid_price - alpha * current_mid_price)
    else:
        condition_up = tf.greater(mean_future_mid_prices, current_mid_price - alpha * current_mid_price)
        condition_down = tf.less(mean_future_mid_prices, current_mid_price + alpha * current_mid_price)

    return tf.where(condition_up, 2, tf.where(condition_down, 0, 1))


def generate_stationary_labels(current_mid_price,future_mid_prices, alpha = 0.0001):
    cumprods = tf.math.cumprod(1 + future_mid_prices)
    pct_mean_cumprods = tf.reduce_mean(cumprods)
    condition_up = tf.greater(pct_mean_cumprods,1 + alpha)
    condition_down = tf.less(pct_mean_cumprods, 1 -alpha)
    return tf.where(condition_up, 2, tf.where(condition_down, 0, 1))

def generate_labels_no_scale(current_mid_price, future_mid_prices):
    alpha = 0.0001
    current_mid_price = current_mid_price
    future_mid = future_mid_prices[-1]
    condition_up = tf.greater(future_mid, current_mid_price + alpha * current_mid_price)
    condition_down = tf.less(future_mid, current_mid_price - alpha * current_mid_price)

    return tf.where(condition_up, 2, tf.where(condition_down, 0, 1))



def make_window_dataset(ds, labels, window_size=100, shift=1, horizon=600):
    # No split the data to time windows and get rid off the last window size and horizon +2(The +2 is for indexing purposes)
    ds_windows = ds.window(window_size, shift=shift, stride=1, drop_remainder=True)
    # ds_windows = ds_windows.take(ds_windows.cardinality().numpy() - (window_size - 1) - (horizon - 1))
    label_windows = labels.window(window_size, shift=shift, stride=1, drop_remainder=True)

    def sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True)

    # First flatten the dataset to 1 dataset, then map it back with batches of size window_size. Now it is iterable as numpy
    ds_windows = ds_windows.flat_map(sub_to_batch)

    label_windows = label_windows.flat_map(sub_to_batch)

    # Now we want to keep only the label from the last lob of every window
    labels = label_windows.map(lambda x: x[-1])

    combined_dataset = tf.data.Dataset.zip((ds_windows, labels))
    return combined_dataset


def train_data_pipe(tf_dataset, window_size, batch_size, k, length):
    mid_prices = tf_dataset.map(get_mid_price, num_parallel_calls=tf.data.AUTOTUNE)
    # Create future windows with size the wanted predicted horizon k
    future_mid_prices = mid_prices.window(size=k, shift=1, drop_remainder=True)
    # Flatmap them and make them numpy iterable
    future_mid_prices = future_mid_prices.flat_map(lambda x: x.batch(k))

    mid_prices = mid_prices.take(len(mid_prices) - k + 1)
    # Now we can create the tensorflow dataset type that consists the label for every timestamp
    tf_labels = tf.data.Dataset.zip((mid_prices, future_mid_prices)).map(
        lambda current, future: (generate_stationary_labels(current, future)), num_parallel_calls=tf.data.AUTOTUNE)

    ds = make_window_dataset(tf_dataset, tf_labels, window_size=window_size, shift=1, horizon=k)

    
    
    neutral = 0
    up = 0
    down = 0
    for _, label in ds:
        if label.numpy() == 1:
            neutral += 1
        elif label.numpy() ==2:
            up += 1
        else:
            down += 1
    up_per = up/(up+down+neutral)
    down_per = down/(up+down+neutral)
    neutral_per = neutral/(up+down+neutral)
    print('Stats of Training Dataset:')
    print(f'{k/10} seconds horizon:')
    print('-------------------')
    print(f'Price went up {up_per*100} %')
    print(f'Price went down {down_per*100} %')
    print(f'Price stayed neutral {neutral_per*100} %')
    proportions = [down_per, neutral_per,up_per]

    ds = ds.shuffle(int((length - window_size - k - 2)/7))

    ds = ds.batch(batch_size=batch_size)

    
    #ds = ds.cache()
    
    #ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds,proportions  #[0.20901199822930707, 0.59070512838963076, 0.2002828733810622]


def test_data_pipe(tf_dataset, window_size, batch_size, k):
    mid_prices = tf_dataset.map(get_mid_price, num_parallel_calls=tf.data.AUTOTUNE)
    # Create future windows with size the wanted predicted horizon k
    future_mid_prices = mid_prices.window(size=k, shift=1, drop_remainder=True)
    # This should be length: len(mid_prices) - (k  - 1)
    # Flatmap them and make them numpy iterable iterable
    future_mid_prices = future_mid_prices.flat_map(lambda x: x.batch(k))

    # Now we can create the tensorflow dataset type that consists the label for every timestamp
    tf_labels = tf.data.Dataset.zip((mid_prices, future_mid_prices)).map(
        lambda current, future: (generate_labels(current, future)), num_parallel_calls=tf.data.AUTOTUNE)

    ds = make_window_dataset(tf_dataset, tf_labels, window_size=window_size, shift=1, horizon=k)
    ds = ds.batch(batch_size=batch_size)
    #ds = ds.cache()

    return ds
