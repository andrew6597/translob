import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils import data


def get_mid_price(df,timestamp):
    bid_price = df.loc[timestamp,'p1_b']
    ask_price = df.loc[timestamp, 'p1_a']
    return (bid_price+ask_price) / 2


def get_label(df, timestamp):
    a = 0.002
    p_t = df[timestamp-1, -1]
    p_t_next = df[timestamp, -1]
    rk_t = (p_t_next - p_t) / p_t

    if rk_t > a:
        return 1
    elif rk_t < -a:
        return 2  # It is actually the -1 class
    else:
        return 0


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, dim, time_window):
        """Initialization"""
        self.a = 0.002
        self.dim = dim
        self.time_window = time_window
        self.data = torch.from_numpy(data)
        self.n_samples = data.shape[0]

    def __len__(self):
        return self.n_samples - self.time_window - 1

    def __getitem__(self, i):
        # Keep only 20 depth -> 40 columns for input
        mean_values = torch.mean(self.data[:i + self.time_window, :40], dim=0)
        std_values = torch.std(self.data[:i + self.time_window, :40], dim=0) + 1e-8
        x = self.data[i: i + self.time_window, :40]

        current_mid = self.data[i+self.time_window, -1]
        target_mid = self.data[i+self.time_window+1, -1]
        if target_mid - current_mid > self.a:
            y = 2 # Higher
        elif target_mid - current_mid < -self.a:
            y = 0 # Lower
        else:
            y = 1 # Neutral
        normalized_x = (x - mean_values) / std_values
        permuted_x = normalized_x.permute(1,0)
        return permuted_x, y


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs, device):
    torch.autograd.set_detect_anomaly(True)
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = 0
        n_train = 0
        for inputs, targets in train_loader:

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)
        train_loss = train_loss/n_train  # a little misleading

        model.eval()
        test_loss = 0
        n_test = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            n_test += inputs.size(0)
        test_loss = test_loss/n_test

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, 'best_model_transformer')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses