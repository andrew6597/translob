import pandas as pd
import torch
from translob import TransLOB
import torch.nn as nn
from torch.utils import data
from preprocessing import get_mid_price, Dataset, batch_gd


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Import the data
    df = pd.read_csv('LOBseries_100ms.csv')
    # Keep 10,000 rows at first
    df = df.iloc[:50000]
    print('Loaded df', df.shape)

    mid_prices = []
    for row in range(len(df)):
        mid_prices.append(get_mid_price(df,row))
    df['mid_price'] = mid_prices
    print('Added mid prices')
    point = len(df)*2//3
    df_train = df.iloc[:point]
    df_test = df.iloc[point:]

    # Hyperparameters
    batch_size = 8
    epochs = 100
    T = 100  # horizon
    lr = 0.001
    num_classes = 3
    dim = 40
    n = 3

    train_dataset = Dataset(df_train.to_numpy(),dim, T)
    test_dataset = Dataset(df_test.to_numpy(), dim, T)
    print('Created Dataset Classes')
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print('Created Dataset Loaders')
    print('Starting Training...')
    # Load the model with the default parameters
    model = TransLOB(conv_n_layers=1,tf_num_layers=1)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    train_losses, val_losses = batch_gd(model, criterion, optimizer,
                                        train_loader, test_loader, epochs, device= device)



