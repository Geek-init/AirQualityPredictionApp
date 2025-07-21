import zipfile
import pandas as pd
from io import BytesIO
import seaborn as sns
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def extract_and_clean(zip_path: str, target_station: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, 'r') as main_zip:
        nested_zip_name = [n for n in main_zip.namelist() if n.endswith('.zip')][0]
        print("Nested archive:", nested_zip_name)

        nested_zip_bytes = main_zip.read(nested_zip_name)

        with zipfile.ZipFile(BytesIO(nested_zip_bytes)) as nested_zip:
            print("Files inside nested:", nested_zip.namelist())

            folder = "PRSA_Data_20130301-20170228"
            csv_name = f'PRSA_Data_{target_station}_20130301-20170228.csv'
            csv_path = f'{folder}/{csv_name}'

            if csv_path not in nested_zip.namelist():
                raise FileNotFoundError(f"{csv_path} not found in archive.")

            with nested_zip.open(csv_path) as f:
                data = pd.read_csv(f)

    data = data.dropna()
    data.to_csv('air_quality_cleaned.csv', index=False)
    print("Saved cleaned CSV: air_quality_cleaned.csv")
    return data


def draw_pm25_trend(data: pd.DataFrame):
    data['datetime'] = pd.to_datetime(data[['year', 'month', 'day']])
    daily_mean = data.groupby('datetime')['PM2.5'].mean()

    plt.figure(figsize=(12, 6))
    daily_mean.plot()
    plt.title('Daily Mean PM2.5')
    plt.xlabel('Date')
    plt.ylabel('PM2.5')
    plt.tight_layout()
    plt.savefig('eda_pm25_trend.pdf')
    plt.close()
    print("Saved: eda_pm25_trend.pdf")

def draw_correlation_matrix(data: pd.DataFrame):
    numeric = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.pdf')
    plt.close()
    print("Saved: eda_correlation_heatmap.pdf")

def draw_pm25_histogram(data: pd.DataFrame):
    data['datetime'] = pd.to_datetime(data[['year', 'month', 'day']])
    daily_mean = data.groupby('datetime')['PM2.5'].mean()

    plt.figure(figsize=(8, 6))
    plt.hist(daily_mean, bins=30)
    plt.title('Histogram of PM2.5')
    plt.xlabel('PM2.5')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('eda_pm25_histogram.pdf')
    plt.close()
    print("Saved: eda_pm25_histogram.pdf")
    


def build_data_loaders(data: pd.DataFrame, batch_size: int = 32):
    data = data.dropna()

    numeric_features = data.drop(columns=['PM2.5']).select_dtypes(include=['float64', 'int64'])
    labels = data['PM2.5']

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(numeric_features)

    with open('scaler.pkl', 'wb') as f_out:
        pickle.dump(scaler, f_out)

    X_tensor = torch.tensor(features_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)

    total = len(data)
    train_len = int(0.8 * total)
    val_len = int(0.1 * total)
    test_len = total - train_len - val_len

    X_train, X_val, X_test = torch.split(X_tensor, [train_len, val_len, test_len])
    y_train, y_val, y_test = torch.split(y_tensor, [train_len, val_len, test_len])

    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

import torch
import torch.nn as nn

class PollutionRegressor(nn.Module):
    def __init__(self, num_inputs: int):
        super(PollutionRegressor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features):
        return self.model(features)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def run_training(
    net: nn.Module,
    loader_train: torch.utils.data.DataLoader,
    loader_val: torch.utils.data.DataLoader,
    loader_test: torch.utils.data.DataLoader,
    epochs: int = 200,
    lr: float = 0.001
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs + 1):
        net.train()
        train_loss = 0.0
        for inputs, targets in loader_train:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(loader_train.dataset)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader_val:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = net(inputs)
                loss = loss_fn(predictions, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(loader_val.dataset)

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(net.state_dict(), 'model.pt')
    print("Saved weights: model.pt")

    net.eval()
    true_vals = []
    predicted_vals = []
    with torch.no_grad():
        for inputs, targets in loader_test:
            inputs = inputs.to(device)
            predictions = net(inputs)
            true_vals.extend(targets.cpu().numpy().flatten())
            predicted_vals.extend(predictions.cpu().numpy().flatten())

    plt.figure(figsize=(10, 5))
    plt.plot(true_vals, label='Actual')
    plt.plot(predicted_vals, label='Predicted')
    plt.legend()
    plt.title('PM2.5 Forecast')
    plt.tight_layout()
    plt.savefig('model_prediction.pdf')
    plt.close()
    print("Saved plot: model_prediction.pdf")



def run_training(
    net: nn.Module,
    loader_train: torch.utils.data.DataLoader,
    loader_val: torch.utils.data.DataLoader,
    loader_test: torch.utils.data.DataLoader,
    epochs: int = 200,
    lr: float = 0.001
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs + 1):
        net.train()
        train_loss = 0.0
        for inputs, targets in loader_train:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(loader_train.dataset)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader_val:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = net(inputs)
                loss = loss_fn(predictions, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(loader_val.dataset)

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(net.state_dict(), 'model.pt')
    print("Saved weights: model.pt")

    net.eval()
    true_vals = []
    predicted_vals = []
    with torch.no_grad():
        for inputs, targets in loader_test:
            inputs = inputs.to(device)
            predictions = net(inputs)
            true_vals.extend(targets.cpu().numpy().flatten())
            predicted_vals.extend(predictions.cpu().numpy().flatten())

    plt.figure(figsize=(10, 5))
    plt.plot(true_vals, label='Actual')
    plt.plot(predicted_vals, label='Predicted')
    plt.legend()
    plt.title('PM2.5 Forecast')
    plt.tight_layout()
    plt.savefig('model_prediction.pdf')
    plt.close()
    print("Saved plot: model_prediction.pdf")

