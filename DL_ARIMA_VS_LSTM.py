# TODO: to run the project run the following command in the cmd :
# pip install -U tensorflow statsmodels scipy pandas yfinance numpy tqdm matplotlib seaborn fredapi scikit-learn

# URL du papier : https://par.nsf.gov/servlets/purl/10186768#:~:text=The%20average%20reduction%20in%20error%20rates%20obtained%20by%20LSTM%20is,the%20number%20of%20training%20times.
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import pandas as pd
import yfinance as yf
from typing import Literal, Optional, Self, Tuple
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.api import ARIMA
from sklearn.metrics import mean_squared_error
from fredapi import Fred

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import optuna


# Fraction of the dataset used as training
TRAIN_TEST_PERCENT_SPLIT = 0.7

INDEX = "DJI-WEEKLY"  # TODO: choose the index to analyze

# Typedef for the index in the article
TICKER = Literal[
    "N225",
    "IXIC",
    "HSI",
    "GSPC",
    "DJI",
    "DJI-WEEKLY",
    "MC",
    "HO",
    "EX",
    "FB",
    "MS",
    "TR",
]
# DO NOT CHANGE IT IT'S THE TOKEN FOR THE FRED API
FRED_API_TOKEN = "3a77d9ea4d8848995628c7edcb822e3d"


############################################### DATA MANAGEMENT ###############################################
class DataFetcher:
    _instance: Optional[Self] = None
    __FINANCIAL_TICKER = ("N225", "IXIC", "HSI", "GSPC", "DJI", "DJI-WEEKLY")
    __ECONOMICS_TICKER = ("MC", "HO", "EX", "FB", "MS", "TR")
    __TICKER_TO_OFFICIAL_TICKER = {
        "MC": "CUSR0000SAM1",
        "HO": "CUUR0300SAH",
        "EX": "TWEXM",
        "FB": "CPIFABSL",
        "MS": "M1SL",
        "TR": "CPITRNSL",
        "N225": "^N225",
        "IXIC": "^IXIC",
        "HSI": "^HSI",
        "GSPC": "^GSPC",
        "DJI": "^DJI",
        "DJI-WEEKLY": "^DJI",
    }

    def __init__(self) -> None:
        self.__FRED_API = Fred(api_key=FRED_API_TOKEN)

    def fetch_data_historical_series(
        self,
        series_name: TICKER,
    ) -> pd.Series:
        """_summary_

        Args:
            series_name (Literal[ &quot;N225&quot;, &quot;IXIC&quot;, &quot;HSI&quot;, &quot;GSPC&quot;, &quot;DJI&quot;, &quot;DJI): The ticker to fetch

        Raises:
            ValueError: The provided ticker is not available.

        Returns:
            pd.Series: The pandas data observation series
        """
        # Transform the ticker provided into an official ticker used with yfinance API or FRED API
        OFFICIAL_TICKER = self.__TICKER_TO_OFFICIAL_TICKER.get(series_name)
        # Handle FRED API or yFinance API case switch
        if series_name in self.__ECONOMICS_TICKER:
            result_series = self.__FRED_API.get_series(OFFICIAL_TICKER)
        elif series_name in self.__FINANCIAL_TICKER:
            result_series = yf.download(
                OFFICIAL_TICKER,
                start="1985-01-01",
                end="2018-08-01",
                interval="1wk" if series_name == "DJI-WEEKLY" else "1m",
            )["Adj Close"]
        else:
            raise ValueError("Error, provide a valid ticker.")
        # Naming convention
        result_series.name = series_name
        return result_series

    def __new__(cls):
        """Method used to implement the singleton design pattern (unique instance of the class available)

        Returns:
            DataFetcher: The class itself
        """
        if cls._instance is None:
            cls._instance = super(DataFetcher, cls).__new__(cls)
        return cls._instance


# Define the object fetcher
DATA_FETCHER_OBJ = DataFetcher()
# Fetch the historical data for a given index
df_SERIES = DATA_FETCHER_OBJ.fetch_data_historical_series(INDEX).to_frame() #.to_numpy()
np_SERIES = df_SERIES.to_numpy()

# Split the train and the test dataframe
np_TRAIN = np_SERIES[: int(TRAIN_TEST_PERCENT_SPLIT * np_SERIES.shape[0])]
np_TEST = np_SERIES[int(TRAIN_TEST_PERCENT_SPLIT * np_SERIES.shape[0]) :]
############################################### FIN DATA MANAGEMENT ##############################################


############################################### ARIMA ###############################################
history = np_TRAIN.copy()
arima_predictions = []
# Forecast
for t in tqdm(range(np_TEST.shape[0]), desc="Training the rolling ARIMA...", leave=False):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    hat = model_fit.forecast(1)
    arima_predictions.append(hat)
    observed = np_TEST[t]
    history = np.append(history, observed)

ARIMA_MSE = mean_squared_error(np_TEST, arima_predictions)
ARIMA_RMSE = np.sqrt(ARIMA_MSE)


############################################### FIN ARIMA ###############################################

ARIMA_RMSE = np.sqrt(ARIMA_MSE)


############################################### FIN ARIMA ###############################################



############################################### LSTM ###############################################

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def prepare_X_y_dataset(
    dataset: npt.NDArray[np.float64], n_steps: int = 5
):
    """
    Prepare the X dataset and y dataset so it could be used by the neural network to train it.

    Args:
        dataset (npt.NDArray[np.float64]): The dataset e.g : TRAIN or TEST. It must be a 1D dataset.
        n_steps (int, optional): The number of steps used in the past to predict the future, here 5 as in the ARIMA. Defaults to 5.

    Returns:
        X_train, y_train, X_test, y_test as torch.tensor object with values as floats and scaler object for rescaling
    """
    # Create a deep copy of the dataset to avoid modifying the original data
    dataset = dc(dataset)

    # Create lagged features for the dataset, shifting the INDEX column by i steps
    for i in range(n_steps, 0, -1):
        dataset[f'{INDEX}(t-{i})'] = dataset[INDEX].shift(i)

    # Remove rows with missing values created by the shifting
    dataset.dropna(inplace=True)
    
    # Convert the dataset to a numpy array
    dataset = dataset.to_numpy()
    
    # Initialize the MinMaxScaler to scale the features to the range (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Fit the scaler to the data and transform the dataset
    dataset = scaler.fit_transform(dataset)
    
    # Separate the dataset into input features (X) and target values (y)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    
    # Split the data into training and testing sets based on a predefined split percentage
    X_train = X[:int(TRAIN_TEST_PERCENT_SPLIT * X.shape[0])]
    X_test = X[int(TRAIN_TEST_PERCENT_SPLIT * X.shape[0]):]

    y_train = y[:int(TRAIN_TEST_PERCENT_SPLIT * y.shape[0])]
    y_test = y[int(TRAIN_TEST_PERCENT_SPLIT * y.shape[0]):]

    # Reshape the input features for the neural network
    X_train = X_train.reshape((-1, n_steps, 1))
    X_test = X_test.reshape((-1, n_steps, 1))

    # Reshape the target values
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # Convert the numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    # Return the prepared training and testing datasets along with the scaler
    return X_train, y_train, X_test, y_test, scaler


# Define a custom dataset class for time series data
class TimeSeriesDataset(Dataset):
    """
    Custom dataset for time series data.

    Args:
        X (torch.tensor): The input features tensor.
        y (torch.tensor): The target values tensor.
    """
    def __init__(self, X, y):
        # Initialize the dataset with input features and target values
        self.X = X
        self.y = y

    def __len__(self):
        # Return the length of the dataset
        return len(self.X)

    def __getitem__(self, i):
        # Return the ith item from the dataset
        return self.X[i], self.y[i]

# Define the LSTM model class
class LSTM(nn.Module):
    """
    LSTM model for time series prediction.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
        num_stacked_layers (int): The number of stacked LSTM layers.
    """
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output of the model.
        """
        batch_size = x.size(0)
        # Initialize the hidden state and cell state
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        # Forward pass through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Pass the output through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out



def fit_and_forcast_LSTM(df_SERIES, batch_size, learning_rate, num_epochs):
    # Prepare the dataset for training and testing
    # Define the function to train the model for one epoch
    def train_one_epoch():
        """
        Train the LSTM model for one epoch.
        """
        model.train(True)
        # print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

        # Iterate over the batches of the training data
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            # Perform the forward pass
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print()

    # Define the function to validate the model for one epoch
    def validate_one_epoch():
        """
        Validate the LSTM model for one epoch.
        """
        model.train(False)
        running_loss = 0.0

        # Iterate over the batches of the validation data
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            # Perform the forward pass without gradient computation
            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        # Calculate the average loss across all batches
        avg_loss_across_batches = running_loss / len(test_loader)

    X_train, y_train, X_test, y_test, scaler = prepare_X_y_dataset(df_SERIES)

    # Create the training and testing datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the LSTM model
    model = LSTM(1, 5, 1)
    model.to(device)  # Move the model to the specified device (e.g., GPU or CPU)
    model

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model for the specified number of epochs
    for epoch in tqdm(range(num_epochs), desc="Training the rolling LSTM...", leave=False):
        train_one_epoch()   # Train the model for one epoch
        validate_one_epoch()  # Validate the model for one epoch

    # Make predictions on the training data
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

    # Flatten the predicted values for training data
    train_predictions = predicted.flatten()

    # Prepare dummies for inverse scaling
    dummies = np.zeros((X_train.shape[0], 6)) 
    dummies[:, 0] = train_predictions

    # Inverse scale the predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])

    # Prepare dummies for the actual training target values
    dummies = np.zeros((X_train.shape[0], 6))
    dummies[:, 0] = y_train.flatten()

    # Inverse scale the actual training target values
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])

    # Make predictions on the testing data
    test_predictions = model(X_test.to(device)).detach().numpy().flatten()

    # Prepare dummies for inverse scaling of test predictions
    dummies = np.zeros((X_test.shape[0], 6))
    dummies[:, 0] = test_predictions

    # Inverse scale the test predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])

    # Prepare dummies for the actual testing target values
    dummies = np.zeros((X_test.shape[0], 6))
    dummies[:, 0] = y_test.flatten()

    # Inverse scale the actual testing target values
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])

    # Calculate the mean squared error and root mean squared error for the test predictions
    LSTM_MSE = mean_squared_error(new_y_test, test_predictions)
    LSTM_RMSE = np.sqrt(LSTM_MSE)

    return LSTM_RMSE, new_y_test, test_predictions


def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)

    # Call the fit_and_forecast_LSTM function with the suggested hyperparameters
    LSTM_RMSE, _, _ = fit_and_forcast_LSTM(df_SERIES, batch_size, learning_rate, num_epochs)

    return LSTM_RMSE

# Create a study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
############################################### FIN LSTM ###############################################

LSTM_RMSE, new_y_test, test_predictions = fit_and_forcast_LSTM(df_SERIES,study.best_params['batch_size'],study.best_params['learning_rate'],study.best_params['num_epochs'])


print(f"Result for {INDEX}")
print(f"ARIMA RMSE: {ARIMA_RMSE}")
print(f"LSTM RMSE: {LSTM_RMSE}")
print(f"Reduction in RMSE: {100*(LSTM_RMSE-ARIMA_RMSE)/ARIMA_RMSE:.2f}%")
print("Positive -> less improvement...")
plt.plot(new_y_test, label="REAL DATA")
plt.plot(test_predictions, label="LSTM PREDICTION")
plt.plot(arima_predictions, label="ARIMA PREDICTION")
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Price")
plt.title(f"Price and price prediction for {INDEX} using ARIMA")
plt.grid()
plt.show()
