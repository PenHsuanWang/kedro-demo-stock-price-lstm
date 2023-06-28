"""
This is a boilerplate pipeline 'data_prepare'
generated using Kedro 0.18.10
"""


import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch


def extracting_training_data(input_data: pd.DataFrame, params) -> np.ndarray:
    """
    extracting from the csv data and convert the target column to numpy array
    """
    return input_data[params['target_column']].values


def scaling_by_column(input_series: np.ndarray) -> pd.DataFrame:
    """
    scaling the input numpy array
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(input_series.reshape(-1, 1)), scaler


def splitting(input_series: np.ndarray, params) -> tuple:
    """
    splitting the input series into training and testing series
    """
    training_data_size = int(len(input_series) * params['training_data_ratio'])
    training_series = input_series[:training_data_size]
    testing_series = input_series[training_data_size:]
    return training_series, testing_series


def sliding_window_masking(input_series: np.ndarray, params) -> tuple:
    """
    sliding window masking the input series
    """
    input_data_size = len(input_series) - params['window_size']
    feeding_model_data = np.zeros((input_data_size, params['window_size'], 1))
    target = np.zeros((input_data_size, 1))
    for i in range(input_data_size):
        feeding_model_data[i] = input_series[i:i + params['window_size']]
        target[i] = input_series[i + params['window_size']]
    return feeding_model_data, target


def convert_data_to_pytorch_tensor(input_data: np.ndarray, input_target: np.ndarray) -> tuple:
    """
    converting the data to pytorch training tensor
    """
    data = torch.from_numpy(input_data).float()
    target = torch.from_numpy(input_target).float()
    return data, target


