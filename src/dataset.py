#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class AirQualityDataset(Dataset):
    def __init__(self, data, features, target, seq_length=336, prediction_horizon=168):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.X, self.y, self.dates, self.station_ids, self.coordinates = self._create_sequences(data, features, target, seq_length, prediction_horizon)
    def _create_sequences(self, data, features, target, seq_length, prediction_horizon):
        X, y, dates, station_ids, coordinates = [], [], [], [], []
        grouped = data.groupby(['longitude', 'latitude'])
        for station_id, ((lon, lat), station_data) in enumerate(grouped):
            station_data = station_data.sort_values(by='date')
            values = station_data[features + [target]].values
            date_values = station_data['date'].values
            if len(values) > seq_length + prediction_horizon:
                for i in range(len(values) - seq_length - prediction_horizon):
                    start_date = pd.to_datetime(date_values[i + seq_length])
                    if start_date.dayofweek != 0:
                        continue
                    X.append(values[i:i+seq_length, :-1])
                    y.append(values[i+seq_length:i+seq_length+prediction_horizon, -1])
                    dates.append(str(start_date))
                    station_ids.append(station_id)
                    coordinates.append((lon, lat))
        return np.array(X), np.array(y), np.array(dates), np.array(station_ids), np.array(coordinates)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
                self.dates[idx],
                self.station_ids[idx],
                self.coordinates[idx])