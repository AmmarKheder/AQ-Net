#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data = data.sort_values(by=['date', 'longitude', 'latitude']).reset_index(drop=True)
    data['z'] = 1.0
    data['day_of_week'] = data['date'].dt.dayofweek
    pollutant_columns = ['CO (mg/m3)', 'NO2 (μg/m3)', 'O3 (μg/m3)', 'PM10 (μg/m3)', 'PM2.5 (μg/m3)', 'SO2 (μg/m3)']
    data = data.loc[(data[pollutant_columns] > 0).all(axis=1)].reset_index(drop=True)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data = data.sort_values(by=['date', 'longitude', 'latitude']).reset_index(drop=True)
    unique_coords = data[['longitude', 'latitude']].drop_duplicates().values
    distance_matrix = cdist(unique_coords, unique_coords, metric='euclidean')
    np.fill_diagonal(distance_matrix, np.inf)
    unique_stations = pd.DataFrame(unique_coords, columns=['longitude', 'latitude'])
    unique_stations['min_distance'] = np.nanmin(np.where(distance_matrix == np.inf, np.nan, distance_matrix), axis=1)
    unique_stations['mean_distance'] = np.nanmean(np.where(distance_matrix == np.inf, 0, distance_matrix), axis=1)
    unique_stations['max_distance'] = np.nanmax(np.where(distance_matrix == np.inf, np.nan, distance_matrix), axis=1)
    unique_stations['std_distance'] = np.nanstd(np.where(distance_matrix == np.inf, 0, distance_matrix), axis=1)
    data = pd.merge(data, unique_stations, on=['longitude', 'latitude'], how='left')
    return data

def split_and_normalize_data(data, features, target):
    station_ids = data.groupby(['longitude', 'latitude']).ngroups
    hidden_ratio = 0.1
    import numpy as np
    np.random.seed(42)
    hidden_ids = np.random.choice(range(station_ids), size=int(hidden_ratio * station_ids), replace=False)
    data['visibility'] = data.groupby(['longitude', 'latitude']).ngroup().apply(lambda x: 'hidden' if x in hidden_ids else 'visible')
    visible_data = data[data['visibility'] == 'visible'].copy()
    train_size = int(0.7 * len(visible_data))
    val_size = int(0.1 * len(visible_data))
    train_visible = visible_data.iloc[:train_size].copy()
    val_visible = visible_data.iloc[train_size:train_size + val_size].copy()
    test_visible = visible_data.iloc[train_size + val_size:].copy()
    train_visible[features + [target]] = train_visible[features + [target]].astype(float)
    val_visible[features + [target]] = val_visible[features + [target]].astype(float)
    test_visible[features + [target]] = test_visible[features + [target]].astype(float)
    from sklearn.preprocessing import MinMaxScaler
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    train_features_norm = scaler_features.fit_transform(train_visible[features].values)
    val_features_norm = scaler_features.transform(val_visible[features].values)
    test_features_norm = scaler_features.transform(test_visible[features].values)
    train_visible.loc[:, features] = train_features_norm
    val_visible.loc[:, features] = val_features_norm
    test_visible.loc[:, features] = test_features_norm
    all_target = np.concatenate([train_visible[target].values, val_visible[target].values, test_visible[target].values]).reshape(-1, 1)
    scaler_target.fit(all_target)
    train_visible.loc[:, target] = scaler_target.transform(train_visible[[target]].values)
    val_visible.loc[:, target] = scaler_target.transform(val_visible[[target]].values)
    test_visible.loc[:, target] = scaler_target.transform(test_visible[[target]].values)
    return train_visible, val_visible, test_visible, scaler_target
