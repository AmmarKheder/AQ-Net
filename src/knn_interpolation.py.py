#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.
"""

import torch
import numpy as np
import pandas as pd
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def index_points(pts, idx):
    batch_size, num_points, fdim = pts.shape
    sample_num, knn_num = idx.shape[1], idx.shape[2]
    idx_expanded = idx.unsqueeze(-1).expand(batch_size, sample_num, knn_num, fdim)
    new_points = torch.gather(pts.unsqueeze(1).expand(batch_size, sample_num, num_points, fdim), 2, idx_expanded)
    return rearrange(new_points, 'b m k c -> b c m k')

def get_knn_pts(k, pts, center_pts, return_idx=False):
    num_points = pts.shape[1]
    k = min(k, num_points)
    dists = torch.cdist(center_pts, pts)
    knn_idx = dists.topk(k, largest=False, dim=-1).indices
    knn_pts = index_points(pts, knn_idx)
    return (knn_pts, knn_idx) if return_idx else knn_pts

def interpolate_feature(k, original_pts, query_pts, original_feat):
    knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
    repeat_query_pts = repeat(query_pts, 'b n c -> b c n k', k=k)
    dist = torch.norm(knn_pts - repeat_query_pts, p=2, dim=1)
    dist_recip = 1.0 / (dist + 1e-8)
    weights = dist_recip / dist_recip.sum(dim=-1, keepdim=True)
    knn_feat = index_points(original_feat, knn_idx)
    interpolated_feat = torch.sum(knn_feat * weights.unsqueeze(1), dim=-1)
    return interpolated_feat

def knn_prediction(data, test_loader, model, scaler_target, seq_length, prediction_horizon):
    model.eval()
    all_predictions = []
    all_station_ids = []
    with torch.no_grad():
        for X_batch, y_batch, batch_dates, batch_station_ids, _ in test_loader:
            X_batch = X_batch.to(model.device)
            outputs, _ = model(X_batch)
            outputs = outputs.squeeze()
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            outputs_np = outputs.cpu().detach().numpy()
            all_predictions.append(outputs_np)
            all_station_ids.extend(batch_station_ids)
    predictions_norm = np.concatenate(all_predictions, axis=0)
    num_samples, horizon = predictions_norm.shape
    predictions_denorm = scaler_target.inverse_transform(predictions_norm.reshape(-1, 1))
    predictions_denorm = predictions_denorm.reshape(num_samples, horizon)
    df_preds = pd.DataFrame(predictions_denorm)
    df_preds['station_id'] = all_station_ids
    df_agg = df_preds.groupby('station_id').mean().reset_index()
    data_visible = data[data['visibility'] == 'visible'].copy()
    data_visible.loc[:, 'station_id'] = data_visible.groupby(['longitude', 'latitude', 'z']).ngroup()
    visible_grouped = data_visible.groupby(['longitude', 'latitude', 'z']).first().reset_index()
    visible_grouped = visible_grouped.sort_values('station_id').reset_index(drop=True)
    visible_grouped['station_id'] = visible_grouped['station_id'].astype(int)
    df_agg['station_id'] = df_agg['station_id'].astype(int)
    visible_grouped = visible_grouped.merge(df_agg, on='station_id', how='left')
    prediction_cols = list(range(prediction_horizon))
    visible_grouped['PM2.5_pred'] = visible_grouped[prediction_cols].apply(lambda row: row.tolist(), axis=1)
    visible_grouped = visible_grouped.drop(columns=prediction_cols)
    visible_grouped['visibility'] = 'visible'
    hidden_data = data[data['visibility'] == 'hidden'].copy()
    hidden_grouped = hidden_data.sort_values(by='date').groupby(['longitude', 'latitude', 'z'])['PM2.5 (μg/m3)'].apply(lambda s: s.values[:prediction_horizon] if len(s) >= prediction_horizon else np.nan).reset_index()
    coord_columns = ['longitude', 'latitude', 'z']
    visible_coords = torch.tensor(visible_grouped[coord_columns].values, dtype=torch.float32).unsqueeze(0)
    visible_preds = torch.tensor(np.stack(visible_grouped['PM2.5_pred'].values), dtype=torch.float32).unsqueeze(0)
    hidden_coords = torch.tensor(hidden_grouped[coord_columns].values, dtype=torch.float32).unsqueeze(0)
    k = min(20, visible_coords.shape[1])
    predicted_hidden_features = interpolate_feature(k, visible_coords, hidden_coords, visible_preds)
    predicted_hidden_features = predicted_hidden_features.squeeze(0).transpose(0, 1).cpu().numpy()
    hidden_grouped["PM2.5_pred"] = list(predicted_hidden_features)
    hidden_grouped['visibility'] = 'hidden'
    all_predictions_df = pd.concat([visible_grouped, hidden_grouped], ignore_index=True)
    return all_predictions_df