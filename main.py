---

### main.py

```python
#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.
"""

from src.data_preprocessing import load_and_preprocess_data, split_and_normalize_data
from src.dataset import AirQualityDataset
from src.model import LSTMAttentionModel
from src.training import train_model, evaluate_model
from src.knn_interpolation import knn_prediction

import torch
from torch.utils.data import DataLoader

def main():
    data_path = 'data/sorted_air_quality_data_with_regions.csv'
    data = load_and_preprocess_data(data_path)
    features = ['CO (mg/m3)', 'NO2 (μg/m3)', 'O3 (μg/m3)', 'PM10 (μg/m3)',
                'SO2 (μg/m3)', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_of_week_sin', 'day_of_week_cos', 'min_distance', 'mean_distance',
                'max_distance', 'std_distance']
    target = 'PM2.5 (μg/m3)'
    train_visible, val_visible, test_visible, scaler_target = split_and_normalize_data(data, features, target)
    seq_length = 336
    prediction_horizon = 168
    train_dataset = AirQualityDataset(train_visible, features, target, seq_length, prediction_horizon)
    val_dataset = AirQualityDataset(val_visible, features, target, seq_length, prediction_horizon)
    test_dataset = AirQualityDataset(test_visible, features, target, seq_length, prediction_horizon)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAttentionModel(input_dim=len(features), hidden_dim=128, attention_heads=2, dropout_rate=0.1)
    model.to(device)
    model.device = device  # assign device attribute for training functions
    import torch.optim as optim
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=50)
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10)
    test_loss, y_true_denorm, y_pred_denorm = evaluate_model(model, test_loader, criterion, scaler_target)
    print("Test Loss:", test_loss)
    all_predictions_df = knn_prediction(data, test_loader, model, scaler_target, seq_length, prediction_horizon)
    print(all_predictions_df.head())

if __name__ == "__main__":
    main()