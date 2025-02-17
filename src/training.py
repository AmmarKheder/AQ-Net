#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.
"""

import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    scaler_grad = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _, _, _ in train_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predictions, attn_weights = model(X_batch)
                predictions = predictions.squeeze()
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
                loss = criterion(predictions, y_batch) + 0.01 * torch.mean(entropy)
            scaler_grad.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val, _, _, _ in val_loader:
                X_val, y_val = X_val.to(model.device), y_val.to(model.device)
                predictions, _ = model(X_val)
                predictions = predictions.squeeze()
                loss = criterion(predictions, y_val)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_lstm_attention_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, criterion, scaler_target):
    model.load_state_dict(torch.load("best_lstm_attention_model.pth"))
    model.eval()
    test_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_test, y_test, _, _, _ in test_loader:
            X_test, y_test = X_test.to(model.device), y_test.to(model.device)
            predictions, _ = model(X_test)
            predictions = predictions.squeeze()
            loss = criterion(predictions, y_test)
            test_loss += loss.item()
            y_true.extend(y_test.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    test_loss /= len(test_loader)
    import numpy as np
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_denorm = scaler_target.inverse_transform(y_true)
    y_pred_denorm = scaler_target.inverse_transform(y_pred)
    return test_loss, y_true_denorm, y_pred_denorm