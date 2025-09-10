# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SAVE_PATH

def train_and_evaluate(model, data, site_names, save_path=SAVE_PATH):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.MSELoss()

    data = data.to(device)
    best_loss = float('inf')

    losses = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.8f}')

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))

    model.eval()
    with torch.no_grad():
        pred = model(data).cpu().numpy()
        y_true = data.y.cpu().numpy()
        train_mask_np = data.train_mask.cpu().numpy()

        mse_list = {}
        for i, site in enumerate(site_names):
            mse = mean_squared_error([y_true[i]], [pred[i]])
            mse_list[site] = mse

        print("Per-site MSE:", mse_list)

        mae_list = {}
        for i, site in enumerate(site_names):
            mae = mean_absolute_error([y_true[i]], [pred[i]])
            mae_list[site] = mae

        print("Per-site MAE:", mae_list)

        mape_list = {}
        for i, site in enumerate(site_names):
            mape = mean_absolute_percentage_error([y_true[i]], [pred[i]])
            mape_list[site] = mape

        print("Per-site MAPE:", mape_list)

        mse = mean_squared_error(y_true[~train_mask_np], pred[~train_mask_np])
        mae = mean_absolute_error(y_true[~train_mask_np], pred[~train_mask_np])
        mape = mean_absolute_percentage_error(y_true[~train_mask_np], pred[~train_mask_np])

        print(f"\nTest MSE: {mse:.8f}, MAE: {mae:.8f}, MAPE: {mape:.8f}")

    return model, losses