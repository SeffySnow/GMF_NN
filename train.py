# train/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import GMF
from preprocessing import preprocess_data
import logging
import json
import os


logger = logging.getLogger()

class MFDataset(torch.utils.data.Dataset):
    def __init__(self, user_indices, item_indices, ratings):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.ratings = ratings

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        user_idx = torch.tensor(self.user_indices[idx], dtype=torch.long)
        item_idx = torch.tensor(self.item_indices[idx], dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float)
        return user_idx, item_idx, rating

def train_model(train_x, test_x, train_y, test_y, val_x, val_y, user_encoder, item_encoder,
                embedding_dim=10, num_epochs=50, learning_rate=0.001):
    # Create DataLoader for training, validation, and test sets
    train_loader = DataLoader(MFDataset(train_x['user_id'].values, train_x['item_id'].values, train_y.values), batch_size=64, shuffle=True)
    test_loader = DataLoader(MFDataset(test_x['user_id'].values, test_x['item_id'].values, test_y.values), batch_size=64, shuffle=False)
    val_loader = DataLoader(MFDataset(val_x['user_id'].values, val_x['item_id'].values, val_y.values), batch_size=64, shuffle=False)

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    model = GMF(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (user_indices, item_indices, ratings) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = model(user_indices, item_indices)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            logger.info(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}')
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch_idx, (user_indices, item_indices, ratings) in enumerate(val_loader):
                    predictions = model(user_indices, item_indices)
                    val_loss += criterion(predictions, ratings).item()
                val_loss /= len(val_loader)
                logger.info(f'Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss:.4f}')

    return model
