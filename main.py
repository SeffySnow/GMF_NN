import logging
import argparse
import json
import os
from train import train_model , MFDataset
from evaluate import evaluate_model
from preprocessing import preprocess_data
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch import nn
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset (e.g., 'ratings_train.csv')")
parser.add_argument("-e", "--embedding_dim", type=int, default=20, help="Embedding dimension (Default: 10)")
parser.add_argument("-ne", "--num_epochs", type=int, default=50, help="Number of epochs (Default: 50)")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate (Default: 0.001)")
args = parser.parse_args()

# Load dataset
data = pd.read_csv(args.dataset)  # Load the dataset passed via the command line


# Preprocess data
train_x, test_x, train_y, test_y, val_x, val_y, user_encoder, item_encoder = preprocess_data(data)

# Create DataLoader for training, validation, and test sets
train_loader = DataLoader(MFDataset(train_x['user_id'].values, train_x['item_id'].values, train_y.values), batch_size=64, shuffle=True)
test_loader = DataLoader(MFDataset(test_x['user_id'].values, test_x['item_id'].values, test_y.values), batch_size=64, shuffle=False)
val_loader = DataLoader(MFDataset(val_x['user_id'].values, val_x['item_id'].values, val_y.values), batch_size=64, shuffle=False)

# Define the loss function (criterion)
criterion = nn.MSELoss()

# Train model
model = train_model(train_x, test_x, train_y, test_y, val_x, val_y, user_encoder, item_encoder,
                    embedding_dim=args.embedding_dim, num_epochs=args.num_epochs, learning_rate=args.learning_rate)

# Evaluate model (pass the criterion to the evaluate_model function)
avg_test_loss, avg_mae, avg_ndcg = evaluate_model(model, test_loader, criterion)

# Save results
import os
import json

# Directory for saving results
results_dir = "dataset_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# File path
results_file = f"{results_dir}/results.json"

# Load existing results if the file exists
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        try:
            existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = {}  # Handle empty or corrupted JSON
else:
    existing_results = {}

# Add new results
new_results = {
    "test_loss": avg_test_loss,
    "mae": avg_mae,
    "ndcg": avg_ndcg
}

# Append the results with a new entry (e.g., using an index or timestamp)
import time
timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp for tracking
existing_results[timestamp] = new_results

# Save back to JSON file
with open(results_file, "w") as f:
    json.dump(existing_results, f, indent=4)

logger.info("Model training and evaluation completed. Results saved successfully.")
