# eval/evaluate.py
import torch
import numpy as np


import logging

logger = logging.getLogger()

def ndcg_at_k(predictions, ground_truth, k=10):
    """Compute NDCG at K"""
    k = min(k, len(predictions))

    # Sort indices based on predictions
    sorted_indices = np.argsort(-predictions)[:k]
    sorted_ground_truth = ground_truth[sorted_indices]

    # Compute DCG
    dcg = 0.0
    for i in range(k):
        dcg += (2 ** sorted_ground_truth[i] - 1) / np.log2(i + 2)

    # Compute IDCG
    ideal_sorted = np.sort(ground_truth)[::-1][:k]
    idcg = 0.0
    for i in range(k):
        idcg += (2 ** ideal_sorted[i] - 1) / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    mae = 0.0
    predictions_list = []
    ground_truth_list = []

    with torch.no_grad():
        for batch_idx, (user_indices, item_indices, ratings) in enumerate(test_loader):
            predictions = model(user_indices, item_indices)
            test_loss += criterion(predictions, ratings).item()
            mae += torch.abs(predictions - ratings).sum().item()

            predictions_list.extend(predictions.cpu().numpy())
            ground_truth_list.extend(ratings.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    avg_mae = mae / len(test_loader.dataset)

    # Compute NDCG
    predictions_list = np.array(predictions_list)
    ground_truth_list = np.array(ground_truth_list)
    avg_ndcg = ndcg_at_k(predictions_list, ground_truth_list, k=10)

    logger.info(f"Average NDCG@10: {avg_ndcg:.4f}")
    logger.info(f"Test Loss (MSE): {avg_test_loss:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {avg_mae:.4f}")

    return avg_test_loss, avg_mae, avg_ndcg
