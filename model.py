# model/model.py
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout=0.2):
        super(GMF, self).__init__()

        # User and Item Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Fully Connected Layers
        self.f1 = nn.Linear(embedding_dim, 64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, 16)

        # Output Layer
        self.matrix = nn.Linear(16, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_indices, item_indices):
        # Get User and Item Embeddings
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)

        # Element-wise Multiplication (GMF)
        out = torch.mul(user_embedding, item_embedding)

        
        out = F.relu(self.f1(out))
        out = self.dropout(out)  
        out = F.relu(self.f2(out))
        out = self.dropout(out)
        out = F.relu(self.f3(out))
        out = self.dropout(out)

        # Output layer
        out = self.matrix(out)

        return torch.sigmoid(out).squeeze()  

# class GMF(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim):
#         super(GMF, self).__init__()

#         self.user_embedding = nn.Embedding(num_users, embedding_dim)
#         self.item_embedding = nn.Embedding(num_items, embedding_dim)
    
#         self.matrix = nn.Linear(embedding_dim, 1)


#     def forward(self, user_indices, item_indices):
#         user_embedding = self.user_embedding(user_indices)
#         item_embedding = self.item_embedding(item_indices)

#         out = torch.mul(user_embedding, item_embedding)
#         out = self.matrix(out)

#         return out.squeeze()
