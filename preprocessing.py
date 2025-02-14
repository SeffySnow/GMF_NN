# preprocessing/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger()

def preprocess_data(data, random_seed=1337):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    
    all_users = data['user_id'].unique()
    all_items = data['item_id'].unique()
    user_encoder.fit(all_users)
    item_encoder.fit(all_items)

   
    data['user_id'] = user_encoder.transform(data['user_id'])
    data['item_id'] = item_encoder.transform(data['item_id'])

    data['rating'] = data['rating'].astype(int)
    scaler = MinMaxScaler()
    data['rating'] = scaler.fit_transform(data['rating'].values.reshape(-1, 1))

    train_x, test_x, train_y, test_y = train_test_split(
        data[['user_id', 'item_id']], data['rating'],
        stratify=data['rating'], test_size=0.2, random_state=random_seed
    )
    test_x, val_x, test_y, val_y = train_test_split(
        test_x, test_y, stratify=test_y, test_size=0.1, random_state=random_seed
    )

    
    logger.info(f"Number of users: {len(user_encoder.classes_)}")
    logger.info(f"Number of items: {len(item_encoder.classes_)}")
    logger.info(f"Rating distribution: {data['rating'].value_counts(normalize=True)}")

    return train_x, test_x, train_y, test_y, val_x, val_y, user_encoder, item_encoder
