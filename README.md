# GMF_NN
The GMF-NN model is an enhancement of Generalized Matrix Factorization (GMF), integrating deep neural networks to improve its predictive capabilities for recommendation systems. GMF is a fundamental approach in collaborative filtering, where user and item interactions are modeled using element-wise multiplicative embedding representations.

![Alt Text](diagram.png)


to run the project follow these steps:

1. pip install -r requirements.txt
2. python main.py --dataset data/ratings_train.csv -e 100 -ne 100 -lr 0.001

 MAE and NDCG are used as the evaluation metrics and the results will be saved in dataset_results/results.json

 Thank you.

