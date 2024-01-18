""" Prediction module for the ExNet model. """

import numpy as np
import pandas as pd

from .exnet import ExNet

# Build params
SEED = 0
N_EXPERTS = 5
SPEC_WEIGHT = 7.7e-4
ENTROPY_WEIGHT = 4.2e-2
EXPERT_ARCHITECTURE = [32, 32]
EMBEDDING_SIZE = 32
DROPOUT_RATES = {"input": 0.1, "hidden": 0.5}
WEIGHT_DECAY = {"l1": 0.0, "l2": 0.0}
GAMMA = 2.5

# Fit params
N_EPOCHS = 400
PATIENCE = 20
BATCH_SIZE = 1024
LEARNING_RATE = 7.8e-4
OPTIMIZER = "nadam"
LOOKAHEAD = True

# ===== Preparing data =====
data = pd.read_csv("data/new_dataset.csv")
active_investors = data["company_short_name"].unique()
investor_mapping = dict(zip(active_investors, range(len(active_investors))))
reverse_mapping = {v: k for k, v in investor_mapping.items()}

data["investor_encoding"] = data["company_short_name"].apply(
    lambda x: investor_mapping[x]
)

n_investors = np.unique(data.investor_encoding.values).shape[0]

features = list(data.columns[4:-1])


# ===== Loading model =====
model = ExNet(
    n_feats=len(features),
    output_dim=3,
    n_experts=N_EXPERTS,
    expert_architecture=EXPERT_ARCHITECTURE,
    n_investors=n_investors,
    embedding_size=EMBEDDING_SIZE,
    dropout_rates=DROPOUT_RATES,
    weight_decay={"l1": 0.0, "l2": 0.0},
    spec_weight=SPEC_WEIGHT,
    entropy_weight=ENTROPY_WEIGHT,
    gamma=GAMMA,
    name="exnet",
)

model.fake_call()
model.load_weights("models/exnet_big.h5")


# ===== Prediction on given data =====
def predict(isin, b_side, n_clients=5, size=None):
    """
    Predicts the top clients recommendation for a given ISIN and buyer/seller side.

    Parameters:
    - isin (str): The ISIN of the security.
    - b_side (str): The side of the client (Buyer or Seller).
    - n_clients (int): The number of top clients to recommend (default: 5).
    - size (int): The size of the security (default: None).

    Returns:
    - top_clients (list): The list of top clients recommended.
    - probabilities (ndarray): The probabilities associated with the top clients.
    - viz_df (DataFrame): The visualization dataframe containing relevant data.

    """
    # Get features
    infos = data.loc[data["ISIN"] == isin].iloc[-1]
    # Change size info
    infos.loc["Size"] = (int(size) * 1e6 - 51715757) / 247139467
    feats = infos[features].values.astype(np.float32)

    # Create testing iteration
    clients = np.arange(87, dtype="int32")
    feats_copied = np.tile(feats, (87, 1))
    to_pred = (feats_copied, clients)
    predictions = model.predict(to_pred)

    # Getting the top clients recommendation
    if b_side == "Buyer":
        top_indices = np.argsort(predictions[:, 1])[-n_clients:][::-1]
        probabilities = np.round(predictions[top_indices, 1], 4)
        viz_df = data[(data["ISIN"] == isin) & (data["Signal"] == 1)][-10:]
    if b_side == "Seller":
        top_indices = np.argsort(predictions[:, 2])[-n_clients:][::-1]
        probabilities = np.round(predictions[top_indices, 2], 4)
        viz_df = data[(data["ISIN"] == isin) & (data["Signal"] == 2)][-10:]

    top_clients = [reverse_mapping[index] for index in top_indices]
    return top_clients, probabilities, viz_df
