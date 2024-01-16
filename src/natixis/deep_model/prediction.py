import natixis
import pandas as pd
import numpy as np
from .exnet import ExNet

# Build params
seed = 0
n_experts = 5
spec_weight = 7.7e-4
entropy_weight = 4.2e-2
expert_architecture = [32, 32]
embedding_size = 32
dropout_rates = {"input": 0.1, "hidden": 0.5}
weight_decay = {"l1": 0.0, "l2": 0.0}
gamma = 2.5

# Fit params
n_epochs = 400
patience = 20
batch_size = 1024
learning_rate = 7.8e-4
optimizer = "nadam"
lookahead = True

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
    n_experts=n_experts,
    expert_architecture=expert_architecture,
    n_investors=n_investors,
    embedding_size=embedding_size,
    dropout_rates=dropout_rates,
    weight_decay={"l1": 0.0, "l2": 0.0},
    spec_weight=spec_weight,
    entropy_weight=entropy_weight,
    gamma=gamma,
    name=f"exnet",
)

model.fake_call()
model.load_weights('models/exnet.h5')

# ===== Prediction on given data =====
def predict(isin, b_side, n_clients=5):
    # Get features
    infos = data.loc[data['ISIN'] == isin].iloc[-1]
    feats = (infos[features].values.astype(np.float32))

    # Create testing iteration
    clients = np.arange(89, dtype='int32')
    feats_copied = np.tile(feats, (89, 1))
    to_pred = (feats_copied, clients)
    predictions = model.predict(to_pred)

    # Getting the top clients recommendation
    if b_side=="Buyer":
        top_indices = np.argsort(predictions[:, 1])[-n_clients:][::-1]
        probabilities = np.round(predictions[top_indices, 1], 4)
    if b_side=="Seller":
        top_indices = np.argsort(predictions[:, 2])[-n_clients:][::-1]
        probabilities = np.round(predictions[top_indices, 2], 4)
    
    top_clients = [reverse_mapping[index] for index in top_indices]
    return top_clients, probabilities