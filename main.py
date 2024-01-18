"""Train the model and export it"""

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import natixis

SEED = 0

with open("config/config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Build params
N_EXPERTS = config["n_experts"]
SPEC_WEIGHT = config["spec_weight"]
ENTROPY_WEIGHT = config["entropy_weight"]
EXPERT_ARCHITECTURE = config["expert_architecture"]
EMBEDDING_SIZE = config["embedding_size"]
DROPOUT_RATES = config["dropout_rates"]
WEIGHT_DECAY = config["weight_decay"]
GAMMA = config["gamma"]

# Fit params
N_EPOCHS = config["n_epochs"]
PATIENCE = config["patience"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
OPTIMIZER = config["optimizer"]
LOOKAHEAD = config["lookahead"]

# ===== Preparing data =====
data = pd.read_csv("data/new_dataset.csv")
active_investors = data["company_short_name"].unique()
investor_mapping = dict(zip(active_investors, range(len(active_investors))))

data["investor_encoding"] = data["company_short_name"].apply(
    lambda x: investor_mapping[x]
)

n_investors = np.unique(data.investor_encoding.values).shape[0]

# Splitting data
indexes = np.arange(data.shape[0])
train_idx, test_idx = train_test_split(indexes, test_size=0.2, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=SEED)

features = list(data.columns[4:-1])
# Removing irrelevant columns - date, encoding, target & splits.
data = data[["Deal_Date", "investor_encoding", "Signal"] + features]

train_data = data.iloc[train_idx]
val_data = data.iloc[val_idx]
test_data = data.iloc[test_idx]

train_data_ = (
    train_data[features].values.astype(np.float32),
    train_data["investor_encoding"].values.astype(np.int32),
    pd.get_dummies(train_data.Signal).values.astype(np.float32),
)

val_data_ = (
    val_data[features].values.astype(np.float32),
    val_data["investor_encoding"].values.astype(np.int32),
    pd.get_dummies(val_data.Signal).values.astype(np.float32),
)

test_data_ = (
    test_data[features].values.astype(np.float32),
    test_data["investor_encoding"].values.astype(np.int32),
    pd.get_dummies(test_data.Signal).values.astype(np.float32),
)

# ===== Training model =====
print(
    f"Training on {train_data_[0].shape[0]} samples, validating on"
    f" {val_data_[0].shape[0]} samples."
)
model = natixis.deep_model.ExNet(
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
model.fit(
    train_data=train_data_,
    val_data=val_data_,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    optimizer="nadam",
    learning_rate=LEARNING_RATE,
    lookahead=False,
    patience=PATIENCE,
    seed=SEED,
    save_path="models/",
)

# ===== Results =====
train_pred = model.predict(train_data_[0:2])
val_pred = model.predict(val_data_[0:2])
test_pred = model.predict(test_data_[0:2])

train_score = average_precision_score(train_data.Signal.values, train_pred)
val_score = average_precision_score(val_data.Signal.values, val_pred)
test_score = average_precision_score(test_data.Signal.values, test_pred)
print(
    f"train ap: {100*train_score:.2f} - val ap: {100*val_score:.2f} - test ap:"
    f" {100*test_score:.2f}"
)

_ = model.get_experts_repartition(print_stats=True)
model.plot_experts_repartition()
model.plot_experts_umap(n_neighbors=10, min_dist=0.1)
