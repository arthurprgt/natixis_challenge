from sklearn.metrics import average_precision_score
import tensorflow as tf
import natixis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# if len(physical_devices):
#     # The experiment will only take necessary memory on the GPU.
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

seed = 0
asset = "TEF"

# Build params
n_experts = 4
spec_weight = 7.7e-4
entropy_weight = 4.2e-2
expert_architecture = [32, 32]
embedding_size = 32
dropout_rates = {"input": 0.1, "hidden": 0.5}
weight_decay = {"l1": 0.0, "l2": 0.0}
gamma = 2.5

# Fit params
n_epochs = 1
patience = 1
batch_size = 1024
learning_rate = 7.8e-4
optimizer = "nadam"
lookahead = True

# ===== Preparing data =====
data = pd.read_csv("data/encoded_data.csv")
### TO REMOVE
data = data.iloc[:10000]
###
active_investors = data["company_short_name"].unique()
investor_mapping = dict(zip(active_investors, range(len(active_investors))))

data["investor_encoding"] = data["company_short_name"].apply(
    lambda x: investor_mapping[x]
)

n_investors = np.unique(data.investor_encoding.values).shape[0]

# Splitting data
indexes = np.arange(data.shape[0])
train_idx, test_idx = train_test_split(indexes, test_size=0.2, random_state=seed)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

# train_idx = data["train_idx"].values
# val_idx = data["val_idx"].values
# test_idx = data["test_idx"].values

features = ["B_Price"] + list(data.columns[6:-1])
# Removing irrelevant columns - date, encoding, target & splits.
data = data[["Deal_Date", "investor_encoding", "buyer"] + features]

train_data = data.iloc[train_idx]
val_data = data.iloc[val_idx]
test_data = data.iloc[test_idx]

train_data_ = (
    train_data[features].values.astype(np.float32),
    train_data['investor_encoding'].values.astype(np.int32),
    pd.get_dummies(train_data.buyer).values.astype(np.float32),
)

val_data_ = (
    val_data[features].values.astype(np.float32),
    val_data['investor_encoding'].values.astype(np.int32),
    pd.get_dummies(val_data.buyer).values.astype(np.float32),
)

test_data_ = (
    test_data[features].values.astype(np.float32),
    test_data['investor_encoding'].values.astype(np.int32),
    pd.get_dummies(test_data.buyer).values.astype(np.float32),
)

# ===== Training model =====
print(
    f"Training on {train_data_[0].shape[0]} samples, validating on {val_data_[0].shape[0]} samples."
)
model = natixis.deep_model.ExNet(
    n_feats=len(features),
    output_dim=2,
    n_experts=n_experts,
    expert_architecture=expert_architecture,
    n_investors=n_investors,
    embedding_size=embedding_size,
    dropout_rates=dropout_rates,
    weight_decay={"l1": 0.0, "l2": 0.0},
    spec_weight=spec_weight,
    entropy_weight=entropy_weight,
    gamma=gamma,
    name=f"ExNet_{asset}",
)
model.fit(
    train_data=train_data_,
    val_data=val_data_,
    n_epochs=n_epochs,
    batch_size=batch_size,
    optimizer="nadam",
    learning_rate=learning_rate,
    lookahead=True,
    patience=patience,
    seed=seed,
    save_path="models/",
)

# ===== Results =====
train_pred = model.predict(train_data_[0:2])
val_pred = model.predict(val_data_[0:2])
test_pred = model.predict(test_data_[0:2])

train_score = average_precision_score(train_data.buyer.values, train_pred[:, 1])
val_score = average_precision_score(val_data.buyer.values, val_pred[:, 1])
test_score = average_precision_score(test_data.buyer.values, test_pred[:, 1])
print(
    f"train ap: {100*train_score:.2f} - val ap: {100*val_score:.2f} - test ap: {100*test_score:.2f}"
)

_ = model.get_experts_repartition(print_stats=True)
model.plot_experts_repartition()
model.plot_experts_umap(n_neighbors=50, min_dist=0.1)
