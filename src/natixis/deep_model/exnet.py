""" Implementation of the ExNet model. """

import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import umap
from sklearn.manifold import TSNE

from ..utils import logger


class Gating(keras.Model):
    """
    Gating model for investor-expert interaction.

    Args:
        n_investors (int): Number of investors.
        n_experts (int): Number of experts.
        embedding_size (int): Size of the embedding vector.
        name (str, optional): Name of the model. Defaults
        to "gating".

    Attributes:
        n_investors (int): Number of investors.
        n_experts (int): Number of experts.
        embedding_size (int): Size of the embedding vector.
        embedding (keras.layers.Embedding): Embedding
        layer for investor input.
        experts_mapping (keras.layers.Dense): Dense
        layer for mapping embeddings to expert logits.

    Methods:
        call(gating_input): Forward pass of the gating model.

    Returns:
        tf.Tensor: Softmax probabilities of expert logits.
    """

    def __init__(self, n_investors, n_experts, embedding_size, name="gating"):
        super().__init__(name=name)
        self.n_investors = n_investors
        self.n_experts = n_experts
        self.embedding_size = embedding_size

        self.embedding = keras.layers.Embedding(
            input_dim=n_investors,
            output_dim=embedding_size,
            embeddings_initializer="normal",
            name="embedding",
        )
        self.experts_mapping = keras.layers.Dense(
            units=n_experts,
            use_bias=False,
            activation=None,
            kernel_initializer="normal",
            name="experts_mapping",
        )

    @tf.function(experimental_relax_shapes=True)
    def call(self, gating_input):
        """
        Applies the deep model to the input data.

        Args:
            gating_input: The input data to be processed.

        Returns:
            The softmax output of the deep model.
        """
        embedding = self.embedding(gating_input)
        embedding = tf.reshape(embedding, (tf.shape(embedding)[0], self.embedding_size))

        gating = self.experts_mapping(embedding)
        return tf.nn.softmax(logits=gating)

    def get_config(self):
        """
        Returns the configuration of the ExNet model.

        Returns:
            dict: The configuration dictionary containing the values of `n_investors`, `n_experts`,
                    `embedding_size`, `embedding`, and `experts_mapping`.
        """
        config = super().get_config().copy()
        config.update(
            {
                "n_investors": self.n_investors,
                "n_experts": self.n_experts,
                "embedding_size": self.embedding_size,
                "embedding": self.embedding,
                "experts_mapping": self.experts_mapping,
            }
        )
        return config


class Expert(keras.Model):
    """
    Expert model for the Natixis challenge.

    Args:
        output_dim (int): The number of output dimensions.
        expert_architecture (list): List of integers representing
        the architecture of the expert model.
        dropout_rates (dict): Dictionary specifying the dropout rates
          for input and hidden layers. Defaults to {"input": 0.0, "hidden": 0.0}.
        weight_decay (dict): Dictionary specifying the L1 and L2
        weight decay values. Defaults to {"l1": 0.0, "l2": 0.0}.
        expert_idx (int): Index of the expert model. Defaults to -1.
        name (str): Name of the expert model. Defaults to "expert".

    Attributes:
        expert_idx (int): Index of the expert model.
        architecture (list): List of integers representing the
        architecture of the expert model.
        dropout_rates (dict): Dictionary specifying the dropout
        rates for input and hidden layers.
        weight_decay (dict): Dictionary specifying the L1 and L2
        weight decay values.
        input_bn (keras.layers.BatchNormalization): Batch
        normalization layer for input.
        input_drop (keras.layers.Dropout): Dropout layer for input.
        exp{expert_idx}_l{layer_idx} (keras.layers.Dense): Dense
        layer for each hidden layer.
        exp{expert_idx}_bn{layer_idx} (keras.layers.BatchNormalization):
        Batch normalization layer for each hidden layer.
        exp{expert_idx}_drop{layer_idx} (keras.layers.Dropout):
        Dropout layer for each hidden layer.
        out (keras.layers.Dense): Dense layer for output.

    """

    def __init__(
        self,
        output_dim,
        expert_architecture,
        dropout_rates={"input": 0.0, "hidden": 0.0},
        weight_decay={"l1": 0.0, "l2": 0.0},
        expert_idx=-1,
        name="expert",
    ):
        super().__init__(name=name)
        self.expert_idx = expert_idx
        self.architecture = expert_architecture
        self.dropout_rates = dropout_rates
        self.weight_decay = weight_decay

        self.input_bn = keras.layers.BatchNormalization(
            name="exp{0}_input_bn".format(expert_idx)
        )
        self.input_drop = keras.layers.Dropout(
            rate=dropout_rates["input"], name="exp{0}_input_drop".format(expert_idx)
        )

        for layer_idx, layer_size in enumerate(self.architecture):
            l = keras.layers.Dense(
                units=layer_size,
                activation=None,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l1_l2(**weight_decay),
                name=f"exp{0}_l{1}".format(expert_idx, layer_idx),
            )
            bn = keras.layers.BatchNormalization(
                name=f"exp{0}_bn{1}".format(expert_idx, layer_idx)
            )
            drop = keras.layers.Dropout(
                rate=dropout_rates["hidden"],
                name=f"exp{0}_drop{1}".format(expert_idx, layer_idx),
            )
            setattr(self, f"exp{0}_l{1}".format(expert_idx, layer_idx), l)
            setattr(self, f"exp{0}_bn{1}".format(expert_idx, layer_idx), bn)
            setattr(self, f"exp{0}_drop{1}".format(expert_idx, layer_idx), drop)

        self.out = keras.layers.Dense(
            units=output_dim,
            activation=None,
            kernel_regularizer=keras.regularizers.l1_l2(**weight_decay),
            name=f"exp{0}_output".format(expert_idx),
        )

    @tf.function(experimental_relax_shapes=True)
    def call(self, expert_input, training=False):
        """
        Forward pass of the ExNet model.

        Args:
            expert_input: Input tensor to the model.
            training: Boolean flag indicating whether the model is in training mode or not.

        Returns:
            out: Output tensor of the model after applying softmax activation.
        """
        x = self.input_bn(expert_input, training=training)
        x = self.input_drop(x, training=training)

        for layer_idx, _ in enumerate(self.architecture):
            l = getattr(self, f"exp{0}_l{1}".format(self.expert_idx, layer_idx))
            bn = getattr(self, f"exp{0}_bn{1}".format(self.expert_idx, layer_idx))
            drop = getattr(self, f"exp{0}_drop{1}".format(self.expert_idx, layer_idx))
            x = l(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
            x = drop(x, training=training)

        out = self.out(x)
        out = tf.nn.softmax(out)
        return out

    def fake_call(self):
        """
        This method generates a fake input and calls the model with it.
        """
        fake_input = np.random.randn(1, 1).astype(np.float32)
        self(fake_input)
        pass


class ExNet(keras.Model):
    """
    Initialize the ExNet model.

    Args:
        n_feats (int): Number of input features.
        output_dim (int): Dimension of the output.
        n_experts (int): Number of experts in the model.
        expert_architecture (list): List of integers representing
        the architecture of each expert.
        n_investors (int): Number of investors in the gating block.
        embedding_size (int): Size of the embedding for the gating block.
        dropout_rates (dict, optional): Dictionary specifying the dropout
        rates for input and hidden layers. Defaults to {"input": 0.0, "hidden": 0.0}.
        weight_decay (dict, optional): Dictionary specifying the L1 and
        L2 weight decay values. Defaults to {"l1": 0.0, "l2": 0.0}.
        spec_weight (float, optional): Weight for the specific loss.
        Defaults to 0.1.
        entropy_weight (float, optional): Weight for the entropy loss.
        Defaults to 0.1.
        gamma (float, optional): Gamma value for the gating block.
        Defaults to 2.0.
        name (str, optional): Name of the model. Defaults to "ExNet_tf2".
    """

    def __init__(
        self,
        n_feats,
        output_dim,
        n_experts,
        expert_architecture,
        n_investors,
        embedding_size,
        dropout_rates={"input": 0.0, "hidden": 0.0},
        weight_decay={"l1": 0.0, "l2": 0.0},
        spec_weight=0.1,
        entropy_weight=0.1,
        gamma=2.0,
        name="ExNet_tf2",
    ):
        super().__init__(name=name)
        self.model_name = name
        self.n_feats = n_feats
        self.n_experts = n_experts
        self.output_dim = output_dim
        self.embedding_size = embedding_size
        self.spec_weight = spec_weight
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.training = (
            False  # Flag used for correct usage of BatchNorm at serving time.
        )
        self.epsilon = 1e-6
        self.opt = None

        self.gating = Gating(
            n_investors=n_investors,
            n_experts=n_experts,
            embedding_size=embedding_size,
            name="gating_block",
        )

        for expert_idx in range(self.n_experts):
            exp = Expert(
                output_dim=output_dim,
                expert_architecture=expert_architecture,
                dropout_rates=dropout_rates,
                weight_decay=weight_decay,
                expert_idx=expert_idx,
                name=f"exp{0}".format(expert_idx),
            )
            setattr(self, f"exp{0}".format(expert_idx), exp)

    def fake_call(self):
        """This method generates fake input data and calls the model.

        Returns:
            None
        """
        # To be able to save & load easily, we require a 'build' function that sets all
        # tensors.
        fake_expert = np.random.randn(1, self.n_feats).astype(np.float32)
        fake_gating = np.array([0])
        self(fake_expert, fake_gating)

    @tf.function(experimental_relax_shapes=True)
    def call(self, expert_input, gating_input):
        """
        Forward pass of the ExNet model.

        Args:
            expert_input: Input to the expert network.
            gating_input: Input to the gating network.

        Returns:
            output: Output of the ExNet model.
            exp_out: Output of each expert network.
            gat_out: Output of the gating network.
        """
        gat_out = self.gating(gating_input)
        gating = tf.transpose(gat_out)
        gating = tf.tile(tf.expand_dims(gating, axis=-1), [1, 1, self.output_dim])

        exp_out = []
        for expert_idx in range(self.n_experts):
            exp = getattr(self, "exp{0}".format(expert_idx))
            expert_output = exp(expert_input, training=self.training)
            expert_output = tf.expand_dims(expert_output, axis=0)
            exp_out.append(expert_output)

        exp_out = tf.concat(exp_out, axis=0)
        output = tf.multiply(gating, exp_out)
        output = tf.reduce_sum(output, axis=0)

        return output, exp_out, gat_out

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, expert_input, gating_input, y_true):
        """
        Performs a single training step for the ExNet model.

        Args:
            expert_input: The input data for the expert network.
            gating_input: The input data for the gating network.
            y_true: The true labels for the input data.

        Returns:
            A tuple containing the loss values for the output, specialization, and entropy,
            respectively.
        """
        logger.info("Starting training")
        with tf.GradientTape() as tape:
            y_pred, exp_out, gat_out = self(expert_input, gating_input)
            output_loss = self.focal_loss(y_true, y_pred)
            if self.n_experts > 1:
                spec_loss = self.specialization_loss(exp_out)
                entropy_loss = self.entropy_loss(gat_out)
            else:
                spec_loss = tf.constant(0.0, dtype=tf.float32)
                entropy_loss = tf.constant(0.0, dtype=tf.float32)
            regularization = sum(self.losses)
            loss = (
                output_loss
                + self.spec_weight * spec_loss
                + self.entropy_weight * entropy_loss
                + regularization
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))
        return float(loss), float(output_loss), float(spec_loss), float(entropy_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, expert_input, gating_input, y_true):
        """
        Perform a single testing step of the model.

        Args:
            expert_input (tf.Tensor): Input to the expert network.
            gating_input (tf.Tensor): Input to the gating network.
            y_true (tf.Tensor): True labels.

        Returns:
            Tuple[float, float, float, float]: A tuple containing the loss, output loss, specialization loss, and entropy loss.
        """
        y_pred, exp_out, gat_out = self(expert_input, gating_input)
        output_loss = self.focal_loss(y_true, y_pred)
        if self.n_experts > 1:
            spec_loss = self.specialization_loss(exp_out)
            entropy_loss = self.entropy_loss(gat_out)
        else:
            spec_loss = tf.constant(0.0, dtype=tf.float32)
            entropy_loss = tf.constant(0.0, dtype=tf.float32)
        regularization = sum(self.losses)
        loss = (
            output_loss
            + self.spec_weight * spec_loss
            + self.entropy_weight * entropy_loss
            + regularization
        )
        return float(loss), float(output_loss), float(spec_loss), float(entropy_loss)

    @tf.function(experimental_relax_shapes=True)
    def specialization_loss(self, experts_tensor):
        """
        Calculates the specialization loss for the deep model.

        Args:
            experts_tensor (tf.Tensor): Tensor containing the expert predictions.

        Returns:
            tf.Tensor: The specialization loss.

        """
        # Cross-attribution weights of each expert.
        probas = self.gating(tf.range(self.gating.n_investors))
        mask = tf.cast(probas > 0, dtype=tf.float32)
        expert_count = tf.reduce_sum(mask, axis=0)

        mean_attrib = tf.math.divide_no_nan(tf.reduce_sum(probas, axis=0), expert_count)
        cross_attrib = tf.matmul(
            tf.expand_dims(mean_attrib, axis=1), tf.expand_dims(mean_attrib, axis=0)
        )
        weights = tf.linalg.set_diag(cross_attrib, tf.zeros([self.n_experts]))
        weights = weights / tf.reduce_sum(weights)

        # Batch-wise cross-experts correlation.
        mean = tf.expand_dims(tf.reduce_mean(experts_tensor, axis=1), axis=1)
        mean = tf.tile(mean, [1, tf.shape(experts_tensor)[1], 1])
        centered = experts_tensor - mean

        rev_std = tf.transpose(
            tf.math.rsqrt(tf.reduce_sum(centered**2, axis=1) + self.epsilon)
        )
        rev_std = tf.matmul(
            tf.expand_dims(rev_std, axis=2), tf.expand_dims(rev_std, axis=1)
        )

        centered = tf.transpose(centered, perm=[2, 1, 0])
        covar = tf.matmul(tf.transpose(centered, perm=[0, 2, 1]), centered)
        correls = covar * rev_std

        loss = tf.reduce_sum(tf.multiply(weights, correls)) / self.output_dim
        loss = (1 + loss) / 2  # Remapping loss to [0, 1].
        return loss

    @tf.function(experimental_relax_shapes=True)
    def entropy_loss(self, gating_tensor):
        """
        Calculates the entropy loss for the given gating tensor.

        Parameters:
            gating_tensor (tf.Tensor): The gating tensor.

        Returns:
            tf.Tensor: The entropy loss.
        """
        return -tf.reduce_mean(
            tf.multiply(gating_tensor, tf.math.log(gating_tensor + self.epsilon))
        )

    @tf.function(experimental_relax_shapes=True)
    def focal_loss(self, y_true, y_pred):
        """
        Compute the focal loss between the true labels and predicted probabilities.

        Parameters:
            y_true (tensor): True labels.
            y_pred (tensor): Predicted probabilities.

        Returns:
            tensor: Focal loss value.
        """

        adaptive_weights = tf.math.pow(1 - y_pred + self.epsilon, self.gamma)
        return -tf.reduce_mean(
            tf.reduce_sum(
                adaptive_weights * y_true * tf.math.log(y_pred + self.epsilon), axis=1
            )
        )

    def fit(
        self,
        train_data,
        val_data,
        n_epochs=10,
        batch_size=32,
        optimizer="nadam",
        learning_rate=1e-3,
        patience=10,
        lookahead=True,
        save_path="",
        seed=0,
    ):
        """
        Trains the model on the given training data and evaluates it on the validation data.

        Args:
            train_data: The training data, a tuple of (x_expert, x_gating, y).
            val_data: The validation data, a tuple of (x_expert, x_gating, y).
            n_epochs: The number of epochs to train the model (default: 10).
            batch_size: The batch size for training (default: 32).
            optimizer: The optimizer to use for training (default: "nadam").
            learning_rate: The learning rate for the optimizer (default: 1e-3).
            patience: The number of epochs to wait for improvement in validation loss before early stopping (default: 10).
            lookahead: Whether to use Lookahead optimizer (default: True).
            save_path: The path to save the trained model weights (default: "").
            seed: The random seed for reproducibility (default: 0).

        Returns:
            history: A pandas DataFrame containing the training history.
        """
        # For reproducibility of results.
        tf.random.set_seed(seed)
        np.random.seed(seed)

        if self.opt is None:
            if optimizer == "adam":
                self.opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer == "nadam":
                self.opt = keras.optimizers.Nadam(learning_rate=learning_rate)
            elif optimizer == "rmsprop":
                self.opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
            elif optimizer == "radam":
                self.opt = keras.optimizers.RectifiedAdam(learning_rate=learning_rate)
            elif optimizer == "sgd":
                self.opt = keras.optimizers.SGD(
                    learning_rate=learning_rate, nesterov=True, momentum=0.9
                )
            else:
                raise ValueError(
                    "Optimizer not recognized. Try in {adam, nadam, radam, rmsprop}."
                )

            if lookahead:
                self.opt = tfa.optimizers.Lookahead(
                    self.opt, sync_period=5, slow_step_size=0.5
                )

        else:
            self.opt.learning_rate = learning_rate

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=8192)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
        val_dataset = val_dataset.batch(val_data[0].shape[0])
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        best_val_loss = np.inf
        counter = 0

        recorded_metrics = []
        history = {k: [] for k in recorded_metrics}

        for epoch in range(n_epochs):
            epoch_duration = time.time()

            # ===== Training =====
            self.training = True

            train_loss = 0.0
            train_output_loss = 0.0
            train_spec_loss = 0.0
            train_entropy_loss = 0.0
            train_n_batches = 0
            for step, (x_expert, x_gating, y) in enumerate(train_dataset):
                batch_losses = self.train_step(x_expert, x_gating, y)
                train_loss += batch_losses[0]
                train_output_loss += batch_losses[1]
                train_spec_loss += batch_losses[2]
                train_entropy_loss += batch_losses[3]
                train_n_batches += 1

            train_loss /= train_n_batches
            train_output_loss /= train_n_batches
            train_spec_loss /= train_n_batches
            train_entropy_loss /= train_n_batches

            # ===== Validation =====
            self.training = False

            val_loss = 0.0
            val_output_loss = 0.0
            val_spec_loss = 0.0
            val_entropy_loss = 0.0
            val_n_batches = 0
            for step, (x_expert, x_gating, y) in enumerate(val_dataset):
                batch_losses = self.test_step(x_expert, x_gating, y)
                val_loss += batch_losses[0]
                val_output_loss += batch_losses[1]
                val_spec_loss += batch_losses[2]
                val_entropy_loss += batch_losses[3]
                val_n_batches += 1

            val_loss /= val_n_batches
            val_output_loss /= val_n_batches
            val_spec_loss /= val_n_batches
            val_entropy_loss /= val_n_batches

            for metric in recorded_metrics:
                history[metric].append(eval(metric).numpy())
            epoch_duration = time.time() - epoch_duration

            print(
                f"Epoch {0}/{1} | {2:.2f}s | (train) loss: {3:.5f} - out loss: {4:.5f} -"
                f" spec loss: {5:.5f} - entropy loss: {6:.5f}".format(
                    epoch + 1,
                    n_epochs,
                    epoch_duration,
                    train_loss,
                    train_output_loss,
                    train_spec_loss,
                    train_entropy_loss,
                )
            )
            print(
                f"(val) loss: {0:.5f} - out loss: {1:.5f} - spec loss: {2:.5f} - entropy"
                f" loss: {3:.5f}".format(
                    val_loss, val_output_loss, val_spec_loss, val_entropy_loss
                )
            )

            # ===== Early stopping =====
            if val_loss < best_val_loss:
                print(
                    f"Best val loss beaten, from {0:.5f} to {1:.5f}. Saving model.\n".format(
                        best_val_loss, val_loss
                    )
                )
                best_val_loss = val_loss
                counter = 0
                self.save_weights(f"{save_path}{self.model_name}.h5")
            else:
                counter += 1
                if counter == patience:
                    print(
                        f"{0} epochs performed without improvement. Stopping training.\n".format(
                            patience
                        )
                    )
                    break
                else:
                    print(
                        f"{0}/{1} epochs performed without improvement. Best val loss:"
                        f" {2:.5f}\n".format(counter, patience, best_val_loss)
                    )

        # Loading best weights.
        self.load_weights(f"{save_path}{self.model_name}.h5")
        history = pd.DataFrame.from_dict(history)
        history["epoch"] = range(len(history))
        return history

    def predict(self, pred_data, batch_size=-1):
        """
        Predicts the output for the given input data.

        Args:
            pred_data (numpy.ndarray): The input data for prediction.
            batch_size (int, optional): The batch size for prediction. Defaults to -1.

        Returns:
            numpy.ndarray: The predicted output.
        """

        self.training = False
        pred_dataset = tf.data.Dataset.from_tensor_slices(pred_data)

        if batch_size == -1:
            pred_dataset = pred_dataset.batch(pred_data[0].shape[0])
        else:
            pred_dataset = pred_dataset.batch(batch_size)
            pred_dataset = pred_dataset.prefetch(buffer_size=batch_size)

        for step, (x_expert, x_gating) in enumerate(pred_dataset):
            pred, _, _ = self(x_expert, x_gating)
            if step == 0:
                preds = pred.numpy()
            else:
                preds = np.concatenate((preds, pred.numpy()))

        return preds

    def get_experts_repartition(self, print_stats=True):
        """
        Calculates the repartition of experts based on their attribution probabilities.

        Args:
            print_stats (bool): Whether to print the statistics or not. Default is True.

        Returns:
            tuple: A tuple containing the attribution probabilities and the class frequency.

        """

        probas = self.gating(tf.range(self.gating.n_investors)).numpy()
        dominant_experts = np.argmax(probas, axis=1)
        values, counts = np.unique(dominant_experts, return_counts=True)

        count_per_expert = dict(zip(range(self.n_experts), [0] * self.n_experts))
        for idx, expert in enumerate(values):
            count_per_expert[expert] = counts[idx]

        class_frequency = dict(
            zip(values, np.round(100 * counts / len(dominant_experts), decimals=2))
        )

        if print_stats:
            print(
                f"\nExpert frequency (rounded - may not exactly sum to 100%): {0}".format(
                    class_frequency
                )
            )

            print("Mean expert attribution probability:")
            mean_probas = np.mean(probas, axis=0)
            for i in range(self.n_experts):
                # Only show 'relevant' experts.
                if mean_probas[i] > 0.01:
                    print(
                        f"Expert {0} - mean proba {1:.2f}, n_allocated {2:d},"
                        f" mean_allocated: {3:.2f}".format(
                            i,
                            100 * mean_probas[i],
                            count_per_expert[i],
                            100 * np.mean(probas[dominant_experts == i, i])
                            if i in dominant_experts
                            else 0,
                        )
                    )

        return probas, class_frequency

    def plot_experts_repartition(self):
        """
        Plots the distribution of experts for all investors.

        Returns:
            reorder_probas (pd.DataFrame): DataFrame containing the reordered probabilities of expert attribution.
            count_per_expert (pd.DataFrame): DataFrame containing the count of experts for each expert index.
            unattributed_experts (list): List of expert indices that are not attributed to any investor.
        """
        probas = self.gating(tf.range(self.gating.n_investors)).numpy()
        classes = np.argmax(probas, axis=1)
        val, counts = np.unique(classes, return_counts=True)
        unattributed_experts = list(set(range(self.n_experts)) - set(val))

        count_per_expert = pd.DataFrame(
            {
                "expert": list(val) + unattributed_experts,
                "count": list(counts) + [0] * len(unattributed_experts),
            }
        )
        count_per_expert = count_per_expert.sort_values(by="count", ascending=False)

        reorder_dict = dict(
            zip(
                [f"proba{i}" for i in range(self.n_experts)],
                [
                    probas[:, count_per_expert.expert.iloc[i]]
                    for i in range(self.n_experts)
                ],
            )
        )
        reorder_dict["investor_idx"] = np.arange(self.gating.n_investors)

        reorder_probas = pd.DataFrame(reorder_dict)
        reorder_probas = reorder_probas.sort_values(
            by=[f"proba{i}" for i in range(self.n_experts)], ascending=False
        )
        probas = probas[reorder_probas.investor_idx.values, :]

        cmap = plt.get_cmap("jet")
        idx = np.linspace(0, 1, self.n_experts)
        bottom = np.zeros(shape=(self.gating.n_investors,))

        for i in count_per_expert.expert.values:
            plt.bar(
                np.arange(self.gating.n_investors),
                probas[:, i],
                width=1,
                bottom=bottom,
                color=cmap(idx[i]),
            )
            bottom += probas[:, i]

        plt.xlim([0, self.gating.n_investors])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Investors")
        plt.ylabel("Percentage of expert attribution")
        plt.title("Experts distribution for all investors")
        plt.show()

        return reorder_probas, count_per_expert, unattributed_experts

    def plot_experts_umap(self, n_neighbors=10, min_dist=0.1):
        """
        Plots the UMAP visualization of the experts' embeddings.

        Parameters:
        - n_neighbors (int): Number of neighbors to consider for each point in the UMAP algorithm. Default is 10.
        - min_dist (float): The effective minimum distance between embedded points. Default is 0.1.
        """

        probas = self.gating(tf.range(self.gating.n_investors)).numpy()
        projector = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        proj_probas = projector.fit_transform(probas)

        dominant_experts = np.argmax(probas, axis=1)
        cmap = plt.get_cmap("jet")
        idx = np.linspace(0, 1, self.n_experts)

        plt.scatter(
            proj_probas[:, 0],
            proj_probas[:, 1],
            color=[cmap(idx[d]) for d in dominant_experts],
        )
        plt.title("UMAP Visualization of Investors Embeddings")
        plt.show()

    def plot_experts_tsne(
        self, perplexity=5, learning_rate=200.0, early_exaggeration=12.0, n_iter=1000
    ):
        """
        Plot t-SNE visualization of investor embeddings.

        Parameters:
        - perplexity: The perplexity parameter of t-SNE (default: 5).
        - learning_rate: The learning rate parameter of t-SNE (default: 200.0).
        - early_exaggeration: The early exaggeration parameter of t-SNE (default: 12.0).
        - n_iter: The number of iterations for t-SNE (default: 1000).
        """
        probas = self.gating(tf.range(self.gating.n_investors)).numpy()
        projector = TSNE(
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter,
        )
        proj_probas = projector.fit_transform(probas)

        dominant_experts = np.argmax(probas, axis=1)
        cmap = plt.get_cmap("jet")
        idx = np.linspace(0, 1, self.n_experts)

        plt.scatter(
            proj_probas[:, 0],
            proj_probas[:, 1],
            color=[cmap(idx[d]) for d in dominant_experts],
        )
        plt.title("t-SNE Visualization of Investors Embeddings")
        plt.show()
