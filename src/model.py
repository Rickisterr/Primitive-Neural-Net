"""
This module defines the entire network of nodes containing primitive
functions that trains to update connections and embeddings. To use,
import the `Network` class.

FOR NOW, ONLY SUPPORTS SINGLE OPERAND (UNARY) DSL FUNCTIONS
"""
import os
import pickle
import json
import torch
import torch.nn.functional as F
from src.embedding import MatrixPairEmbedding
from src.nodes import Node

class Network:
    """
    Neural Network model comprising of nodes consisting of functional primitives
    in which the nodes themselves as well as the connections between them are 
    responsible for the flow of data through them to produce an output most
    similar to the desired output from a given input.

    Args:
        example_input (tensor): Input matrix to help node form its own example to learn from.
        hidden_dim (int): Number of channels of embeddings. Default is 128.
        output_dim (int): Number of features per channel in embeddings. Default is 64.
    """
    def __init__(
        self,
        example_input: torch.Tensor,
        hidden_dim=128,
        output_dim=64,
    ):
        self.embed_model = MatrixPairEmbedding(hidden_dim, output_dim)
        self.example_input = example_input

        self.current_input = None
        self.current_output = None

        self.nodes = []

    def _calculate_weighted_mse_loss(
        self,
        node_output: torch.Tensor,
        final_output: torch.Tensor,
        alpha=10.0,
        beta=1.0
    ):
        """
        Calculates the supervised loss between the output at a node during training with the 
        actual final output of the input-output pair in the dataset.

        Args:
            node_output (torch.Tensor): Output at a particular node.
            final_output (torch.Tensor): Actual final output that the network should produce.
            alpha (float): Weightage for mismatch in shape contributing to loss. Default is 10.
            beta (float): Weight for mismatch in element values affecting loss. Default is 1.
        """
        shape_loss = ((node_output.size(0) - final_output.size(0))**2 +
            (node_output.size(1) - final_output.size(1))**2)

        value_loss = F.mse_loss(
            F.interpolate(node_output.unsqueeze(0).unsqueeze(0),
                size=final_output.shape, mode="bilinear", align_corners=False).squeeze(),
            final_output
        )

        loss = alpha * shape_loss + beta * value_loss

        return loss

    def _calculate_l1_loss(
        self,
        node_output: torch.Tensor,
        final_output: torch.Tensor
    ):
        """
        Calculates the supervised loss between the output at a node during training with the 
        actual final output of the input-output pair in the dataset.

        Args:
            node_output (torch.Tensor): Output at a particular node.
            final_output (torch.Tensor): Actual final output that the network should produce.
        """
        value_loss = F.smooth_l1_loss(
            F.interpolate(
            node_output.unsqueeze(0).unsqueeze(0),
            size=final_output.shape,
            mode="bilinear",
            align_corners=False).squeeze(0).squeeze(0),
            final_output,
            beta=0.1
        )

        return value_loss

    def _calculate_bce_dice_loss(
        self,
        node_output: torch.Tensor,
        final_output: torch.Tensor,
        alpha=1.0,  # Weight for BCE
        beta=1.0    # Weight for Dice
    ):
        """
        Calculates combined BCE + Dice loss between a predicted 2D tensor (node_output)
        and a ground-truth 2D tensor (final_output), handling shape mismatches.

        Args:
            node_output (torch.Tensor): Predicted 2D tensor.
            final_output (torch.Tensor): Ground-truth 2D tensor.
            alpha (float): Weight for BCE loss.
            beta (float): Weight for Dice loss.
        """
        target = final_output
        pred = torch.sigmoid(F.interpolate(
            node_output.unsqueeze(0).unsqueeze(0),
            size=final_output.shape,
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0))
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)

        bce_loss = F.binary_cross_entropy(pred, target)

        smooth = 1e-6
        intersection = (pred * target).sum()
        dice_coeff = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        dice_loss = 1.0 - dice_coeff

        loss = alpha * bce_loss + beta * dice_loss

        return loss

    def _calculate_output_loss(
        self,
        node_output: torch.Tensor,
        final_output: torch.Tensor,
        loss_op="weighted_mse",
        **kwargs
    ):
        """
        Calculates the supervised loss between the output at a node during training with the 
        actual final output of the input-output pair in the dataset.

        Can perform loss operations of `weighted_mse`, `l1`, `bce_dice` when loss_op is set 
        to any of the listed values.

        Args:
            node_output (torch.Tensor): Output at a particular node.
            final_output (torch.Tensor): Actual final output that the network should produce.
            loss_op (str): Type of loss to apply for network. Default is weighted MSE loss.
            **kwargs: Keyworded parameters to the loss function.
        """
        if loss_op.lower() == "weighted_mse":
            if "alpha" in kwargs and "beta" in kwargs:
                return self._calculate_weighted_mse_loss(
                    node_output,
                    final_output,
                    kwargs["alpha"],
                    kwargs["beta"]
                )
            else:
                return self._calculate_weighted_mse_loss(
                    node_output,
                    final_output
                )
        elif loss_op.lower() == "l1":
            return self._calculate_l1_loss(
                node_output,
                final_output
            )
        elif loss_op.lower() == "bce_dice":
            if "alpha" in kwargs and "beta" in kwargs:
                return self._calculate_bce_dice_loss(
                    node_output,
                    final_output,
                    kwargs["alpha"],
                    kwargs["beta"]
                )
            else:
                return self._calculate_bce_dice_loss(
                    node_output,
                    final_output
                )
        else:
            raise ValueError(f"loss_op value {loss_op} does not exist.")

    def _find_next_nodes(
        self,
        embed: torch.Tensor,
        n_similars=3
    ):
        """
        Calculates similarity scores between the embedding inputted and the embeddings of all
        nodes present in the network and returns the most similar node object as well as the 
        id for the same.

        Args:
            embed (tensor): Embedding matrix to compare with all node embeddings.
            n_similars (int): Most n similar node embeddings.

        Returns:
            Node: Most similar node to the input embedding.
            int: ID of most similar node.
        """
        embed_flat = embed.reshape(1, -1)
        similarities = {}

        for node_idx, node in enumerate(self.nodes):
            node_embed = node.get_embedding().reshape(1, -1)

            sim = F.cosine_similarity(embed_flat, node_embed, dim=1)        # pylint: disable=E1102
            sim = (sim + 1.0) * 0.5

            similarities[node_idx] = sim.item()

        similar_ids_desc = sorted(similarities, key=lambda k: similarities[k], reverse=True)

        return similar_ids_desc[:n_similars]

    def add_nodes(
        self,
        func_keywords: list,
    ):
        """
        Appends a new node to the network, given a valid function keyword.

        Args:
            func_keywords (list): List of keyword names for all new nodes' primitive functions.
        """
        for keyword in func_keywords:
            new_node = Node(len(self.nodes), keyword, self.example_input, self.embed_model)
            self.nodes.append(new_node)

        return

    def train_iter(
        self,
        max_overfits=10,
        loss_op="l1",
        n_similars=3
    ):
        """
        One iteration of training the network with a single input-output
        pair of matrices. The input is processed through the network until
        overfitting begins (loss starts increasing), at which point the 
        output is finalized and the part of the network traversed is adjusted
        by backpropagation according to the final calculated loss.

        Currently uses hit and trial checking for next node since sample space
        for primitive functions is relatively small.

        Args:
            max_overfits (int): Number of overfits checked via loss before training stopped.
            loss_op (str): Determines the loss func used. Can be `weighted_mse`, `l1`, `bce_dice`.
            n_similars (int): Small number of most similar nodes to check for during training.
        """
        if not isinstance(self.current_input, torch.Tensor):
            raise TypeError(f"self.current_input must be of type `torch.Tensor`\
                but is of type `{type(self.current_input)}`.")

        if not isinstance(self.current_output, torch.Tensor):
            raise TypeError(f"self.current_output must be of type `torch.Tensor`\
                but is of type `{type(self.current_output)}`.")

        best_loss = 9999999
        overfits = 0
        best_output = None
        best_nodes_idx_path = None
        next_node_candidates = []

        # TODO: Check if clone() needed for input copy here (can original tensor be changed)
        node_input = self.current_input
        traversed_nodes = []
        pair_embeds = []

        while overfits < max_overfits:
            candidate_loss = 9999999
            chosen_output, chosen_node = None, None

            # Adding pair embedding for best found node
            pair_embed = self.embed_model.step(
                node_input,
                self.current_output,
                train=True,
                pair_features=True
            )[2]

            next_node_candidates = self._find_next_nodes(pair_embed, n_similars)

            for cand_id in next_node_candidates:
                # Prevents any loops for non-loopable functions
                if traversed_nodes and cand_id == traversed_nodes[-1]:
                    if self.nodes[cand_id].is_loopable is False:
                        continue

                cand_output = self.nodes[cand_id].forward(
                    node_input,
                    actual_output=self.current_output
                )
                loss = self._calculate_output_loss(cand_output, self.current_output, loss_op)

                if loss < candidate_loss:
                    candidate_loss = loss
                    chosen_output = cand_output
                    chosen_node = cand_id

            node_input = chosen_output
            traversed_nodes.append(chosen_node)
            pair_embeds.append(pair_embed)

            if candidate_loss >= best_loss:
                overfits += 1
            else:
                best_output = chosen_output
                best_nodes_idx_path = traversed_nodes
                overfits = (overfits - 1) if overfits > 0 else 0

                best_loss = candidate_loss

        best_loss = self._calculate_output_loss(best_output, self.current_output, loss_op)

        # Updating all traversed nodes' embeddings with complete output loss
        for idx, node_idx in enumerate(best_nodes_idx_path):
            self.nodes[node_idx].update_embedding(pair_embeds[idx], best_loss)

        return (
            best_nodes_idx_path,
            best_output,
            best_loss
        )

    def train(
        self,
        epochs=10,
        training_dir="./data/ARC-AGI-2-main/data/training/",
        loss_op="l1",
        save=True,
        batch_verbose=False,
        max_batches=None
    ):
        """
        Method used to train the network model and update its embeddings accordingly.

        Args:
            epochs (int): Number of epochs to run training for.
            training_dir (str, optional): Relative folder location to all training examples.
            save (bool): Whether to save the model as pickle file after training. Default is True.
            batch_verbose (bool): Whether to display loss for each batch individually.
            max_batches (int): Maximum number of batches to train per epoch. None for all batches.
        """
        if os.path.exists(training_dir) and os.path.isdir(training_dir):
            filenames = os.listdir(training_dir)
        else:
            raise NotADirectoryError(f"{training_dir} does not exist or is not a folder location.")

        if len(filenames) == 0:
            raise FileNotFoundError(f"No files found at location {training_dir}.")

        if len(self.nodes) <= 2:
            raise RuntimeError(f"List of nodes in network too small: {len(self.nodes)} present.")

        if max_batches is not None and isinstance(max_batches, int):
            filenames = filenames[:max_batches]

        for epoch in range(epochs):
            epoch_loss = 0

            for filename in filenames:
                iter_loss = 0

                with open(os.path.join(training_dir, filename), "rb") as file:
                    data = json.load(file)
                    file.close()

                for example in data["train"]:
                    self.current_input = torch.tensor(example["input"], dtype=torch.float32)
                    self.current_output = torch.tensor(example["output"], dtype=torch.float32)

                    _, _, example_loss = self.train_iter(loss_op=loss_op)
                    iter_loss += round(example_loss.item(), 3)

                iter_loss = iter_loss / len(data["train"])
                epoch_loss += iter_loss
                if batch_verbose:
                    print(f"Batch loss for task {filename.split(".")[0]}: {round(iter_loss, 3)}")

            epoch_loss = epoch_loss / len(filenames)
            print(f"\nEpoch {epoch+1}/{epochs}: loss - {round(epoch_loss, 3)}")

        if save:
            self.save_model()

        return

    def save_model(self, path="saves/"):
        """
        Method to save model state as a pickle file so it can be reused
        later without training from scratch.

        Args:
            path (str, optional): Path to folder to save model to. Defaults to "saves/".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if len(os.listdir(path)) != 0:
            filename = f"model_{len(os.listdir(path))}.pkl"
        else:
            filename = "model.pkl"

        path = os.path.join(path, filename)

        with open(path, "wb") as f:
            pickle.dump(self, f)
            f.close()

        print(f"Model saved to {path}.")
        return

    @classmethod
    def load_model(cls, path="saves/model.pkl"):
        """
        Loads a previously saved model object and returns it to a receiving
        variable.

        Args:
            path (str, optional): Path to folder with the model saved in it.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
            f.close()

        print(f"Model loaded from {path}.")
        return model

__all__ = [ 'Network' ]
