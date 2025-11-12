"""
This module defines the entire network of nodes containing primitive
functions that trains to update connections and embeddings. To use,
import the `Network` class.

FOR NOW, ONLY SUPPORTS SINGLE OPERAND (UNARY) DSL FUNCTIONS
"""
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
        hidden_dim (int): Number of channels of embeddings. Default is 128.
        output_dim (int): Number of features per channel in embeddings. Default is 64.
    """
    def __init__(self, hidden_dim=128, output_dim=64):
        self.embed_model = MatrixPairEmbedding(hidden_dim, output_dim)

        self.current_input = None
        self.current_output = None

        self.nodes = []
        self.nodes_embeds_flat = []

    def _calculate_output_loss(
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
        shape_loss = torch.sum((
            torch.tensor(node_output.shape, dtype=torch.float32)
            - torch.tensor(final_output.shape, dtype=torch.float32)
        ) ** 2)

        min_h = min(node_output.shape[0], final_output.shape[0])
        min_w = min(node_output.shape[1], final_output.shape[1])

        value_loss = torch.mean((
            node_output[:min_h, :min_w] - final_output[:min_h, :min_w]
        ) ** 2)

        return alpha * shape_loss + beta * value_loss

    def _find_next_node(self, embed: torch.Tensor, next_node_ids: list):
        """
        Calculates similarity scores between the embedding inputted and the embeddings of all
        nodes present in the network and returns the most similar node object as well as the 
        id for the same.

        Args:
            embed (tensor): Embedding matrix to compare with all node embeddings.
            next_node_ids (list): List of node ids to which the output is allowed to pass to.

        Returns:
            Node: Most similar node to the input embedding.
            int: ID of most similar node.
        """
        embed_flat = embed.flatten()
        most_similar_idx = None
        max_similarity = 0

        for idx in next_node_ids:
            sim = F.cosine_similarity(embed_flat, self.nodes_embeds_flat[idx])      # pylint: disable=E1102
            sim = (sim + 1.0) * 0.5

            if max_similarity < sim:
                max_similarity = sim
                most_similar_idx = idx

        return self.nodes[most_similar_idx], most_similar_idx

    def add_node(self, func_keyword: str, init_node_embed: torch.Tensor, is_chainable=False):
        """
        Appends a new node to the network, given a valid function keyword.

        Args:
            func_keyword (str): Keyword name for the new node's DSL defined primitive function.
            init_node_embed (tensor): Node's feature embedding on initialization.
            is_chainable (bool): Whether node can pass its output back to itself. Default is False.
        """
        next_nodes = [idx for idx in range(len(self.nodes))]
        if is_chainable:
            next_nodes.append(len(self.nodes))

        new_node = Node(len(self.nodes), next_nodes, func_keyword, init_node_embed)
        self.nodes.append(new_node)

        self.nodes_embeds_flat.append(init_node_embed.flatten())

        return

    def train_iter(self, max_overfits=5):
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
        """
        if not isinstance(self.current_input, torch.Tensor):
            raise TypeError(f"self.current_input must be of type `torch.Tensor`\
                but is of type `{type(self.current_input)}`.")

        if not isinstance(self.current_output, torch.Tensor):
            raise TypeError(f"self.current_output must be of type `torch.Tensor`\
                but is of type `{type(self.current_output)}`.")

        prev_loss = 9999
        overfits = 0
        best_output = None
        best_traversed_nodes = None
        next_node_candidates = [idx for idx in range(len(self.nodes))]

        # TODO: Check if clone() needed for input copy here (can original tensor be changed)
        node_input = self.current_input
        traversed_nodes = []

        while overfits <= max_overfits:
            candidate_loss = 9999
            chosen_output, chosen_node = None, None

            for cand_id in next_node_candidates:
                cand_output = self.nodes[cand_id].forward(node_input)
                loss = self._calculate_output_loss(cand_output, self.current_output)

                if loss < candidate_loss:
                    candidate_loss = loss
                    chosen_output = cand_output
                    chosen_node = self.nodes[cand_id]

            node_input = chosen_output
            traversed_nodes.append(chosen_node)

            if candidate_loss >= prev_loss:
                overfits += 1
            else:
                best_output = chosen_output
                best_traversed_nodes = traversed_nodes
                overfits = (overfits - 1) if overfits > 0 else 0

        return (
            best_traversed_nodes,
            best_output,
            self._calculate_output_loss(best_output, self.current_output)
        )
