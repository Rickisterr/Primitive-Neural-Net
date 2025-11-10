"""
This module defines the entire network of nodes containing primitive
functions that trains to update connections and embeddings. To use,
import the `Network` class.
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

        self.nodes = []
        self.nodes_embeds_flat = []

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

    def train_iter(self, input_mat, output_mat):
        """
        One iteration of training the network with a single input-output
        pair of matrices. The input is processed through the network until
        the loss calculated at each node starts increasing, at which point
        the output is finalized, 

        Args:
            input_mat (_type_): _description_
            output_mat (_type_): _description_
        """
