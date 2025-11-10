"""
This module contains the class required to define a node in the network.
To use, import the `Node` class.
"""
import torch
from src.dsl import DSLPrimitives

class Node:
    """
    Class for a node in a graph comprising of a domain specific language
    to depict some transformation of the input data at each node.

    Args:
        node_id (int): ID by which to identify this particular node.
        next_node_ids (list): Array of nodes current one can pass output to.
        dsl_keyword (function): DSL for this node as determined by a function.
        init_embedding (tensor): Feature embedding representing inputs to node.
    """
    def __init__(self, node_id, next_node_ids, dsl_keyword, init_embedding):
        self.node_id = node_id
        self.next_nodes = next_node_ids

        self.dsl_func = None
        self.set_dsl(dsl_keyword)              # Node function defined by domain specific language

        self.__feature_embed = init_embedding     # Embedding representing node inputs' features

    def _update_embedding(self, input_embedding, eps=1e-8):
        """
        Class method to update the feature embedding representing the overall
        features of all inputs passed through the node to gather a general
        representation of inputs that have passed through this node.

        Args:
            input_embedding (tensor): Embedding of the input to this node.

        Returns:
            None
        """
        if input_embedding.shape != self.__feature_embed.shape:
            raise ValueError(f"input_embedding shape {input_embedding.shape} must be {self.__feature_embed.shape}")       # pylint: disable=C0301

        similarity = torch.mm(input_embedding, self.__feature_embed) / (
            torch.norm(input_embedding) * torch.norm(self.__feature_embed) + eps
        )
        alpha = torch.clamp(similarity, 0, 1)

        # Updating node's feature embedding
        self.__feature_embed = alpha * input_embedding + (1 - alpha) * self.__feature_embed

        return

    def _apply_func(self, *args, **kwargs):
        """
        Apply the primitive DSL function to the inputs to this function.

        Args:
            *args: Indexable arguments passed (tensor inputs).
            **kwargs: Keyworded arguments passed (config related options).

        Returns:

        """
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                raise ValueError(f"{arg} must be of type 'torch.Tensor' \
                    instead of '{type(arg)}'.")

        return self.dsl_func(*args, **kwargs)

    def get_embedding(self):
        """
        Retrieve the node embedding for this node.
        """
        return self.__feature_embed

    def update_output_connections(self, new_connections):
        """
        Method to update the array of all next nodes from current node.

        Args:
            new_connections (list): New connections from current node.
        """
        self.next_nodes = new_connections
        return

    def set_dsl(self, dsl_keyword):
        """
        Update the DSL function of the node using a particular keyword.

        Args:
            dsl_keyword (str): Keyword to attempt to switch the dsl function attached to the node.
        """
        if isinstance(self.dsl_func, DSLPrimitives):
            self.dsl_func.update_func(dsl_keyword)
        elif self.dsl_func is None:
            self.dsl_func = DSLPrimitives(dsl_keyword)
        else:
            raise ValueError(f"self.dsl_func should be of type 'DSLPrimitive' \
                instead of '{type(self.dsl_func)}'.")

        return

    def forward(self, mat, mat_embed, mat_2=None, **kwargs):
        """
        Pass inputs through the node to produce an output from the same.

        Args:
            mat (tensor): Main tensor input for this node.
            mat_embed (tensor): Embedding for main tensor input.
            mat_2 (tensor): Secondary tensor input for this node if needed. None if unary function.
            **kwargs: Keyworded arguments passed (config related options).

        Returns:
            tensor: Tensor output of node's function.
        """
        if mat_2 is None:
            output = self._apply_func(mat, **kwargs)
        else:
            output = self._apply_func(mat, mat_2, **kwargs)

        # If primitive function returns None (function cannot be run on this input(s)), skip node
        if output is None:
            return mat

        self._update_embedding(mat_embed)

        return output

__all__ = [ 'Node' ]
