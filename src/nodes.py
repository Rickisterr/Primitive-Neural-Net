"""
This module contains the class required to define a node in the network.
To use, import the `Node` class.
"""
import torch
from src.dsl import DSLPrimitives
from src.embedding import MatrixPairEmbedding

class Node:
    """
    Class for a node in a graph comprising of a domain specific language
    to depict some transformation of the input data at each node.

    Args:
        node_id (int): ID by which to identify this particular node.
        next_node_ids (list): Array of nodes current one can pass output to.
        dsl_keyword (str): DSL name for this node as determined by a function.
        example_input (tensor): Example tensor to initialize embedding with.
        *args: Indexable arguments passed to dsl function.
        **kwargs: Keyworded arguments passed to dsl function.
    """
    def __init__(
        self,
        node_id: int,
        next_node_ids: list,
        dsl_keyword: str,
        example_input: torch.Tensor,
        *args,
        **kwargs
    ):
        self.node_id = node_id
        self.node_func = dsl_keyword
        self.next_nodes = next_node_ids
        self.__node_embed_model = MatrixPairEmbedding()

        self.dsl_func = None
        self.is_loopable = False
        self.set_dsl(dsl_keyword)           # Node function defined by domain specific language

        example_output = self._apply_func(example_input, *args, **kwargs)
        if not isinstance(example_output, torch.Tensor):
            raise TypeError(
                "example_output should be of type `torch.Tensor` " +
                f"but is of type `{type(example_output)}`."
            )

        self.__feature_embed = self.__node_embed_model.step(
            example_input,
            example_output,
            pair_features=True
        )[2]

    def _update_embedding(
        self,
        pair_embedding: torch.Tensor,
        eps=1e-8
    ):
        """
        Class method to update the feature embedding representing the overall
        features of all inputs passed through the node to gather a general
        representation of inputs that have passed through this node.

        Args:
            pair_embedding (tensor): Pair embedding of this node's input and final output.

        Returns:
            None
        """
        if pair_embedding.shape != self.__feature_embed.shape:
            raise ValueError(f"pair_embedding shape {pair_embedding.shape} must be {self.__feature_embed.shape}")       # pylint: disable=C0301

        feature_embed_shape = self.__feature_embed.shape
        pair_embed_shape = pair_embedding.shape

        pair_embedding = pair_embedding.reshape(1, -1)
        self.__feature_embed = self.__feature_embed.reshape(-1, 1)

        similarity = torch.mm(pair_embedding, self.__feature_embed) / (
            torch.norm(pair_embedding) * torch.norm(self.__feature_embed) + eps
        )
        alpha = torch.clamp(similarity, 0, 1)

        # Updating node's feature embedding
        self.__feature_embed = self.__feature_embed.reshape(feature_embed_shape)
        pair_embedding = pair_embedding.reshape(pair_embed_shape)
        self.__feature_embed = alpha * pair_embedding + (1 - alpha) * self.__feature_embed

        return

    def _apply_func(
        self,
        *args,
        **kwargs
    ):
        """
        Apply the primitive DSL function to the inputs to this function.

        Args:
            *args: Indexable arguments passed (tensor inputs).
            **kwargs: Keyworded arguments passed (config related options).

        Returns:

        """
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    "arg must be of type `torch.Tensor` " +
                    f"instead of `{type(arg)}`."
                )

            if len(arg.shape) != 2:
                raise IndexError(
                    "arg must be 2 dimensional but " +
                    f"is {len(arg.shape)} dimensional instead."
                )

        return self.dsl_func.apply_func(*args, **kwargs)

    def get_embedding(self):
        """
        Retrieve the node embedding for this node.
        """
        return self.__feature_embed

    def update_output_connections(
        self,
        new_connections
    ):
        """
        Method to update the array of all next nodes from current node.

        Args:
            new_connections (list): New connections from current node.
        """
        self.next_nodes = new_connections
        return

    def set_dsl(
        self,
        dsl_keyword
    ):
        """
        Update the DSL function of the node using a particular keyword.

        Args:
            dsl_keyword (str): Keyword to attempt to switch the dsl function attached to the node.
        """
        if isinstance(self.dsl_func, DSLPrimitives):
            self.dsl_func.update_func(dsl_keyword)
            self.is_loopable = self.dsl_func.is_loopable[dsl_keyword]
        elif self.dsl_func is None:
            self.dsl_func = DSLPrimitives(dsl_keyword)
            self.is_loopable = self.dsl_func.is_loopable[dsl_keyword]
        else:
            raise ValueError(
                "self.dsl_func should be of type 'DSLPrimitive' " +
                f"instead of '{type(self.dsl_func)}'."
            )

        return

    def forward(
        self,
        mat,
        mat_2=None,
        actual_output=None,
        train=False,
        **kwargs
    ):
        """
        Pass inputs through the node to produce an output from the same.

        Args:
            mat (tensor): Main tensor input for this node.
            mat_2 (tensor): Secondary tensor input for this node if needed. None if unary function.
            actual_output (tensor): Actual output of the entire network to compare to.
            train (bool): Whether to train node's embedding model on this or not.
            **kwargs: Keyworded arguments passed (config related options).

        Returns:
            tensor: Tensor output of node's function.
        """
        if mat_2 is None:
            output = self._apply_func(mat, **kwargs)
        else:
            output = self._apply_func(mat, mat_2, **kwargs)

        if actual_output is None or not isinstance(actual_output, torch.Tensor):
            raise TypeError(
                "actual_output must be of type `torch.Tensor` but is " +
                f"of type `{type(actual_output)}`."
            )

        # If primitive function returns None (function cannot be run on this input(s)), skip node
        if output is None:
            return mat

        pair_embed = self.__node_embed_model.step(
            mat,
            actual_output,
            train=train,
            pair_features=True
        )[2]

        self._update_embedding(pair_embed)

        return output

__all__ = [ 'Node' ]
