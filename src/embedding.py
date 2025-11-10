"""
This module contains the classes required to perform embedding generation
on a pair of input, output pairs. To use, import the `MatrixPairEmbedding`
class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class _MatrixEncoder(nn.Module):
    """
    Encoder part of Embedding generator Module.

    Args:
        in_channels (int): Number of input channels for an input. Default is 1.
        hidden_dim (int): Number of channels to process input to in hidden layers. Default is 128.
        output_dim (int): Number of channels for embedding when outputted. Default is 64.
    """
    def __init__(self, in_channels=1, hidden_dim=128, output_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pooling_down = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Linear(1, output_dim)

        return

    def forward(self, x: torch.Tensor):
        """
        Forward method for MatrixPairEmbedding class.

        Args:
            x (torch.Tensor): Input tensor to process using model

        Returns:
            torch.Tensor: Output tensor after processing x through model.
        """
        features = self.encoder(x)
        pooled = self.pooling_down(features)

        z = self.projector(pooled)
        z = F.normalize(z, dim=-1)

        return z

class MatrixPairEmbedding:
    """
    Calculates the Embeddings for a pair of input-output matrices.

    Args:
        hidden_dim (int): Number of channels of embeddings. Default is 128.
        output_dim (int): Number of features per channel in embeddings. Default is 64.
        lr (float): Learning rate for model optimizer.
        decay_rate (float): Rate of decay for model weights.
        device (str): Whether to use gpu (cuda) for compute or cpu. Default is "cuda".
    """
    def __init__(self, hidden_dim=128, output_dim=64, lr=1e-4, decay_rate=1e-2, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = _MatrixEncoder(hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay_rate)
        self.temp = 0.1

    def _calculate_loss(self, embed_input: torch.Tensor, embed_output: torch.Tensor, temp=0.1):
        """
        Calculates unsupervised loss to give similar pairs more positive values
        and different pairs more negative values to learn to perform embeddings
        on matrix pairs.

        Args:
            embed_input (torch.Tensor): Embedding of input matrix in input-output pair.
            embed_output (torch.Tensor): Embedding of output matrix in input-output pair.
            temp (float): Factor determining the degree of contrast. Defaults to 0.1.

        Returns:
            torch.Tensor: Cross entropy loss calculated
        """
        embed_input = embed_input.flatten(start_dim=1)
        embed_output = embed_output.flatten(start_dim=1)

        batch_size = embed_input.size(0)
        z_combined = torch.cat([embed_input, embed_output], dim=0)
        similarity = F.cosine_similarity(z_combined.unsqueeze(1), z_combined.unsqueeze(0), dim=2)       # pylint: disable=E1102

        labels = torch.arange(batch_size).to(embed_input.device)
        labels = torch.cat([labels, labels], dim=0)

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(embed_input.device)
        similarity = similarity.masked_fill(mask, -9e15)

        similarity /= temp
        loss = F.cross_entropy(similarity, labels)

        return loss

    def _pair_embeddings(self, embed1: torch.Tensor, embed2: torch.Tensor):
        """
        Returns the compiled embedding for a matrix pair, retaining the features
        of the input and output matrices by themselves as well as the difference
        in their features in a single embedding.

        Args:
            embed1 (torch.Tensor): Embedding of input matrix
            embed2 (torch.Tensor): Embedding of output matrix

        Returns:
            torch.Tensor: Tensor representing the embedding of the entire pair.
        """
        embed_compiled = torch.cat([embed1, embed2, embed1 - embed2], dim=-1)
        embed_compiled = F.normalize(embed_compiled, dim=-1)

        return embed_compiled

    def _preprocess_data(self, input_mat: torch.Tensor, min_val=0, max_val=9):
        """
        Preprocesses the data before embedding.

        Args:
            input_mat (torch.Tensor): Input matrix to preprocess before embedding.
            min_val (int): Minimum value possible for any of the tensor's elements.
            max_val (int): Maximum value possible for any of the tensor's elements.

        Returns:
            torch.Tensor: Preprocessed result of input tensor.
        """
        input_mat = input_mat.float()
        input_mat = (input_mat - min_val) / (max_val - min_val + 1e-9)

        return input_mat

    def step(self, input_mats: torch.Tensor, output_mats: torch.Tensor, train=True, pair_features=False):       # pylint: disable=C0301
        """
        Processes one input-output matrix pair to fit to the model and returns the
        individual and paired feature embeddings of the input-output pair.

        Args:
            input_mat (torch.Tensor): Input matrix of the input-output pair.
            output_mat (torch.Tensor): Output matrix of the input-output pair.
            train (bool): Training and evaluating if True; else, just inference. Default is True.
            pair_features (bool): Whether to calculate pair embedding. Default is False.

        Returns:
            torch.Tensor: Individual and compiled embeddings for an input-output matrix pair.
        """
        input_mats = self._preprocess_data(input_mats)
        output_mats = self._preprocess_data(output_mats)

        if len(input_mats.shape) == 2:
            input_mats = input_mats.unsqueeze(0).unsqueeze(0)
        else:
            raise IndexError(f"input_mats must be 2 dimensions, but has {len(input_mats.shape)}.")

        if len(output_mats.shape) == 2:
            output_mats = output_mats.unsqueeze(0).unsqueeze(0)
        else:
            raise IndexError(f"output_mats must be 2 dimensions, but has {len(input_mats.shape)}.")

        input_mats = input_mats.to(self.device)
        output_mats = output_mats.to(self.device)

        max_h = max(input_mats.shape[2], output_mats.shape[2])
        max_w = max(input_mats.shape[3], output_mats.shape[3])

        input_mats = F.pad(input_mats, (0, max_w - input_mats.shape[3], 0, max_h - input_mats.shape[2]))        # pylint: disable=C0301
        output_mats = F.pad(output_mats, (0, max_w - output_mats.shape[3], 0, max_h - output_mats.shape[2]))    # pylint: disable=C0301

        if train:
            self.optimizer.zero_grad()
            embed_in = self.model(input_mats)
            embed_out = self.model(output_mats)

            loss = self._calculate_loss(embed_in, embed_out, self.temp)
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                embed_in = self.model(input_mats)
                embed_out = self.model(output_mats)

        # TODO: If final supervised loss of outputs should backpropagate to fix embeddings,
        # stop detaching embeddings on returning from step function; try stopping detaches
        # regardless.
        if pair_features:
            embed_pairs = self._pair_embeddings(embed_in, embed_out)

            return (
                embed_in.detach().cpu(),
                embed_out.detach().cpu(),
                embed_pairs.detach().cpu()
            )

        return (
            embed_in.detach().cpu(),
            embed_out.detach().cpu()
        )

__all__ = [ "MatrixPairEmbedding" ]
