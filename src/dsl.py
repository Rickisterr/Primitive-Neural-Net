import torch
import torch.nn.functional as F

class DSLPrimitives:
    """
    Class containing DSL primitives from which select primitives can be chosen
    to be used when initializing a network of nodes. The DSL primitives are
    stored as methods of this class and need to be simply imported and passed
    as arguments to the network or its nodes as and where required.
    """
    def __init__(self, func_keyword):
        self.dsl_keywords = {
            "matrix_transpose": self.__mat_transpose,
            "matrix_concat": self.__mat_concat,
            "matrix_split": self.__mat_split,
            "matrix_filter_rows": self.__mat_filter_rows,
            "matrix_filter_cols": self.__mat_filter_cols,
            "matrix_pool": self.__mat_pool,
            "matrix_rotate": self.__mat_rotate,
            "matrix_reflect": self.__mat_reflect,
            "matrix_translate" : self.__mat_translate,
            "matrix_add" : self.__mat_add,
            "matrix_sub" : self.__mat_sub,
            "matrix_mult" : self.__mat_mult
        }

        self.dsl_func = None
        self.update_func(func_keyword)

    def __mat_transpose(self, mat):
        """
        Method used to transpose a matrix.

        Args:
            mat (tensor): Input 2D tensor.

        Returns:
            tensor: Transposed matrix.
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError(f"Matrix must be of type 'torch.Tensor' but is '{type(mat)}'.")
        if len(mat.shape) != 2:
            raise ValueError(f"Matrix should be 2D but has {len(mat.shape)} dimensions.")

        return mat.T

    def __mat_concat(self, mat_1, mat_2, axis=0):
        """
        Method used to concatenate two matrices along a specified axis.

        Args:
            mat_1 (tensor): First matrix.
            mat_2 (tensor): Second matrix.
            axis (int): Axis to concatenate along (0 = vertical, 1 = horizontal).

        Returns:
            tensor: Concatenated matrix.
            None: Concatenation not possible due to shape mismatch.
        """
        if not isinstance(mat_1, torch.Tensor) or not isinstance(mat_2, torch.Tensor):
            raise TypeError("Both inputs must be tensors.")

        if len(mat_1.shape) != 2 or len(mat_2.shape) != 2:
            raise ValueError("Both inputs must be 2D matrices.")

        try:
            out = torch.cat((mat_1, mat_2), dim=axis)
        except RuntimeError:
            return None

        return out

    def __mat_split(self, mat, axis=0, index=1, split_index=0):
        """
        Method used to split a matrix along a specified axis.

        Args:
            mat (tensor): Input matrix.
            axis (int): Axis to split along.
            index (int): Index at which to split.
            split_index (int): Which split to return. 0 for first split and 1 for second.

        Returns:
            tensor: required split of the original
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError("Matrix must be a tensor.")
        if len(mat.shape) != 2:
            raise ValueError("Matrix must be 2D.")

        if axis == 0:
            if split_index == 0:
                return mat[:index, :]
            else:
                return mat[index:, :]
        elif axis == 1:
            if split_index == 0:
                return mat[:, :index]
            else:
                return mat[:, index:]
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    def __mat_filter_rows(self, mat, predicate):
        """
        Method used to select rows satisfying a predicate.

        Args:
            mat (tensor): Input matrix.
            predicate (callable): Function returning True/False for a row.

        Returns:
            tensor: Filtered matrix containing only selected rows.
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError("Matrix must be a tensor.")
        if not callable(predicate):
            raise TypeError("Predicate must be callable.")

        mask = torch.tensor([predicate(row) for row in mat], dtype=torch.bool)

        return mat[mask]

    def __mat_filter_cols(self, mat, predicate):
        """
        Method used to select columns satisfying a predicate.

        Args:
            mat (tensor): Input matrix.
            predicate (callable): Function returning True/False for a column.

        Returns:
            tensor: Filtered matrix containing only selected columns.
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError("Matrix must be a tensor.")
        if not callable(predicate):
            raise TypeError("Predicate must be callable.")

        mask = torch.tensor([predicate(mat[:, i]) for i in range(mat.shape[1])], dtype=torch.bool)
        return mat[:, mask]

    def __mat_pool(self, mat, pool_sz=(2, 2), pool_op="max"):
        """
        Method used to pool (max/avg) within non-overlapping blocks.

        Args:
            mat (tensor): Input matrix.
            pool_sz (tuple): Block size (h, w).
            pool_op (str): 'max' or 'avg'.

        Returns:
            tensor: Pooled matrix.
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError("Matrix must be a tensor.")
        if len(mat.shape) != 2:
            raise ValueError("Matrix must be 2D.")

        mat_reshaped = mat.unfold(0, pool_sz[0], pool_sz[0]).unfold(1, pool_sz[1], pool_sz[1])

        if pool_op == "max":
            out = mat_reshaped.contiguous().view(-1, pool_sz[0]*pool_sz[1]).max(dim=1)[0]
        elif pool_op == "avg":
            out = mat_reshaped.contiguous().view(-1, pool_sz[0]*pool_sz[1]).mean(dim=1)
        else:
            raise ValueError("pool_op must be 'max' or 'avg'.")

        side = int(out.numel() ** 0.5)
        return out.view(side, side)

    def __mat_rotate(self, mat, k=1):
        """
        Method used to rotate a matrix by 90 degrees k times.

        Args:
            mat (tensor): Input matrix.
            k (int): Number of 90-degree rotations (clockwise).

        Returns:
            tensor: Rotated matrix.
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError("Matrix must be a tensor.")
        if len(mat.shape) != 2:
            raise ValueError("Matrix must be 2D.")

        k = k % 4
        return torch.rot90(mat, k, [0, 1])

    def __mat_reflect(self, mat, axis="horizontal"):
        """
        Method used to reflect a matrix horizontally or vertically.

        Args:
            mat (tensor): Input matrix.
            axis (str): 'horizontal' or 'vertical' to determine axis of reflection.

        Returns:
            tensor: Reflected matrix.
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError("Matrix must be a tensor.")
        if len(mat.shape) != 2:
            raise ValueError("Matrix must be 2D.")

        if axis == "horizontal":
            return torch.flip(mat, [1])
        elif axis == "vertical":
            return torch.flip(mat, [0])
        else:
            raise ValueError("Axis must be 'horizontal' or 'vertical'.")

    def __mat_translate(self, mat, translation):
        """
        Method used to translate matrices by a certain defined distance.

        Args:
            mat (tensor): Input matrix.
            translation (tuple): tuple of x, y values to translate matrices by.

        Returns:
            tensor: Output translated matrix.
        """
        if not isinstance(mat, torch.Tensor):
            raise TypeError(f"Argument passed to dsl primitive must be of type 'torch.Tensor' \
                but is of type '{type(mat)}'.")

        if not isinstance(translation, tuple):
            raise TypeError(f"Argument passed to dsl primitive must be of type 'tuple' \
                but is of type '{type(translation)}'.")

        shift = [0, 0, 0, 0]

        if translation[0] < 0:
            shift[1] = abs(translation[0])
        else:
            shift[0] = translation[0]

        if translation[1] < 0:
            shift[2] = abs(translation[1])
        else:
            shift[3] = translation[1]

        translated = F.pad(mat, shift)

        return translated

    def __mat_add(self, mat_1, mat_2):
        """
        Method used to add two matrices together.

        Args:
            mat_1 (tensor): First matrix addend.
            mat_2 (tensor): Second matrix addend.

        Returns:
            tensor: Sum of both matrices.
            None: Sum not possible (shape mismatch). Tells node not to update its embedding.
        """
        if not isinstance(mat_1, torch.Tensor) or not isinstance(mat_2, torch.Tensor):
            raise TypeError(f"Matrices must both be of type 'torch.Tensor' but \
                are actually '{type(mat_1)}' and '{type(mat_2)}'.")

        if len(mat_1.shape) != 2 or len(mat_2.shape) != 2:
            raise ValueError(f"Matrices should both be 2 dimensions but \
                are actually '{len(mat_1.shape)}' and '{len(mat_2.shape)}'.")

        if mat_1.shape == mat_2.shape:
            out = torch.add(mat_1, mat_2)
        else:
            return None

        return out

    def __mat_sub(self, mat_1, mat_2):
        """
        Method used to subtract two matrices together.

        Args:
            mat_1 (tensor): First matrix to subtract from.
            mat_2 (tensor): Second matrix to subtract with.

        Returns:
            tensor: Difference of both matrices.
            None: Difference not possible (shape mismatch). Tells node not to update its embedding.
        """
        if not isinstance(mat_1, torch.Tensor) or not isinstance(mat_2, torch.Tensor):
            raise TypeError(f"Matrices must both be of type 'torch.Tensor' but \
                are actually '{type(mat_1)}' and '{type(mat_2)}'.")

        if len(mat_1.shape) != 2 or len(mat_2.shape) != 2:
            raise ValueError(f"Matrices should both be 2 dimensions but \
                are actually '{len(mat_1.shape)}' and '{len(mat_2.shape)}'.")

        if mat_1.shape == mat_2.shape:
            out = torch.sub(mat_1, mat_2)
        else:
            return None

        return out

    def __mat_mult(self, mat_1, mat_2):
        """
        Method used to multiply two matrices together.

        Args:
            mat_1 (tensor): First matrix multiplier.
            mat_2 (tensor): Second matrix multiplier.

        Returns:
            tensor: Product of both matrices
            None: Product not possible (shape mismatch). Tells node not to update its embedding.
        """
        if not isinstance(mat_1, torch.Tensor) or not isinstance(mat_2, torch.Tensor):
            raise TypeError(f"Matrices must both be of type 'torch.Tensor' but \
                are actually '{type(mat_1)}' and '{type(mat_2)}'.")

        if len(mat_1.shape) != 2 or len(mat_2.shape) != 2:
            raise ValueError(f"Matrices should both be 2 dimensions but \
                are actually '{len(mat_1.shape)}' and '{len(mat_2.shape)}'.")

        if mat_1.shape[1] == mat_2.shape[0]:
            out = torch.mm(mat_1, mat_2)
        else:
            return None

        return out

    def update_func(self, func_keyword):
        """
        Method to update the DSL primitive being used.

        Args:
            func_keyword (str): Name of function stored in the DSL functions keywords directory.
        """
        if func_keyword not in self.dsl_keywords:
            raise ValueError(f"Function '{func_keyword}' not registered as a DSL primitive.")

        self.dsl_func = self.dsl_keywords[func_keyword]

    def apply_func(self, *args, **kwargs):
        """
        Class method to process an input and produce an output for this
        DSL primitive object.

        Args:
            *args: Arguments to the dsl function without keywords.
            **kwargs: Keyworded arguments to the dsl for configuration.
        """
        return self.dsl_func(*args, **kwargs)

__all__ = [ 'DSLPrimitives' ]
