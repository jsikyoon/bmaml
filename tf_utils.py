import tensorflow as tf 
import numpy as np

# NOTE: some functions are copied from pysgmcmc https://github.com/MFreidank/pysgmcmc

def pdist(tensor, metric="euclidean"):
    assert isinstance(tensor, (tf.Variable, tf.Tensor,)), "tensor_utils.pdist: Input must be a `tensorflow.Tensor` instance."

    if len(tensor.shape.as_list()) != 2:
        raise ValueError('tensor_utils.pdist: A 2-d tensor must be passed.')

    if metric == "euclidean":

        def pairwise_euclidean_distance(tensor):
            def euclidean_distance(tensor1, tensor2):
                return tf.norm(tensor1 - tensor2)

            m = tensor.shape.as_list()[0]

            distances = []
            for i in range(m):
                for j in range(i + 1, m):
                    distances.append(euclidean_distance(tensor[i], tensor[j]))
            return tf.convert_to_tensor(distances)

        metric_function = pairwise_euclidean_distance
    else:
        raise NotImplementedError(
            "tensor_utils.pdist: "
            "Metric '{metric}' currently not supported!".format(metric=metric)

        )

    return metric_function(tensor)


def _is_vector(tensor):
    return len(tensor.shape.as_list()) == 1 

def median(tensor):
    tensor_reshaped = tf.reshape(tensor, [-1])
    n_elements = tensor_reshaped.get_shape()[0]
    sorted_tensor = tf.nn.top_k(tensor_reshaped, n_elements, sorted=True)
    mid_index = n_elements // 2
    if n_elements % 2 == 1:
        return sorted_tensor.values[mid_index]
    return (sorted_tensor.values[mid_index - 1] + sorted_tensor.values[mid_index]) / 2

def squareform(tensor):
    assert isinstance(tensor, tf.Tensor), "tensor_utils.squareform: Input must be a `tensorflow.Tensor` instance."

    tensor_shape = tensor.shape.as_list()
    n_elements = tensor_shape[0]

    if _is_vector(tensor):
        # vector to matrix
        if n_elements == 0:
            return tf.zeros((1, 1), dtype=tensor.dtype)

        # Grab the closest value to the square root of the number
        # of elements times 2 to see if the number of elements is
        # indeed a binomial coefficient
        dimension = int(np.ceil(np.sqrt(n_elements * 2)))

        # Check that `tensor` is of valid dimensions
        if dimension * (dimension - 1) != n_elements * 2:
            raise ValueError(
                "Incompatible vector size. It must be a binomial "
                "coefficient n choose 2 for some integer n >=2."
            )

        n_total_elements_matrix = dimension ** 2

        # Stitch together an upper triangular matrix for our redundant
        # distance matrix from our condensed distance tensor and
        # two tensors filled with zeros.

        n_diagonal_zeros = dimension
        n_fill_zeros = n_total_elements_matrix - n_elements - n_diagonal_zeros

        condensed_distance_tensor = tf.reshape(tensor, shape=(n_elements, 1))
        diagonal_zeros = tf.zeros(
            shape=(n_diagonal_zeros, 1), dtype=condensed_distance_tensor.dtype
        )
        fill_zeros = tf.zeros(
            shape=(n_fill_zeros, 1), dtype=condensed_distance_tensor.dtype
        )

        def upper_triangular_indices(dimension):
            """ For a square matrix with shape (`dimension`, `dimension`),
                return a list of indices into a vector with
                `dimension * dimension` elements that correspond to its
                upper triangular part after reshaping.
            Parameters
            ----------
            dimension : int
                Target dimensionality of the square matrix we want to
                obtain by reshaping a `dimension * dimension` element
                vector.
            Yields
            -------
            index: int
                Indices are indices into a `dimension * dimension` element
                vector that correspond to the upper triangular part of the
                matrix obtained by reshaping it into shape
                `(dimension, dimension)`.
            """
            assert dimension > 0, "tensor_utils.upper_triangular_indices: Dimension must be positive integer!"

            for row in range(dimension):
                for column in range(row + 1, dimension):
                    element_index = dimension * row + column
                    yield element_index

        # General Idea: Use that redundant distance matrices are symmetric:
        # First construct only an upper triangular part and fill
        # everything else with zeros.
        # To the resulting matrix add its transpose, which results in a full
        # redundant distance matrix.

        all_indices = set(range(n_total_elements_matrix))
        diagonal_indices = list(range(0, n_total_elements_matrix, dimension + 1))
        upper_triangular = list(upper_triangular_indices(dimension))

        remaining_indices = all_indices.difference(
            set(diagonal_indices).union(upper_triangular)
        )

        data = (
            # diagonal zeros of our redundant distance matrix
            diagonal_zeros,
            # upper triangular part of our redundant distance matrix
            condensed_distance_tensor,
            # fill zeros for lower triangular part
            fill_zeros
        )

        indices = (
            tuple(diagonal_indices),
            tuple(upper_triangular),
            tuple(remaining_indices)
        )

        stitch_vector = tf.dynamic_stitch(data=data, indices=indices)

        # reshape into matrix
        upper_triangular = tf.reshape(stitch_vector, (dimension, dimension))

        # redundant distance matrices are symmetric
        lower_triangular = tf.transpose(upper_triangular)

        return upper_triangular + lower_triangular
    else:
        raise NotImplementedError(
            "tensor_utils.squareform: Only 1-d "
            "(vector) input is supported!"
        )

