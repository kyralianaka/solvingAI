import numpy as np


class solvingAI:
    def __init__(self):
        pass

    def initialize(self):
        # Nischal
        N = self.N
        k = self.k
        D = int(self.e * N)
        w = np.random.rand(N, N)

        sparse_weight_matrix = np.random.rand(N, N)
        rates_0 = np.random.rand(N, 1)
        # make matrix k percent sparse
        num_zero_elements = int((k / 100) * N * N)
        zero_indices = np.random.choice(N * N, num_zero_elements, replace=False)
        sparse_weight_matrix.flat[zero_indices] = 0

        # set the IE weights to zero
        sparse_weight_matrix[: N - D, N - D :] = 0

        # get indices of non-zero elements that were originally in the top right block
        non_zero_indices = np.transpose(np.nonzero(weight_matrix[: N - D, N - D :]))
        non_zero_indices[:, 1] += N - D

        return sparse_weight_matrix, rates_0, non_zero_indices

    def rate_eqns(self):
        # Kyra
        pass

    def weight_update(self):
        # Nischal
        pass

    def integrate(self):
        # Nischal/Kyra
        pass

    def run_sim(self):
        weights_0 = self.initialize()

        soln, weights = self.integrate(weights_0)

        return soln, weights
