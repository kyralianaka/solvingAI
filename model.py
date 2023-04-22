import numpy as np


class solvingAI:
    def __init__(self):
        # Number of neurons
        self.N = 100
        # Fraction of excitatory neurons
        self.i = 0.2
        self.D = int(self.i * self.N)
        # Fraction of null connections
        self.k = 90
        self.D = int(self.i * self.N)

        # Start with ReLu activation function
        self.activate = lambda x: np.maximum(0, x)

        # Integration parameters
        self.h = 0.05  # step size, 1/s
        self.T = 5  # total time, sec

        # Weight update parameters
        self.lr = 0.05  # learning rate
        self.rho = 1  # target firing rate

    def initialize(self):
        # Nischal
        N = self.N
        k = self.k
        D = int(self.i * N)
        w = np.random.rand(N, N)

        sparse_weight_matrix = np.random.rand(N, N)
        weight_matrix = np.copy(sparse_weight_matrix)
        rates_0 = np.random.rand(N, 1)
        # make matrix k percent sparse
        num_zero_elements = int((k / 100) * N * N)
        zero_indices = np.random.choice(N * N, num_zero_elements, replace=False)
        sparse_weight_matrix.flat[zero_indices] = 0
        non_zero_indices = np.transpose(np.nonzero(sparse_weight_matrix[: N - D, N - D :]))

        # indices of connections w.r.t to the top right block
        self.IEcxns = np.asarray(non_zero_indices)

        # indices of connections w.r.t to the full weight matrix
        self.full_idx = self.IEcxns.copy()
        self.full_idx[:,1] += self.N - self.D

        # set the IE weights to zero
        sparse_weight_matrix[: N - D, N - D :] = 0

        return sparse_weight_matrix, rates_0

    def rate_eqns(self, y, w):
        return -y + 3*w @ self.activate(y + 10)

    def weight_update(self, weights, rates):

        r_e= rates[:self.N - self.D]
        r_i = rates[self.N - self.D :]

        updates = self.lr * (r_e @ r_i.T - self.rho * r_e * np.ones((1, self.D)))
        weights[self.full_idx[:,1], self.full_idx[:,1]] += updates[self.IEcxns[:,0],self.IEcxns[:,1]]

        return weights

    def integrate(self, weights_0, rates_0):
        # Initialize
        soln = np.zeros((self.N, int(1/self.h * self.T) + 1))
        soln[:, 0] = np.squeeze(rates_0, axis=1)
        weights = weights_0
        weight_traj = np.zeros((self.IEcxns.shape[0], int(self.T * 1/self.h)))

        for i in range(0, self.T):
            soln[:, i+1] = soln[:, i] + self.rate_eqns(soln[:, i], weights) * self.h

            # Save the nonzero IE weights before changing them
            # weight_traj[:, i] = weights[self.IEcxns]
            # Update the weights
            if i % 10 == 0:
                weights = self.weight_update(weights, soln[:, i+1])

        return soln, weights

    def run_sim(self):
        weights_0, rates_0 = self.initialize()
        soln, weights = self.integrate(weights_0, rates_0)

        return soln, weights
