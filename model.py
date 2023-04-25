import numpy as np


class solvingAI:

    def __init__(self):
        self.N = 100  # Number of neurons
        self.i = 0.1  # Fraction of inhitatory neurons
        self.D = int(self.i * self.N)  # Number of inhibitory connections
        self.k = 0.4  # Fraction of null connections

        # Start with ReLu activation function
        self.activate = lambda x: np.maximum(0, x)

        # Integration parameters
        self.h = 0.03  # step size, 1/s
        self.T = 10  # total time, sec

        # Weight update parameters
        self.lr = 0.001  # learning rate
        self.rho = 1  # target firing rate


    def initialize(self):
        N = self.N
        D = self.D

        # Initialize firing rates
        rates_0 = np.random.lognormal(1, 0.6, (N, 1))*2

        # make matrix sparse with 1-k fraction of connections
        # sparse_weight_matrix = np.random.rand(N, N)
        sparse_weight_matrix = np.random.lognormal(0, 0.6, (N, N))
        num_zero_elements = int(self.k * N * N)
        zero_indices = np.random.choice(N * N, num_zero_elements, replace=False)
        sparse_weight_matrix.flat[zero_indices] = 0

        # indices of connections w.r.t to the top right block
        non_zero_indices = np.transpose(
            np.nonzero(sparse_weight_matrix[: N - D, N - D :])
        )
        self.IEcxns = np.asarray(non_zero_indices)

        # indices of connections w.r.t to the full weight matrix
        self.full_idx = self.IEcxns.copy()
        self.full_idx[:, 1] += self.N - self.D

        # set the IE weights to zero
        sparse_weight_matrix[: N - D, N - D :] = 0

        return sparse_weight_matrix/((1 - self.k)*self.N), rates_0


    def rate_eqns(self, y, w):

        h_e = y[:self.N - self.D] # excitatory firing rates
        h_i = y[self.N - self.D :] # inhibitory firing rates

        # Divide the weight matrix into the four parts
        w_ee = w[:self.N-self.D, :self.N-self.D]
        w_ie = w[:self.N-self.D, self.N-self.D:]
        w_ei = w[self.N-self.D:, :self.N-self.D]
        w_ii = w[self.N-self.D:, self.N-self.D:]

        # Calculate the firing rate derivatives
        h_e = -h_e + w_ee @ self.activate(h_e) -  w_ie @ self.activate(h_i) + np.random.normal(5, 6)
        h_i = -h_i + w_ei @ self.activate(h_e) - w_ii @ self.activate(h_i) + 5

        return np.concatenate((h_e, 2*h_i))


    def weight_update(self, weights, rates):
        # Get the firing rates of the excitatory and inhibitory neurons
        r_e = rates[: self.N - self.D].reshape(self.N - self.D, 1)
        r_i = rates[self.N - self.D :].reshape(self.D, 1)

        # Update with Henning's learning rule
        updates = self.lr * (
            r_e @ r_i.T - self.rho * np.ones((self.N - self.D, 1)) @ r_i.T
        )
        weights[self.full_idx[:, 0], self.full_idx[:, 1]] += updates[
            self.IEcxns[:, 0], self.IEcxns[:, 1]
        ] - 0.1*weights[self.full_idx[:, 0], self.full_idx[:, 1]]

        return weights


    def integrate(self, weights_0, rates_0):
        # Initialize solution and weights
        soln = np.zeros((self.N, int(1 / self.h * self.T) + 1))
        soln[:, 0] = np.squeeze(rates_0, axis=1)
        weights = weights_0.copy()
        weight_traj = np.zeros((self.IEcxns.shape[0], int(self.T * 1 / self.h)))

        # Euler integration steps
        for i in range(0, int(self.T * 1 / self.h)):
            soln[:, i + 1] = soln[:, i] + self.rate_eqns(soln[:, i], weights) * self.h

            # Save the nonzero IE weights before changing them
            weight_traj[:, i] = weights[self.full_idx[:, 0], self.full_idx[:, 1]]
            # Update the weights
            if i % 1 == 0:
                weights = self.weight_update(weights, soln[:, i + 1])

        return soln, weight_traj, weights


    def run_sim(self):

        weights_0, rates_0 = self.initialize()
        soln, w_traj, w_final = self.integrate(weights_0, rates_0)

        return soln, weights_0, w_traj, w_final



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ei = solvingAI()
    soln, weights_0, w_traj, w_final = ei.run_sim()

    im = plt.imshow(weights_0)
    plt.colorbar(im)
    plt.show()
