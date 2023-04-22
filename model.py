import numpy as np
<<<<<<< HEAD



class solvingAI:
=======
>>>>>>> origin/nisch-branch


class solvingAI:
    def __init__(self):

<<<<<<< HEAD
        # Number of neurons
        self.N = 1000
        # Fraction of excitatory neurons
        self.e = 0.8
        # Fraction of null connections
        self.k = 0.9 
        
        # Start with ReLu activation function
        self.activate = lambda x: max(0, x)

        # Integration parameters
        self.h = 0.05 # step size, 1/s
        self.T = 5 # total time, sec

        # Weight update parameters
        self.lr = 0.01 # learning rate
        self.rho = 1 # target firing rate

=======
>>>>>>> origin/nisch-branch
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

    def rate_eqns(self, y, w):
        return -y + w @ self.activate(y)

    def weight_update(self):
        # Nischal
        pass

<<<<<<< HEAD

    def integrate(self, weights_0, rates_0):
        
        # Initialize
        soln = np.zeros((self.n, self.h*self.T + 1))
        soln[:, 0] = rates_0
        weights = weights_0
        weight_traj = np.zeros((len(self.IEcxns), self.T*self.h))

        for i in range(0, self.T):
            soln[i+1] = soln[i] + self.rate_eqns(soln[i], weights)*self.h

            # Save the nonzero IE weights before changing them 
            weight_traj[:, i] = weights[self.IEcxns]
            # Update the weights
            weights = self.weight_update(weights)

        return soln

=======
    def integrate(self):
        # Nischal/Kyra
        pass
>>>>>>> origin/nisch-branch

    def run_sim(self):
<<<<<<< HEAD

        weights_0, rates_0 = self.initialize()
=======
        weights_0 = self.initialize()
>>>>>>> origin/nisch-branch

        soln, weights = self.integrate(weights_0, rates_0)

        return soln, weights
