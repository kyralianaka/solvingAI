import numpy as np



class solvingAI:


    def __init__(self):

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

    def initialize(self):
        # Nischal
        pass


    def rate_eqns(self, y, w):
        return -y + w @ self.activate(y)


    def weight_update(self):
        # Nischal
        pass


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



    def run_sim(self):

        weights_0, rates_0 = self.initialize()

        soln, weights = self.integrate(weights_0, rates_0)

        return soln, weights
