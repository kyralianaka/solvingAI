class solvingAI:


    def __init__(self):
        pass


    def initialize(self):
        # Nischal
        pass


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
