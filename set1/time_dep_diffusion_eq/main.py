# Imports
import math
import numpy as np

class diffusion:

    def __init__(self, N, D=1, dt=0.001):

        # Construct the initial grid
        self.grid = np.zeros((N, N))
        self.grid[0] = 1

        self.D = D
        self.dt = dt

        self.exact = []
    
    def update_c(self, space):
        # Copy the grid from previous timestep
        new_space = np.copy(space)
        for j in range(1, len(space) - 1):
            for i in range(len(space)):
                if i == 0:
                    factor = space[j, -2] + space[j, i+1] + space[j-1, i] + space[j+1, i] - 4 * space[j, i]
                elif i == len(space) - 1:
                    factor = space[j, 1] + space[j, i-1] + space[j-1, i] + space[j+1, i] - 4 * space[j, i]
                else:
                    factor = space[j, i+1] + space[j, i-1] + space[j-1, i] + space[j+1, i] - 4 * space[j, i]
                # update coordinate
                new_space[j,i] = space[j,i] + (self.D * self.dt) * len(space)**2 * factor

        return new_space
    
    def exact_sol(self, y, t, n=1000):
        value = 0
        for i in range(n):
            value += math.erfc((1 - y + 2*i) / (2 * np.sqrt(self.D*t)))
            value -= math.erfc((1 + y + 2*i) / (2 * np.sqrt(self.D*t)))
        return value
    
    def store_data(self, filename):
        f = open(filename, 'a')
        np.savetxt(f, self.grid)
        f.write('\n')
        f.close()
    
    def run(self, time, file=None):
        t = self.dt
        counter = 1

        while np.round(t, 8) <= time:
            # Calculate new concentrations
            self.grid = self.update_c(self.grid)
            # Store data (every 10 iterations)
            if counter % 100 == 0 and file != None:
                self.store_data(file)

            t += self.dt
            counter += 1
