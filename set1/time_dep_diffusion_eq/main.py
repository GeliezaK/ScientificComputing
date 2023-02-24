"""Class for solving the time-dependent diffusion equation."""

# Imports
import math
import numpy as np


class diffusion:

    def __init__(self, N, D=1, dt=0.001):
        """
        Initialize diffusion object.

        Parameters
        ----------
        N : int
            size of the grid (N by N)
        D : float
            diffusion constant
        dt : float
            length of a timestep
        """
        # Construct the initial grid
        self.grid = np.zeros((N, N))
        self.grid[0] = 1

        self.D = D
        self.dt = dt
        # Rows that contain insulation material
        self.insulation = []

        self.exact = []
    
    def set_insulation(self, rows):
        """
        Add insulation to specific rows.

        Parameters
        ----------
        rows : list
            list with indices of the rows that contain the insulatory material
        """
        self.insulation = rows
        
    def update_c(self, space):
        """
        Update the concentration at every gridpoint.

        Parameters
        ----------
        space : numpy.ndarray
            grid from the previous timestep

        Returns
        -------
        new_space : numpy.ndarray
            updated grid
        """
        # Copy the grid from previous timestep
        new_space = np.copy(space)
        for j in range(1, len(space) - 1):
            D = self.D
            if j in self.insulation:
                D = 0.05
            for i in range(len(space)):
                if i == 0:
                    factor = space[j, -2] + space[j, i+1] + space[j-1, i] + space[j+1, i] - 4 * space[j, i]
                elif i == len(space) - 1:
                    factor = space[j, 1] + space[j, i-1] + space[j-1, i] + space[j+1, i] - 4 * space[j, i]
                else:
                    factor = space[j, i+1] + space[j, i-1] + space[j-1, i] + space[j+1, i] - 4 * space[j, i]
                # update coordinate
                new_space[j, i] = space[j, i] + (D * self.dt) * len(space)**2 * factor

        return new_space

    def exact_sol(self, y, t, n=1000):
        """
        Calculate the exact solution.

        Parameters
        ----------
        y : float
            position in the y-direction
        t : float
            time
        n : int
            number of iterations

        Returns
        -------
        value : float
            concentration
        """
        value = 0
        for i in range(n):
            value += math.erfc((1 - y + 2*i) / (2 * np.sqrt(self.D*t)))
            value -= math.erfc((1 + y + 2*i) / (2 * np.sqrt(self.D*t)))
        return value

    def store_data(self, filename):
        """
        Store the grid in a txt file.

        Parameters
        ----------
        filename : str
            name of the file to store the grid
        """
        f = open(filename, 'a')
        np.savetxt(f, self.grid)
        f.write('\n')
        f.close()

    def run(self, time, file=None):
        """
        Run simulation of diffusion process.

        Parameters
        ----------
        time : float
            length of the simulation
        file : str, optional
            name of the file to store the grid
        """
        t = self.dt
        counter = 1

        while np.round(t, 8) <= time:
            # Calculate new concentrations
            self.grid = self.update_c(self.grid)
            # Store data (every 100 iterations)
            if counter % 100 == 0 and file is not None:
                self.store_data(file)

            t += self.dt
            counter += 1
