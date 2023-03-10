"""Class for simulating the diffusion limited aggregation growht model."""

# Imports
import numpy as np
from laplace import *

class dla_model:

    def __init__(self, N, seed, eta=1):
        """
        Initialize a diffusion limited aggregation model.

        Parameters
        ----------
        N : int
            size of the grid (N by N)
        seed : tuple
            coordinate of the seed of the object
        eta : float
            parameter that determines the final shape of the object, default 1
        """
        self.seed = seed
        self.N = N
        # Create a laplace solver object
        self.solver = laplace_solver(N)
        self.solver.set_objects([seed])
        self.iterative_method = 'sor'
        
        # Parameter that determines the shape of the object
        self.eta = eta
        # Counter for the total number iterations
        self.tot_iter = 0
    
    def reset(self):
        """Reset the model."""
        self.solver.construct_grid()
        self.solver.objects = [self.seed]
        self.tot_iter = 0
    
    def set_iterative_method(self, method):
        """
        Change the iterative method to solve the Laplace equation.

        Parameters
        ----------
        method : str
            method used in the iterative scheme
        """
        if method != 'jacobi' and method != 'gauss-seidel' and method != 'sor':
            raise ValueError('Unknown iterative method, %s. Options are "jacobi", "gauss-seidel" or "sor".' %method)
        self.iterative_method = method
    
    def select_candidate(self, candidates):
        """
        Select one of the growth candidates to be part of the object.

        Parameters
        ----------
        candidates : list
            concentrations at the position of all the candidates
        
        Returns
        -------
        pos : int
            index of the chosen candidate
        """
        # Denominator in the equation for the growth probability
        candidates = np.round(candidates, 5)
        candidates = np.array([np.maximum(i, 0) for i in candidates])
        total = np.sum(candidates ** self.eta)

        # Calculate the probabilities of all the candidates
        cdf = [(candidates[0] ** self.eta) / total]
        for coord in candidates[1:]:
            p = (coord ** self.eta) / total
            cdf.append(cdf[-1] + p)
        
        # Choose a candidate
        pos = 0
        u = np.random.uniform()
        while u > cdf[pos]:
            pos += 1
        
        return pos

    def growth_step(self):
        """
        Growth step of the DLA model.
        
        Returns
        -------
        result : int
            0 if succesfull, 1 otherwise
        """
        # Extract the object and grid
        obj = self.solver.objects
        grid = self.solver.grid
        # Lists for the coordinates and concentration of the growth candidates
        coords = []
        conc = []

        # Determine all the growth candidates
        dx = [0, 0, 1, -1]
        for pos in obj:
            # Check 4 neighbours
            for i in range(4):
                n = (pos[0] + dx[i], (pos[1] + dx[3-i]) % self.N)
                if n[0] < 0:
                    n = (0, n[1])
                elif n[0] == self.N:
                    n = (self.N - 1, n[1])
                # Check if not already in object
                if n not in obj and n not in coords:
                    coords.append(n)
                    conc.append(grid[n[0], n[1]])
        
        # Terminate the process if there are no candidates
        if len(coords) == 0:
            return 1
        # Determine which cell gets added to the object
        pos = self.select_candidate(conc)
        # Terminate the process if no good candidate is found
        if pos == len(coords):
            return 1
        # Terminate if the process has reached the top row
        if coords[pos][0] == 0:
            return 1
        # Add cell to the object
        self.solver.add_object(coords[pos])
        self.solver.grid[coords[pos][0], coords[pos][1]] = 0

        return 0
                
    def step(self):
        """
        Single step in the DLA model.
        
        Returns
        -------
        result : int
            0 if growth was succesfull, 1 otherwise
        """
        # Diffusion
        self.tot_iter += self.solver.solve(self.iterative_method)
        # Growth
        result = self.growth_step()

        return result

    def store_grid(self, filename):
        """
        Store the grid in a txt file.

        Parameters
        ----------
        filename : str
            name of the file to store the grid
        """
        grid = np.copy(self.solver.grid)
        for coord in self.solver.objects:
            grid[coord[0], coord[1]] = np.nan

        f = open(filename, 'a')
        np.savetxt(f, grid)
        f.write('\n')
        f.close()
    
    def run(self, file=None, max_iter=500):
        """
        Run the DLA model for n_iter steps.

        Parameters
        ----------
        file : str
            name of the file to store the grid, default None
        max_iter : int
            maximum number of steps to run the model, default 5000
        """
        result = 0
        iter = 0
        while result == 0:
            result = self.step()
            if file is not None:
                self.store_grid(file)
            if iter > max_iter:
                break
            iter += 1
