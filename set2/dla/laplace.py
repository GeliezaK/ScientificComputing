"""Class for solving the time-independent diffusion equation."""

# Imports
import math
import numpy as np


class laplace_solver:

    def __init__(self, N, tolerance=1E-5, objects=[]):
        """
        Initialize a laplace_solver object.

        Parameters
        ----------
        N : int
            size of the grid (N by N)
        tolerance : float
            stopping criteria for the iterative scheme
        objects : list, optional
            coordinates of objects on the grid [(row_index, col_index)]
        """
        # Construct the initial grid
        self.N = N
        self.grid = None
        self.objects = objects
        self.construct_grid()

        self.delta = tolerance

        # Parameter for SOR scheme
        self.omega = 1

    def set_objects(self, coordinates):
        """
        Set objects to the grid.

        Parameters
        ----------
        coordinates : list
            coordinates of objects on the grid [(row_index, col_index)]
        """
        self.objects = coordinates
    
    def add_object(self, coordinate):
        """
        Add single gridcell as object.

        Parameters
        ----------
        coordinate : tuple
            coordinate of the gridcell (row_index, col_index)
        """
        self.objects.append(coordinate)

    def construct_grid(self):
        """Construct the initial state of the grid."""
        grid = np.zeros((self.N, self.N))
        # Source on the top row
        grid[0] = 1

        self.grid = grid

    def set_threshold(self, tolerance):
        """
        Set the stopping criteria of the iterative solver.

        Parameters
        ----------
        tolerance : float
            value of the maximal difference between two iterations
        """
        self.delta = tolerance

    def set_omega(self, omega):
        """
        Set the relaxation parameter of the SOR method.

        Parameters
        ----------
        omega : float
            new value for the relaxation parameter
        """
        self.omega = omega

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
            value += math.erfc((1 - y + 2*i) / (2 * np.sqrt(t)))
            value -= math.erfc((1 + y + 2*i) / (2 * np.sqrt(t)))
        return value

    def update_c(self, j, i):
        """
        Calculate the concentration at a specific grid point.

        Parameters
        ----------
        j : int
            row-index of the grid point
        i : int
            column-index of the grid point

        Returns
        -------
        value : float
            concentration
        """
        # Check if there is an object present on this coordinate
        if self.objects:
            if (j, i) in self.objects:
                return 0
        if i == 0:
            value = self.grid[j, i+1] + self.grid[j, -1] + self.grid[j+1, i] + self.grid[j-1, i]
        elif i == len(self.grid) - 1:
            value = self.grid[j, 0] + self.grid[j, i-1] + self.grid[j+1, i] + self.grid[j-1, i]
        else:
            value = self.grid[j, i+1] + self.grid[j, i-1] + self.grid[j+1, i] + self.grid[j-1, i]

        return value / 4

    def jacobi_step(self):
        """
        Single iteration in the Jacobi method.

        Returns
        -------
        converged : boolean
            indicates if the the solution is converged
        """
        converged = True
        # Copy the grid from the previous iteration
        new_grid = np.copy(self.grid)

        for j in range(1, len(self.grid) - 1):
            for i in range(len(self.grid)):
                new_grid[j, i] = self.update_c(j, i)
                # Check convergence
                if np.abs(new_grid[j, i] - self.grid[j, i]) > self.delta:
                    converged = False

        # Replace the old grid by the new one
        self.grid = new_grid

        return converged

    def gauss_seidel_step(self):
        """
        Single iteration in the Gauss-Seidel method.

        Returns
        -------
        converged : boolean
            indicates if the the solution is converged
        """
        converged = True

        for j in range(1, len(self.grid) - 1):
            for i in range(len(self.grid)):
                value = self.update_c(j, i)
                # Check convergence
                if np.abs(value - self.grid[j, i]) > self.delta:
                    converged = False
                self.grid[j, i] = self.update_c(j, i)

        return converged

    def sor_step(self):
        """
        Single iteration in the succesive over relaxation method.

        Returns
        -------
        converged : boolean
            indicates if the the solution is converged
        """
        converged = True

        for j in range(1, len(self.grid) - 1):
            for i in range(len(self.grid)):
                value = self.update_c(j, i) * self.omega + (1-self.omega) * self.grid[j, i]
                # Check convergence
                if np.abs(value - self.grid[j, i]) > self.delta:
                    converged = False
                self.grid[j, i] = value

        return converged

    def solve(self, method='jacobi'):
        """
        Solves the Laplace equation.

        Parameters
        ----------
        method : str, optional
            the iterative scheme that is used, default is Jacobi

        Returns
        -------
        n_iter : int
            number of iterations it took to solve the equation
        """
        converged = False
        n_iter = 0
        while converged is False:
            if method == 'jacobi':
                converged = self.jacobi_step()
            elif method == 'gauss-seidel':
                converged = self.gauss_seidel_step()
            elif method == 'sor':
                converged = self.sor_step()
            else:
                raise ValueError('Unknown method')
            n_iter += 1

        return n_iter
