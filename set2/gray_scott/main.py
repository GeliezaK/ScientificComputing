"""Class for solving the time-dependent diffusion equation."""

# Imports
import math
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.size'] = 16

class GrayScottReactionDiffusion:

    def __init__(self, N = 100, Du=0.16, Dv = 0.08, f = 0.035, k = 0.06, dt=1, dx = 1):
        """
        Initialize diffusion object.

        Parameters
        ----------
        M : int
            number of rows
        N : int
            number of columns
        Du : float
            diffusion constant of chemical U
        Dv : float
            diffusion constant of chemical V
        f : float
            rate at which U is supplied
        k : float
            f + k is rate at which V decays
        dt : float
            length of a timestep
        """
        # Construct the initial grid
        self.grid = np.zeros((N, N, 2)) # u = 0, v = 1
        self.grid[:, :, 0] = np.full((N, N), 0.5) # u = 0.5 everywhere
        mid = math.floor(N/2)
        centerl = math.floor(N/8)
        self.grid[mid-centerl: mid+centerl, mid-centerl:mid+centerl, 1] = np.full((2*centerl, 2*centerl), 0.25)
        # Add noise +- 0.01 to the concentrations
        for i in range(N):
            for j in range(N):
                self.grid[i, j, 0] = self.grid[i, j, 0] + self.grid[i, j, 0] * 0.01 * np.random.normal()
                self.grid[i, j, 1] = self.grid[i, j, 1] + self.grid[i, j, 1] * 0.01 * np.random.normal()
        self.N = N
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        self.dt = dt
        self.dx = dx

    def noise(self, param):
        """
        Add dynamic noise to the parameter. Returns the altered parameter.
        """
        noisy_param = param
        if param >= 1:
            noisy_param += np.sqrt(param) * np.random.normal()
        else :
            noisy_param += param ** 2 * np.random.normal()
        return noisy_param

    def update_c(self):
        """
        Update the concentration at every gridpoint. Assume cyclic boundaries in both directions. Stores the new
        concentrations in self.grid and the previous concentrations in self.prev_grid.
        """
        new_grid = np.zeros((self.N, self.N, 2))
        for i in range(self.N):
            # Determine neighbor rows
            i_up = i - 1
            i_down = i + 1
            # First and last row are neighbors
            if i == 0:
                i_up = self.N -1
            elif i == self.N - 1:
                i_down = 0

            for j in range(self.N):
                # Determine neighbor columns
                j_left = j - 1
                j_right = j + 1
                # First and last column are neighbors
                if j == 0:
                    j_left = self.N -1
                elif j == self.N -1:
                    j_right = 0

                for chem in range(2):
                    # Calculate change through diffusion
                    D = self.Du
                    if chem == 1:
                        D = self.Dv
                    D = self.noise(D)
                    factor = self.grid[i_down, j, chem] + self.grid[i_up, j, chem] + self.grid[i, j_left, chem] + \
                             self.grid[i, j_right, chem] - 4 * self.grid[i, j, chem]

                    diffusion_t = self.grid[i, j, chem] + (D * self.dt) / (self.dx ** 2) * factor
                    # Calculate change through reaction
                    u = self.grid[i, j, 0]
                    v = self.grid[i, j, 1]
                    f = self.noise(self.f)
                    k = self.noise(self.k)
                    if chem == 0:
                        reaction_t = - u * v ** 2 + f * (1 - u)
                    elif chem == 1:
                        reaction_t = + u * v ** 2 - (f + k) * v
                    # Update new concentration
                    new_grid[i, j, chem] = diffusion_t + reaction_t
                    # if i == 3 and j == 4:
                    #     print(f"\nCell (3,4) chem {chem}:{np.round(self.grid[i,j,chem],5)}")
                    #     print(f"factor:{np.round(factor,3)}, D:{D}, diffusion_t:{np.round(diffusion_t,3)}, reaction_t:{np.round(reaction_t,3)}")
                    #     print(f"new values: chem {chem}:{np.round(new_grid[i, j, 1],3)}")

        # Update grid
        self.grid = new_grid


if __name__ == '__main__':
    Du = 0.01
    Dv = 0.005
    f = 0.035
    k = 0.06
    gray_scott = GrayScottReactionDiffusion(10, Du= Du, Dv= Dv, f=f, k = k)
    print(np.round(gray_scott.grid[:, :, 0], 3))
    print(np.round(gray_scott.grid[:, :, 1], 3))
    nit = 201
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Concentration of U", fontsize=18)
    count = 1
    for i in range(nit):
        gray_scott.update_c()
        if i % 40 == 0:
            # TODO: store data and plot afterwards
            ax = plt.subplot(2, 3, count)
            sns.heatmap(gray_scott.grid[:, :, 0])
            plt.title(f"{i} iterations")
            count += 1
    plt.savefig(f"U-conc-{nit}-Du{Du}-Dv{Dv}-f{f}-k{k}.png", dpi=300)
    plt.show()

    #
    # plt.figure(figsize=(15, 12))
    # plt.subplots_adjust(hspace=0.5)
    # plt.suptitle("Concentration of V", fontsize=18)
    # count = 1
    # for i in range(nit):
    #     gray_scott.update_c()
    #     if i % 20 == 0:
    #         ax = plt.subplot(2, 3, count)
    #         sns.heatmap(gray_scott.grid[:, :, 1], vmax=0.3, vmin=0.0)
    #         plt.title(f"{i} iterations")
    #         count += 1
    # plt.savefig(f"V-conc-{nit}.png", dpi=300)
    # plt.show()





