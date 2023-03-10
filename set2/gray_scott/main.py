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

    def __init__(self, N=100, Du=0.16, Dv=0.08, f=0.035, k=0.06, dt=1, dx=1):
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
        # u = 0.5 everywhere
        self.U_grid = np.full((N + 2, N + 2), 0.5)  # two halo-rows for faster computation
        # v = 0.25 in small center square and v=0 everywhere else
        self.V_grid = np.zeros((N + 2, N + 2))
        mid = math.floor(N / 2)
        centerl = math.floor(N / 10)
        self.V_grid[mid - centerl: mid + centerl, mid - centerl:mid + centerl] = np.full((2 * centerl, 2 * centerl),
                                                                                         0.25)
        # Add noise +- 1 % to the concentrations
        self.U_grid[1:-1, 1:-1] = self.U_grid[1:-1, 1:-1] + self.U_grid[1:-1, 1:-1] * 0.01 * np.random.normal(
            size=(N, N))
        self.V_grid[1:-1, 1:-1] = self.V_grid[1:-1, 1:-1] + self.V_grid[1:-1, 1:-1] * 0.01 * np.random.normal(
            size=(N, N))
        self.update_ghost_cells(self.U_grid)
        self.update_ghost_cells(self.V_grid)
        self.N = N
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        self.dt = dt
        self.dx = dx

    def update_ghost_cells(self, grid):
        grid[0, :] = grid[-2, :]
        grid[-1, :] = grid[1, :]
        grid[:, 0] = grid[:, -2]
        grid[:, -1] = grid[:, 1]
        return grid

    def diffusion_factor(self, grid):
        """
        Compute the second order spatial derivative of the concentration in finite differences. Return the factor.
        """
        return (grid[:-2, 1:-1] +
                grid[1:-1, :-2] - 4 * grid[1:-1, 1:-1] + grid[1:-1, 2:] +
                +   grid[2:, 1:-1])

    def update_c(self):
        """
        Update the concentration at every gridpoint. Assume cyclic boundaries in both directions. Stores the new
        concentrations in self.grid and the previous concentrations in self.prev_grid.
        """
        # Extract true grids without ghost cells
        U = self.U_grid[1:-1, 1:-1]
        V = self.V_grid[1:-1, 1:-1]

        # Update whole grids in one go
        diffusion_U = self.diffusion_factor(self.U_grid)
        diffusion_V = self.diffusion_factor(self.V_grid)

        UVV = U * V * V
        f = self.f + self.f * 0.01 * np.random.normal()
        k = self.k + self.k * 0.01 * np.random.normal()
        U += (self.Du * self.dt) / (self.dx ** 2) * diffusion_U - UVV + f * (1 - U)
        V += (self.Dv * self.dt) / (self.dx ** 2) * diffusion_V + UVV - (f + k) * V

        self.update_ghost_cells(self.U_grid)
        self.update_ghost_cells(self.V_grid)


if __name__ == '__main__':
    Du = 0.16
    Dv = 0.08
    f = 0.032
    k = 0.057
    N = 300
    debug =0

    plot_t_debug = [0, 10, 20, 40, 80, 160, 320, 640, 1280]
    plot_t = [0, 100, 200, 500, 1000, 2000, 4000, 8000, 16000]
    plot_its = plot_t_debug if debug else plot_t
    chem = 1
    chem_label = "V" if chem == 1 else "U"
    gray_scott = GrayScottReactionDiffusion(N=N, Du=Du, Dv=Dv, f=f, k=k)

    nit = plot_its[-1] + 1
    fig, axn = plt.subplots(3, 3, sharex=True, sharey=True)
    fig.tight_layout(rect=[0, 0, 0.9, 0.85])
    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.7])
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f"Concentration of {chem_label} \n" rf"$D_u = {Du}, D_v = {Dv}, f = {f}, k = {k}$",
                 fontsize=18)
    count = 0
    for i in range(nit):
        gray_scott.update_c()
        if i in plot_its:
            ax = axn.flat[count]
            sns.heatmap(gray_scott.V_grid, ax = ax, cbar= count == 0, vmax=0.4, vmin=0, cbar_ax=None if count else cbar_ax)
            ax.set_title(f"t = {i}", fontsize = 14)
            count += 1

    plt.savefig(f"figures/{chem_label}-N{N}-it{nit}-Du{Du}-Dv{Dv}-f{f}-k{k}.png", dpi=300)
    plt.show()
    print(np.round(gray_scott.U_grid, 3))
    print(np.round(gray_scott.V_grid, 3))
