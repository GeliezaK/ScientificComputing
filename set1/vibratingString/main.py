import numpy as np
import matplotlib.pyplot as plt


inp1 = lambda x : np.sin(2*np.pi * x)
inp2 = lambda x : np.sin(5*np.pi * x)
inp3 = lambda x : [np.sin(5*np.pi * xi) if 0.2 < xi < 0.4 else 0 for xi in x]


class String:

    def __init__(self, L, N, c, init_position):
        assert np.isclose(init_position[0], 0), 'String position should be 0 at  x=0'
        assert np.isclose(init_position[0], 0), 'String position should be 0 at  x=L'
        self.L = L
        self.N = N
        self.c = c
        self.dx = L/N
        self.state = np.array(init_position, dtype=float)
        self.prev_state = np.array(init_position, dtype=float)
        self.time_elapsed = 0

    def step(self, dt):
        """
        Calculate the new position after one timestep of size dt.
        :param dt: Length of the timestep.
        :return:
        """
        length = len(self.state)
        new_state = np.zeros(length, dtype=float)
        for ind, y in enumerate(self.state[1:-1]):
            # Only iterate over "inner" x-points because boundaries are fixed
            ind_left = ind - 1
            ind_right = ind + 1

            ctx = self.c ** 2 * (dt**2 / self.dx ** 2)
            xdev = self.state[ind_right] + self.state[ind_left] - 2 * self.state[ind]
            new_state[ind] = ctx * xdev - self.prev_state[ind] + 2 * self.state[ind]
        self.prev_state = self.state
        self.state = new_state
        self.time_elapsed += dt





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    L = 1
    N = 200
    dt = 0.001
    num_it = 1000
    x = np.arange(0, L + L/N, L/N)
    initial_position = inp3(x)
    string = String(L, N, 1, init_position=initial_position)
    colors = plt.cm.viridis(np.linspace(0, 1, num_it))
    for i in range(num_it):
        string.step(dt)
        plt.plot(x, string.state, color=colors[i])
    plt.plot(x, initial_position, color="black", label=r"$\sin(5 \pi x)$" f"\nif 0.2 < x < 0.4")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Altitude")
    plt.title(f"Result after {num_it} iterations \ndt={dt}, N={N}")
    plt.savefig("figures/inp3-colored-plot-numit1000.png")
    plt.show()




