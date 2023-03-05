"""Analysis of the Diffusion Limited Aggregation model."""


# Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from main import *

# Parameters
N = 100
seed = (N-1, N // 2)
etas = [0, 0.5, 1, 1.5, 2, 2.5]

# Visualization settings
color = matplotlib.cm.get_cmap('viridis').copy()
color.set_bad('white')
yy, xx = np.meshgrid(np.linspace(1,0, N), np.linspace(1,0,N))

def animate(i):
    """
    The animation function. It is called sequentially
    :param i:
    :return: the element to be plotted
    """
    cax.set_array(data[i*N:(i+1)*N])

# Running the DLA model with different eta values
for eta in etas:

    dla = dla_model(N, seed, eta)
    dla.solver.set_omega(1.7)

    dla.run('data/data_%s.txt'%int(eta*10))

    # Creating the animation
    data = np.loadtxt('data/data_%s.txt'%int(eta*10))
    fig, ax = plt.subplots(figsize=(7,7))
    cax = ax.pcolormesh(yy, xx, data[:N], cmap=color)

    anim = animation.FuncAnimation(fig, animate, frames=int(len(data)//N), interval=50)
    anim.save('figures/dla_%s.gif'%int(eta*10))
