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
'''
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
'''
# Plot the final state of the growth model
data_0_0 = np.loadtxt('data/data_0.txt')[-N:]
data_0_5 = np.loadtxt('data/data_5.txt')[-N:]
data_1_0 = np.loadtxt('data/data_10.txt')[-N:]
data_1_5 = np.loadtxt('data/data_15.txt')[-N:]
data_2_0 = np.loadtxt('data/data_20.txt')[-N:]
data_2_5 = np.loadtxt('data/data_25.txt')[-N:]

data = [data_0_0, data_0_5, data_1_0, data_1_5, data_2_0, data_2_5]
etas = [0, 0.5, 1, 1.5, 2, 2.5]

fig, axs = plt.subplots(2, 3, figsize = (14,7), sharex=True, sharey=True)
counter = 0

for j in range(2):    
    for i in range(3):
        im = axs[j%2][i%3].pcolormesh(yy, xx, data[counter], cmap=color)
        axs[j%2][i%3].set_title('$\eta$ = %s'%etas[counter], fontsize = 18)
        counter += 1

for ax in axs.flat:
    ax.set_xlabel('x', fontsize = 18)
    ax.set_ylabel('y', fontsize = 18)
    ax.label_outer()
    ax.set_aspect('equal')

plt.tight_layout()

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.9, axs[1][2].get_position().y0, 0.025, axs[0][2].get_position().y1 - axs[1][2].get_position().y0])

fig.colorbar(im, cax=cbar_ax)
plt.savefig('figures/dla_eta.png')
