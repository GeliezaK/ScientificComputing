# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from main import diffusion

# Settings
N = 40
D = 1
dt = 0.0001

simulation = diffusion(N, D, dt)
simulation.run(1, 'data.txt')

# Read out the data
data = np.loadtxt('data.txt')

#######################################
# Comparison with analytical solution #
#######################################

# Extract the numerical solution as a function of y at different timesteps
num_0_01 = data[:N][:,0]
num_0_1 = data[9*N:10*N][:,0]
num_1_0 = data[99*N:][:,0]

# Calculate the analytical solution at the different timesteps
exact_0_01 = []
exact_0_1 = []
exact_1_0 = []

y = np.linspace(0, 1, 1000)
for pos in y:
    exact_0_01.append(simulation.exact_sol(pos, 0.01))
    exact_0_1.append(simulation.exact_sol(pos, 0.1))
    exact_1_0.append(simulation.exact_sol(pos, 1))

# Plot the numerical and analytical solution
plt.figure(figsize=(10,7))

plt.plot(y, exact_0_01, color = 'black', label = 't = 0.01 (analytical)')
plt.plot(y, exact_0_1, color = 'blue', label = 't = 0.1 (analytical)')
plt.plot(y, exact_1_0, color = 'red', label = 't = 1.0 (analytical)')
plt.scatter(np.linspace(0,1, N), num_0_01[::-1], color = 'black', label = 't = 0.01 (numerical)')
plt.scatter(np.linspace(0,1, N), num_0_1[::-1], color = 'blue', label = 't = 0.1 (numerical)')
plt.scatter(np.linspace(0,1, N), num_1_0[::-1], color = 'red', label = 't = 1.0 (numerical)')
plt.xlabel('y')
plt.ylabel('concentration')
plt.legend()
plt.savefig('figures/1D-plot.png')
plt.show()

##################################
# 2D plot at different timesteps #
##################################

simulation = diffusion(N, D, dt)
data_0_0 = simulation.grid
simulation.run(0.001)
data_0_001 = simulation.grid

data_0_01 = data[:N]
data_0_1 = data[9*N:10*N]
data_0_2 = data[19*N:20*N]
data_1_0 = data[99*N:]

yy, xx = np.meshgrid(np.linspace(1,0, N), np.linspace(1,0,N))

fig, axs = plt.subplots(2,3, figsize = (14,7), sharex=True, sharey=True)

im = axs[0][0].pcolormesh(yy, xx, data_0_0, cmap='Reds')
axs[0][1].pcolormesh(yy, xx, data_0_001, cmap='Reds')
axs[0][2].pcolormesh(yy, xx, data_0_01, cmap='Reds')
axs[1][0].pcolormesh(yy, xx, data_0_1, cmap='Reds')
axs[1][1].pcolormesh(yy, xx, data_0_2, cmap='Reds')
axs[1][2].pcolormesh(yy, xx, data_1_0, cmap='Reds')

axs[0][0].set_title('t = 0')
axs[0][1].set_title('t = 0.001')
axs[0][2].set_title('t = 0.01')
axs[1][0].set_title('t = 0.1')
axs[1][1].set_title('t = 0.2')
axs[1][2].set_title('t = 1')

for ax in axs.flat:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.label_outer()
    ax.set_aspect('equal')

plt.tight_layout()

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.9, axs[1][2].get_position().y0, 0.025, axs[0][2].get_position().y1 - axs[1][2].get_position().y0])

fig.colorbar(im, cax=cbar_ax)
plt.savefig('figures/2D-plot.png')
plt.show()

#################
# Animated plot #
#################

fig, ax = plt.subplots(figsize=(7,7))
cax = ax.pcolormesh(yy, xx, data[:N], cmap='Reds')

def animate(i):
    """
    The animation function. It is called sequentially
    :param i:
    :return: the element to be plotted
    """
    cax.set_array(data[i*N:(i+1)*N])

anim = animation.FuncAnimation(fig, animate, frames=int(len(data)/N))
anim.save('figures/diffusion.gif')
plt.show()


