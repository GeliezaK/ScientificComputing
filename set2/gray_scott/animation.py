import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
from main import GrayScottReactionDiffusion

chem = 1
num_it = 100
fig = plt.figure()
gray_scott = GrayScottReactionDiffusion(100)
initial_grid = gray_scott.grid[:, :, chem]
sns.heatmap(initial_grid)

def init():
    sns.heatmap(initial_grid, cbar=False)

def animate(i):
    gray_scott.update_c()
    data = gray_scott.grid[:, :, chem]
    sns.heatmap(data, cbar = False)

# Call the animator. blit=True means only re-draw the parts that have changed to save time
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_it, interval=20, repeat=False)
anim.save(f'animations/V-{num_it}.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


