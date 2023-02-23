import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from main import String, inp1, inp2, inp3

# Set up figure, axis and plot element to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
line, = ax.plot([], [], lw=2)
plt.xlabel("x")
plt.ylabel("Amplitude")

# Initialize animation data
L = 1
N = 500
dt = 0.001
num_it = 1000
x = np.arange(0, L + L / N, L / N)
initial_position = inp3(x)
string = String(L, N, 1, init_position=initial_position)


def init():
    """
    Initialize function for animation. Plot the background of each frame
    :return: the element to be plotted
    """
    line.set_data([], [])
    return line,


def animate(i):
    """
    The animation function. It is called sequentially
    :param i:
    :return: the element to be plotted
    """
    string.step(dt)
    line.set_data(x, string.state)
    return line,


# Call the animator. blit=True means only re-draw the parts that have changed to save time
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_it, interval=10, blit=True)
anim.save('animations/string_inp3_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
