import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
from main import GrayScottReactionDiffusion

chem = 1
chem_label = "V" if chem == 1 else "U"
Du = 0.1
Dv = 0.05
f = 0.0545
k = 0.062
N = 300
nit = 500
fig = plt.figure()
gray_scott = GrayScottReactionDiffusion(N, Du = Du, Dv=Dv, f=f, k=k)
initial_grid = gray_scott.V_grid
sns.heatmap(initial_grid, vmin = 0, vmax = 0.5)
plt.suptitle(f"Concentration of {chem_label} \n" rf"$D_u = {Du}, D_v = {Dv}, f = {f}, k = {k}, t = {nit}, dt = 1$",
             fontsize=18)

def init():
    sns.heatmap(initial_grid, cbar=False, vmin = 0, vmax = 0.5)

def animate(i):
    # only plot every 40 iterations
    for k in range(40):
        gray_scott.update_c()
    sns.heatmap(gray_scott.V_grid, cbar = False, vmin = 0, vmax= 0.5)

# Call the animator. blit=True means only re-draw the parts that have changed to save time
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nit, interval=20, repeat=False)
anim.save(f'animations/{chem_label}-N{N}-it{nit}-Du{Du}-Dv{Dv}-f{f}-k{k}.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


