# Imports
import numpy as np
import matplotlib.pyplot as plt

from main import laplace_solver

#######################################
# Comparison with analytical solution #
#######################################
"""
N = 50
system = laplace_solver(N)

# Jacobi iteration method
system.solve()
sol_jacobi = system.grid[:,0]

# Gauss-seidel iteration method
system.construct_grid()
system.solve(method = 'gauss-seidel')
sol_gs = system.grid[:,0]

# SOR iteration method
system.construct_grid()
system.set_omega(1.9)
system.solve(method='sor')
sol_sor = system.grid[:,0]

# Analytical solution
exact = []
y = np.linspace(0, 1, 1000)
for pos in y:
    exact.append(system.exact_sol(pos, 100))

# Plot the different solutions
plt.figure(figsize=(10,7))

plt.plot(np.linspace(0,1, N), sol_jacobi[::-1], color = 'red', label = 'Jacobi iteration')
plt.plot(np.linspace(0,1, N), sol_gs[::-1], color = 'blue', label = 'Gauss-Seidel iteration')
plt.plot(np.linspace(0,1, N), sol_sor[::-1], color = 'goldenrod', label = 'SOR iteration')
plt.plot(y, exact, color = 'black', label = 'Analytical solution', linestyle = 'dashed')

plt.xlabel('y', fontsize = 16)
plt.ylabel('concentration', fontsize = 16)
plt.legend()
plt.savefig('figures/compare_to_exact.png')

###################################################
# Number of iterations as a function of tolerance #
###################################################

tolerance = [1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8]

n_iter_jacobi = []
n_iter_gs = []
n_iter_sor_1_5 = []
n_iter_sor_1_9 = []

for d in tolerance:
    system.delta = d

    system.construct_grid()
    n_iter_jacobi.append(system.solve())

    system.construct_grid()
    n_iter_gs.append(system.solve(method='gauss-seidel'))

    system.construct_grid()
    system.set_omega(1.5)
    n_iter_sor_1_5.append(system.solve(method='sor'))

    system.construct_grid()
    system.set_omega(1.9)
    n_iter_sor_1_9.append(system.solve(method='sor'))

plt.figure()

plt.semilogx(tolerance, n_iter_jacobi, color = 'red', label = 'Jacobi iteration', marker='o')
plt.semilogx(tolerance, n_iter_gs, color = 'blue', label = 'Gauss-Seidel iteration', marker='x')
plt.semilogx(tolerance, n_iter_sor_1_5, color = 'goldenrod', label = 'SOR iteration ($\omega$ = 1.5)', marker='s')
plt.semilogx(tolerance, n_iter_sor_1_9, color = 'black', label = 'SOR iteration ($\omega$ = 1.9)', marker='^')

plt.legend()
plt.xlabel('Tolerance', fontsize = 16)
plt.ylabel('Iterations', fontsize = 16)
plt.savefig('figures/convergence.png')

#######################################
# Determine the optimal omega for SOR #
#######################################

sizes = [10, 30, 50]
omegas = np.linspace(1.5, 1.99, 50)
iterations = []
optimal_omega = []

for N in sizes:
    system = laplace_solver(N)
    n_iter = []

    for omega in omegas:
        system.omega = omega
        system.construct_grid()
        n_iter.append(system.solve(method='sor'))

    # Store results
    optimal_omega.append(omegas[np.argwhere(n_iter == np.min(n_iter))[0,0]])   
    iterations.append(n_iter)

fig, axs = plt.subplots(1,2, figsize = (14,5))

axs[0].plot(omegas, iterations[0], color='black', label = 'N = 10')
axs[0].plot(omegas, iterations[1], color = 'red', label = 'N = 30')
axs[0].plot(omegas, iterations[2], color = 'blue', label = 'N = 50')
axs[0].scatter(omegas[np.argwhere(iterations[0] == np.min(iterations[0]))[0,0]], iterations[0][np.argwhere(iterations[0] == np.min(iterations[0]))[0,0]], color = 'black')
axs[0].scatter(omegas[np.argwhere(iterations[1] == np.min(iterations[1]))[0,0]], iterations[1][np.argwhere(iterations[1] == np.min(iterations[1]))[0,0]], color = 'red')
axs[0].scatter(omegas[np.argwhere(iterations[2] == np.min(iterations[2]))[0,0]], iterations[2][np.argwhere(iterations[2] == np.min(iterations[2]))[0,0]], color = 'blue')
axs[0].legend()

axs[1].plot(sizes, optimal_omega, marker = 'o', color = 'black')

axs[0].set_xlabel('$\omega$', fontsize = 16)
axs[0].set_ylabel('iterations', fontsize = 16)
axs[1].set_xlabel('N', fontsize = 16)
axs[1].set_ylabel('$\omega_{opt}$', fontsize = 16)

plt.tight_layout()
plt.savefig(('figures/opt_omega.png'))
"""
##################
# Adding objects #
##################

N = 50

# Defining rectangle and triangle
rectangle = []

for j in range(8, 16):
    for i in range(22, 28):
        rectangle.append((j,i))

triangle = []
center = 25

for j in range(8, 16, 2):
    # Edges
    length = (j - 8)//2
    triangle.append((j,center - length))
    triangle.append((j+1,center - length))
    if length != 0:
        triangle.append((j, center + length))
        triangle.append((j+1, center + length))
    
    # Bottom row
    if j == 14:
        print(center - length)
        for i in range(center - length + 1, center + length):
            triangle.append((j+1, i))

omegas = np.linspace(1.5, 1.99, 50)

system0 = laplace_solver(N)

system1 = laplace_solver(N)
system1.set_objects(rectangle)

system2 = laplace_solver(N)
system2.set_objects(triangle)

n_iter = [[],[],[]]

for omega in omegas:
    system0.omega = omega
    system1.omega = omega
    system2.omega = omega

    system0.construct_grid()
    system1.construct_grid()
    system2.construct_grid()

    n_iter[0].append(system0.solve(method='sor'))
    n_iter[1].append(system1.solve(method='sor'))
    n_iter[2].append(system2.solve(method='sor'))

fig, axs = plt.subplots(figsize = (10,7))

axs.plot(omegas, n_iter[0], color='black', label = 'No objects')
axs.plot(omegas, n_iter[1], color = 'red', label = 'Rectangle')
axs.plot(omegas, n_iter[2], color = 'blue', label = 'Triangle')
axs.scatter(omegas[np.argwhere(n_iter[0] == np.min(n_iter[0]))[0,0]], n_iter[0][np.argwhere(n_iter[0] == np.min(n_iter[0]))[0,0]], color = 'black')
axs.scatter(omegas[np.argwhere(n_iter[1] == np.min(n_iter[1]))[0,0]], n_iter[1][np.argwhere(n_iter[1] == np.min(n_iter[1]))[0,0]], color = 'red')
axs.scatter(omegas[np.argwhere(n_iter[2] == np.min(n_iter[2]))[0,0]], n_iter[2][np.argwhere(n_iter[2] == np.min(n_iter[2]))[0,0]], color = 'blue')
axs.legend()

axs.set_xlabel('$\omega$', fontsize = 16)
axs.set_ylabel('iterations', fontsize = 16)

plt.tight_layout()
plt.savefig('figures/objects_omega.png')

yy, xx = np.meshgrid(np.linspace(1,0, N), np.linspace(1,0,N))

fig, axs = plt.subplots(1, 3, figsize = (21,7), sharex=True, sharey=True)

axs[0].pcolormesh(yy, xx, system0.grid, cmap='Reds')
axs[1].pcolormesh(yy, xx, system1.grid, cmap='Reds')
axs[2].pcolormesh(yy, xx, system2.grid, cmap='Reds')

plt.savefig('figures/objects-2D.png')
