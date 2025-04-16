import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

B = 1.5
N = 0.2
M = 3
def noisy_multimodal_f(x,y):
    np.random.seed(42)
    x, y = x, y
    value = 0
    for m in range(1, M + 1):
        value += np.sin(m * x) * np.cos(m * y)
    noise = np.random.normal(0, N)
    return value + noise

x = np.linspace(-10, 2, 5)
y = np.linspace(-10, 2, 5)
X, Y = np.meshgrid(x, y)
Z = noisy_multimodal_f(X, Y)

plt.figure(figsize=(10, 8))
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('График noisy_multimodal_f_bfgs.txt')

data = np.loadtxt('noisy_multimodal_f_bfgs.txt', skiprows=1)
trajectory_x = data[:, 0]
trajectory_y = data[:, 1]

plt.plot(trajectory_x, trajectory_y, 'ro-', linewidth=2, markersize=5, label='Траектория')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

traj_z = noisy_multimodal_f(trajectory_x, trajectory_y)
ax.plot(trajectory_x, trajectory_y, traj_z, 'r.-', linewidth=2, markersize=5, label='Траектория')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('3D график функции noisy_multimodal_f_bfgs.txt и траектория метода')
plt.legend()
plt.show()