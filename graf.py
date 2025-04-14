import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

B = 1.5

def f3(x, y):
    return x**2 + y**2

x = np.linspace(-10, 6, 10)
y = np.linspace(-10, 6, 10)
X, Y = np.meshgrid(x, y)
Z = f3(X, Y)

plt.figure(figsize=(10, 8))
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('График f3_Armijo')

data = np.loadtxt('f1_1_Armijo.txt', skiprows=1)
trajectory_x = data[:, 0]
trajectory_y = data[:, 1]

plt.plot(trajectory_x, trajectory_y, 'ro-', linewidth=2, markersize=5, label='Траектория')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

traj_z = f3(trajectory_x, trajectory_y)
ax.plot(trajectory_x, trajectory_y, traj_z, 'r.-', linewidth=2, markersize=5, label='Траектория')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f3(x, y)')
ax.set_title('3D график функции f3 и траектория метода')
plt.legend()
plt.show()