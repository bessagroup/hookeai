
import torch

print('LOSS - APPROACH 1')

loss_1 = torch.nn.MSELoss()

input_1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

target_1 = torch.tensor([2.0, 3.0, 4.0])

total_loss_1 = 0

total_loss_1 += loss_1(input_1, target_1)
print('step 1 - total_loss_1 = ', total_loss_1)

total_loss_1 += loss_1(input_1, target_1)
print('step 2 - total_loss_1 = ', total_loss_1)

print('final - total_loss_1 = ', total_loss_1)
print('final - backward = ', input_1.grad)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('LOSS - APPROACH 2')

input_1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

target_1 = torch.tensor([2.0, 3.0, 4.0])

total_loss_1 = 0

loss_1 = torch.nn.MSELoss()
total_loss_1 += loss_1(input_1, target_1)
print('step 1 - total_loss_1 = ', total_loss_1)

loss_1 = torch.nn.MSELoss()
total_loss_1 += loss_1(input_1, target_1)
print('step 2 - total_loss_1 = ', total_loss_1)

print('final - total_loss_1 = ', total_loss_1)
print('final - backward = ', input_1.grad)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('BACKWARD - APPROACH 1')

x = torch.ones(2, 2, requires_grad=True)

out = 0

for i in range(3):
    y = x + 2
    z = y * y * 3
    out += z.mean()

out.backward()

print(x.grad)


import os

path = '/home/User/Documents/text.txt'
path2 = '/home/User/Documents/'

dirname = os.path.dirname(path)

print(dirname)
print(os.path.normpath(path2))
print(os.path.basename(path))



key = 'train'
val = 0.112311111111111111111
print(f'Part size must be contained between 0 and 1. '
      f'Check part: {key} ({val:.2f})')

import numpy as np

parts_sizes = [0.2, 0.8]

print(np.isclose(np.sum(parts_sizes), 1.0))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\n\n')

# example of a standardization
from numpy import asarray
from sklearn.preprocessing import StandardScaler

# define data
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
print(data)


# define standard scaler
scaler = StandardScaler()
scaler_inc = StandardScaler()


# transform data
scaled = scaler.fit_transform(data)
print(scaled)


denormalized_data = scaler.inverse_transform(scaled)

print(denormalized_data)



for i in range(data.shape[0]):
    scaler_inc.partial_fit(data[i, :].reshape(-1, 2))
    
scaled_inc = scaler_inc.transform(data)
print(scaled_inc)



scaled = scaler.transform(data)
print(scaled)


def fun1():
    fun2()
    
def fun2():
    print('hey')
    
fun1()



from material_patch.patch_generator import rotation_tensor_from_euler_angles


p1 = np.array([0.0, 1.0])
p2 = np.array([0.0, 0.0])


euler_deg = (-90, 0, 0)
rotation = rotation_tensor_from_euler_angles(euler_deg)[:2, :2]

p1_rot = np.matmul(rotation, p1)
p2_rot = np.matmul(rotation, p2)

print(p1, p2, ' -> ', p1_rot, p2_rot)

print('\nTest rotation:')
p1 = np.array([2.0, 1.0])
p2 = np.array([3.0, 2.0])
p1p2 = p2 - p1
p1p2 = (1.0/np.linalg.norm(p1p2))*p1p2

rotation = rotation_tensor_from_euler_angles((-90, 0, 0))[:2, :2]
ort = np.matmul(rotation, p1p2)
print('ort = ', ort)

rotation = np.zeros((2, 2))
rotation[:, 0] = p1p2
rotation[:, 1] = ort
rotation = np.transpose(rotation)



v = np.array([1.0, 0.0])
#v = p1p2

print('rot_v = ', np.matmul(rotation, v))
print('rot_p1 = ', np.matmul(rotation, p1))
print('rot_p2 = ', np.matmul(rotation, p2))
print('rot_p2 - rot_p1 = ', np.matmul(rotation, p1p2))
print('ort = ', ort)



translation = np.matmul(rotation, p1)
local_p1 = np.matmul(rotation, p1) - translation
print('local_p1 = ', local_p1)


translation = -p1
print('p1 = ', np.matmul(np.transpose(rotation), local_p1) - translation)



v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 1.0])
v3 = np.cross(v2, v1)
print(v3)


import numpy as np
from shapely.geometry import Polygon, LinearRing

points = np.array([
    (5,0),
    (6,4),
    (4,5),
    (1,5),
    (1,0)
])

P = Polygon(points)

import matplotlib.pyplot as plt

x,y = P.exterior.coords.xy
plt.plot(x,y)
plt.axis('equal')
plt.grid()
plt.show()

print(type(P.exterior))
print(P.exterior)
print(LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]))
print('is counterclockwise?', P.exterior.is_ccw)


points = np.flipud(points)
P1 = Polygon(points)
print('is counterclockwise?', P1.exterior.is_ccw)