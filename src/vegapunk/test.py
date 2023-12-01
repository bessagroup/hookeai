
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
#plt.plot(x,y)
#plt.axis('equal')
#plt.grid()
#plt.show()

print(type(P.exterior))
print(P.exterior)
print(LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]))
print('is counterclockwise?', P.exterior.is_ccw)


points = np.flipud(points)
P1 = Polygon(points)
print('is counterclockwise?', P1.exterior.is_ccw)



class TestClass():
    
    def __init__(self, x):
        self._x = x
        
    def method(self):
        y = self._x
        self._x = None
        return y
    
x = torch.rand(2, 3)
test_class = TestClass(x)
y = test_class.method()

print(y)

"""
collator = torch_geometric.loader.dataloader.Collater(follow_batch=None, exclude_keys=None)
data_loader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=collator)
"""

class TorchStandardScaler:
    """PyTorch tensor standardization data scaler.
    
    Attributes
    ----------
    _n_features : int
        Number of features to standardize.
    _mean : torch.Tensor
        Features standardization mean tensor stored as a torch.Tensor with
        shape (n_features,).
    _std : torch.Tensor
        Features standardization standard deviation tensor stored as a
        torch.Tensor with shape (n_features,).
    
    Methods
    -------
    set_mean(self, mean)
        Set features standardization mean tensor.
    set_std(self, std)
        Set features standardization standard deviation tensor.    
    fit(self, tensor)
        Fit features standardization mean and standard deviation tensors.
    transform(self, tensor)
        Standardize features tensor.
    inverse_transform(self, tensor)
        Destandardize features tensor.
    _check_std(self, std)
        Check features standardization standard deviation tensor.
    """
    def __init__(self, n_features, mean=None, std=None):
        """Constructor.
        
        Parameters
        ----------
        n_features : int
            Number of features to standardize.
        mean : torch.Tensor, default=None
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        std : torch.Tensor, default=None
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        """
        self._n_features = n_features
        if mean is not None:
            self._mean = self._check_mean(mean)
        if std is not None:
            self._std = self._check_std(std)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_mean(self, mean):
        """Set features standardization mean tensor.
        
        Parameters
        ----------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        self._mean = self._check_mean(mean)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_std(self, std):
        """Set features standardization standard deviation tensor.
        
        Parameters
        ----------
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        """
        self._std = self._check_std(std)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fit(self, tensor, is_bessel=False):
        """Fit features standardization mean and standard deviation tensors.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features).
        is_bessel : bool, default=False
            Apply Bessel's correction to compute standard deviation, False
            otherwise.
        """
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError('Features tensor is not a torch.Tensor.')
        elif len(tensor.shape) != 2:
            raise RuntimeError('Features tensor is not a torch.Tensor with '
                               'shape (n_samples, n_features).')
        elif tensor.shape[1] != self._n_features:
            raise RuntimeError('Features tensor is not consistent with data'
                               'scaler number of features.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._mean = torch.mean(tensor, dim=0)
        self._std = torch.std(tensor, dim=0, unbiased=is_bessel)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def transform(self, tensor):
        """Standardize features tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features).
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Standardized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features).
        """
        # Get number of samples
        n_samples = tensor.shape[0]
        # Build mean and standard deviation tensors for standardization
        mean = torch.tile(self._mean, (n_samples, 1))
        std = torch.tile(self._std, (n_samples, 1))
        # Standardize features tensor
        transformed_tensor = torch.div(tensor - mean, std)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def inverse_transform(self, tensor):
        """Destandardize features tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Standardized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features).
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features).
        """
        # Get number of samples
        n_samples = tensor.shape[0]
        # Build mean and standard deviation tensors for standardization
        mean = torch.tile(self._mean, (n_samples, 1))
        std = torch.tile(self._std, (n_samples, 1))
        # Destandardize features tensor
        transformed_tensor = torch.mul(tensor, std) + mean
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # -------------------------------------------------------------------------
    def _check_mean(self, mean):
        """Check features standardization mean tensor.
        
        Parameters
        ----------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
            
        Returns
        -------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        # Check features standardization mean tensor
        if not isinstance(mean, torch.Tensor):
            raise RuntimeError('Features standardization mean tensor is not a'
                                'torch.Tensor.')
        elif len(mean):
            raise RuntimeError('Features standardization mean tensor is not a '
                               'torch.Tensor(1d) with shape (n_features,).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return mean
    # -------------------------------------------------------------------------
    def _check_std(self, std):
        """Check features standardization standard deviation tensor.
        
        Parameters
        ----------
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
            
        Returns
        -------
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        """
        # Check features standardization mean tensor
        if not isinstance(std, torch.Tensor):
            raise RuntimeError('Features standardization standard deviation '
                               'tensor is not a torch.Tensor.')
        elif len(std):
            raise RuntimeError('Features standardization standard deviation '
                               'tensor is not a torch.Tensor(1d) with shape '
                               '(n_features,).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return std






print('\n\n')
data = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]]).float()
print(data.shape)


foo = TorchStandardScaler(data.shape[1])
foo.fit(data)



print(f"mean {foo._mean}, std {foo._std}")
data = foo.transform(data)

print(data)

data = foo.inverse_transform(data)

print(data)


data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))

print(scaler.transform(data))
print(scaler.transform([[2, 2]]))


print(scaler.mean_)
print(np.sqrt(scaler.var_))

mean = torch.tensor(scaler.mean_)
std = torch.sqrt(torch.tensor(scaler.var_))

print(mean)
print(std)


class TestClassInit():
    def __init__(self, x):
        self.x = x
    
    
    def init_from_call(x):
        return TestClassInit(x)
    
test_class = TestClassInit.init_from_call(10)

print(type(test_class))
print(test_class.x)


import sklearn.model_selection

n_sample = 10
n_fold = 4

k_folder = sklearn.model_selection.KFold(n_splits=n_fold, shuffle=True,
                                         random_state=0)

folds_indexes = []

for (train_index, test_index) in k_folder.split(np.zeros((n_sample, 3))):
    folds_indexes.append((train_index, test_index))

print(folds_indexes)



a = torch.tensor(
    [[ 0.2035,  1.2959,  1.8101, -0.4644],
     [ 1.5027, -0.3270,  0.5905,  0.6538],
     [-1.5745,  1.3330, -0.5596, -0.6548],
     [ 0.1264, -0.5080,  1.6420,  0.1992]])
print(torch.std(a, correction=1))
print(torch.std(a, unbiased=True))
print(torch.std(a, correction=0))
print(torch.std(a, unbiased=False))
print(torch.std(a, correction=2))

