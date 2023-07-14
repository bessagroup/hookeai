class A:
    def __init__(self, x):
        self.x = x
        
class B:
    def __init__(self, x):
        x = 2
        
a = A(1)
b = B(a.x)


import numpy as np
from shapely.geometry import Polygon

def compute_quadrilateral_area(corners_coord):
    
    x = corners_coord[:, 0]
    y = corners_coord[:, 1]
    
    area = 0.5*((x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0])
                - (x[1]*y[0] + x[2]*y[1] + x[3]*y[2] + x[0]*y[3]))
    
    return area
    
corners_coord = np.array([[0.0, 0.0],
                          [1.0, 0.0],
                          [1.0, 1.0],
                          [0.0, 1.0]])

area = compute_quadrilateral_area(corners_coord)

print('area: ', area)

polygon = Polygon(corners_coord)

print(polygon.is_valid)

