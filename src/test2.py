import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
width = 4
height = 8
bias_coords = [(0,0,-2)]
input_coords = [
    (x,y) for x in np.linspace(1, width, width) for y in np.linspace(1, height, height)
]

m = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print(np.array(input_coords))
print(np.array(input_coords).flatten())
print(m.shape)

