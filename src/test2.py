import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
width = 4
height = 4
bias_coords = [(0,0,-2)]
input_coords = [
    (x, y, -1 + z) for x in np.linspace(-1, 1, width) for y in np.linspace(-1, 1, height) for z in np.linspace(-.1, .1, 3)
]
print(input_coords)
print(np.array(input_coords).flatten())
