from collections import deque

import numpy as np
noise = []
padding = np.zeros((10 - len(noise), 2))
print(padding[0])
for _ in range(len(padding)):
    noise.append(padding[_])
print(noise)
noise = deque(noise)
print(noise[0])