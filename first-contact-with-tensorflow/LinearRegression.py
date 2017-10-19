import numpy as np
import matplotlib.pyplot as plt

numPoints = 1000
vectorsSet = []

for i in range(numPoints):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectorsSet.append([x1, y1])

xData = [v[0] for v in vectorsSet]
yData = [v[1] for v in vectorsSet]


plt.plot(xData, yData, 'ro', label='Original data')
plt.legend()
plt.show()

