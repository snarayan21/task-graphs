import numpy as np
import matplotlib.pyplot as plt


a = 1.0
b = 10
c = 0.2
x = np.linspace(0, 1.0, 100)

plt.plot(x, a / (1 + np.exp(-1 * b * (x - c))), color='red')

a = 1.0
b = 5.0
c = 1.0
plt.plot(x, (a - c*np.exp(-1 * b * x)), color='blue')

a = 5.0
plt.plot(x, a*x**2.0, color='green')
plt.show()

