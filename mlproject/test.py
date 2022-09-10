import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

x = np.linspace(-10,10)
plt.plot(x,f(x))
plt.show()
