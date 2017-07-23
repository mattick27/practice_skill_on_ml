from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt



iris = datasets.load_iris()
x = iris.data[:100]
y = iris.target[:100]
a = np.array(x)[:,[0]]
b = np.array(x)[:,[1]]
c = np.array(x)[:,[2]]
d = np.array(x)[:,[3]]
plt.plot(a,'ro')
plt.plot(b,'y+')
plt.plot(c,'go')
plt.plot(d,'b.')
plt.plot(y,'g^')
plt.show()

