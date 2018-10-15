# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# Data
X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])

# Define Gaussian distribution as basis function
def phi(x) :
	s = 0.1 # assume (s: the width of Gaussian distribution)
	return np.append(1, np.exp(-(x - np.arange(0, 1 + s, s)) ** 2 / (2 * s * s)))

# Normal Linear Regression
PHI = np.array([phi(x) for x in X])  
print ("PHI = ",PHI)

omega_normal = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, t))
print ("omega (normal linear regression) = ", omega_normal)
print(PHI.shape[1],'-'*40)
# Bayesian Linear Regression
alpha = 0.1 # assume
beta = 9.0  # assume
Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, t))
print ("mu_N (Bayesian linear regression) = ", mu_N)

def f(omega, x) :
	return np.dot(omega, phi(x))

# Draw the graph
xlist = np.arange(0, 1, 0.01)
normal_lr = [f(omega_normal, x) for x in xlist]
# bayesian_lr = [np.dot(mu_N, phi(x)) for x in xlist]
bayesian_lr = [f(mu_N, x) for x in xlist]
print(len(bayesian_lr))
#plt.plot(xlist, normal_lr, 'r',label='real')
plt.plot(xlist, bayesian_lr, 'r--',label='BY')
plt.legend()
plt.plot(X, t, 'o') # => plot the data
plt.show()