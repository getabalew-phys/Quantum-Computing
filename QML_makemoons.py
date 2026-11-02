# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:46:24 2026

@author: getsh
"""

import qiskit
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp

from scipy.optimize import minimize

iris = load_iris()

make_moon = make_moons(100)
X, y = make_moon[0], make_moon[1]
X, y = iris.data[:100, [1,3,2]], iris.target[:100]


X = np.pi*(np.max(X, axis=0, keepdims=True) - X)/(np.max(X, axis=0) - 
                                            np.min(X, axis=0))
Xtrain, Xtest, ytrain,ytest = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

"""
# Here is the visualization of the data 

plt.scatter(X[y==0, 0], X[y==0, 1], marker="o", alpha =0.8, label="class 0")
plt.scatter(X[y==1, 0], X[y==1, 1], marker=">", alpha=0.8, label="class 1")
plt.legend()
plt.grid()
plt.xlabel("$X_1$", size=10)
plt.ylabel("$X_2$", size=10)
plt.title("Make moons dataset", size=20) """

qp = QuantumCircuit(1)
qp.ry(0.2, 0)

estimator = StatevectorEstimator()
H = SparsePauliOp("XZZ")

def encoder(X):
    n_qubit = len(X)
    qubit = QuantumCircuit(n_qubit)
    
    for idx, item in enumerate(X):
        qubit.rx(item, idx)
    
    
    return qubit



def myAnsatz(param, encoded_state):
    theta1, theta2, theta3 = param
    ansatz = encoded_state
    ansatz.rx(theta1, 0)
    ansatz.ry(theta2, 1)
    ansatz.ry(theta3, 2)
    
    ansatz.cx(1,2)
    ansatz.cx(0,1)
   
    
    return ansatz

def predictor(param, X, estimator, hamilton):
    
    encoded_state = encoder(X)
    ansatz = myAnsatz(param*X, encoded_state)  
    
    pub = (ansatz, hamilton)
    result = estimator.run([pub]).result()[0] 
    pred = (result.data.evs) 
    return pred

def cost_fun(param, X, y, estimator, hamilton):
    
     
    cost = 0.0
    correct = 0
    for xdata, ylabel in zip(X, y):
        cost_epoch = 0.0 
        ypred = predictor(param, xdata, estimator, hamilton)
        cost_epoch = (ylabel - ypred)**2
        cost += cost_epoch
        correct += (ylabel == (ypred > 0))
  
        
    return cost/X.shape[0]

param = np.random.uniform(0, 0.1, 3)

#cost = cost_fun(Xtrain, ytrain, param, estimator, H)


ytrain = np.where(y==0, -1, 1)        
final = minimize(cost_fun, param, args=(Xtrain, ytrain, estimator, H), method="COBYLA")

parameters = final.x
#parameters = [2.521e-01, -1.951e+00,  1.849e+00] this set of parameters for iris brings 96% accuracy
#ypred = predictor(parameters, Xtest, estimator, H)

ypred = []
for i in range(len(Xtest)):
    ypred.append(predictor(parameters, Xtest[i, :], estimator, H) > 0 )
    print(ytest[i], predictor(parameters, Xtest[i, :], estimator, H) )
accuracy = np.mean( ytest == ypred)


print(final)
print(f"Accuracy on the test data: {accuracy*100: .2f}%")



