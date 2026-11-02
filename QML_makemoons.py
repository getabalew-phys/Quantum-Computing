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
Xir, yir = iris.data[:100, [1,2]], iris.target[:100]
make_moon = make_moons(600)
X, y = iris.data[:100, [0,3,2]], iris.target[:100]

y = np.where(y==0, -1, 1)
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
H = SparsePauliOp("XYZ")

def encoder(X):
    x1, x2, x3 = X
    qubit = QuantumCircuit(3)
    qubit.rx(x1, 0)
    qubit.rx(x2, 1)
    qubit.rx(x3, 2)
    
    
    return qubit



def myAnsatz(param, encoded_state):
    theta1, theta2, theta3 = param
    ansatz = encoded_state
    ansatz.ry(theta1, 0)
    ansatz.ry(theta2, 1)
    ansatz.ry(theta3, 2)
    
    ansatz.cz(1,2)
    #ansatz.cz(1,2)
   
    
    return ansatz

def predictor(param, X, estimator, hamilton):
    
    encoded_state = encoder(X)
    ansatz = myAnsatz(param, encoded_state)  
    
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
        correct += (ylabel == (ypred >0))
    global accuracy
    accuracy = correct/X.shape[0]
        
    return cost/X.shape[0]

param = np.random.uniform(0, 0.1, 3)

#cost = cost_fun(Xtrain, ytrain, param, estimator, H)


         
final = minimize(cost_fun, param, args=(Xtrain, ytrain, estimator, H), method="BFGS")

parameters = final.x 
ypred = predictor(parameters, Xtest[5, :], estimator, H)

test_cost = cost_fun(parameters, Xtest, ytest, estimator, H)

print(final)
print(f"Accuracy on the test data: {accuracy*100: .2f}%")



