
from qiskit import QuantumCircuit
from qiskit.circuit.library import n_local
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from scipy.optimize import minimize

# Reference state
ref_state = QuantumCircuit(2)
ref_state.x(0)

# Parameterized ansatz
var_state = n_local(
    2,
    rotation_blocks=["rx", "ry"],
    entanglement_blocks=["cx"],
    entanglement="linear"
)

ansatz = ref_state.compose(var_state)

J = 2.0
H = 1.0
# Hamiltonian
H = SparsePauliOp.from_list([
    ("XX", J/4), ("YY", J/4), ("ZZ", J/4),
    ("ZI", H), ("IZ", H),  ("XI", H), ("IX", H),  ("YI", H), ("IY", H)
])

estimator = StatevectorEstimator()

# Cost function: must return a scalar
def cost_fun(param, ansatz, hamilton, estimator):
    pub = (ansatz, hamilton, param)
    result = estimator.run([pub]).result()[0]
    cost = result.data.evs
    return float(np.asarray(cost).item())

# Match the parameter count to the ansatz
print("Number of parameters:", ansatz.num_parameters)

param = 2 * np.pi * np.random.random(ansatz.num_parameters)

result = minimize(
    cost_fun,
    param,
    args=(ansatz, H, estimator),
    method="BFGS"
)

print("Optimal parameters:", result.x)
print("Minimum energy:", result.fun)
print("Converged:", result.success)
print("Message:", result.message)

print("\n-----------------Eigen Values of the Hamiltonian-------------------------")
print(np.linalg.eigvalsh(H.to_matrix()))
