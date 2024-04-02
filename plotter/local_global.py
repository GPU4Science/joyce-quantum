import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import glob
import os

file_path = "gate_runtime_figure"
# Initialize lists to hold the x and y values
qubits = []
local_time = []
nv_time = []
global_time = []

    
with open(file_path, 'r') as file:
    for line in file:
        q, l, n, g = line.split()  # Split each line into x and y components
        qubits.append(int(q))
        local_time.append(float(l))
        nv_time.append(float(n))
        global_time.append(float(g))
        
        
plt.plot(qubits, local_time, color="k", linewidth=1.3)
plt.plot(qubits, global_time, color="g", linewidth=1.3)
plt.plot(qubits, nv_time, color="r", linewidth=1.3)
plt.yscale("log")
plt.xlabel('qubit numbers')
plt.ylabel('log runtime (s)')
plt.title('Quantum gate: local vs nvLink vs global')



plt.savefig('runtime_plots.png')
