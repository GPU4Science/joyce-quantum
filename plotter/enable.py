import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import glob
import os

file_path = "enable_figure"
# Initialize lists to hold the x and y values
qubits = []
enable_time = []
disable_time = []
enabled_comm = []
disabled_comm = []
    
with open(file_path, 'r') as file:
    for line in file:
        q, e, d, ec, dc = line.split()  # Split each line into x and y components
        qubits.append(int(q))
        enable_time.append(float(e))
        disable_time.append(float(d))
        enabled_comm.append(float(ec))
        disabled_comm.append(float(dc))
        
overhead = [a+b for a,b in zip(enable_time,disable_time)]
counted = [a+b for a,b in zip(overhead,enabled_comm)]

y = np.vstack([enabled_comm,overhead])
        
# plt.plot(qubits, enabled_comm)
# plt.plot(qubits, disabled_comm)
stacks = plt.stackplot(qubits,y)
plt.plot(qubits, disabled_comm, color="k", linewidth=1.3)
plt.yscale("log")
plt.xlabel('qubit numbers')
plt.ylabel('log runtime (s)')
plt.title('Time for NVLink enabled and disabled communication')
# Creating a legend manually
labels = ['NVLink communication', 'PeerAccess Overhead', "Disabled"]
handles = [plt.Rectangle((0, 0), 1, 1, color=stack.get_facecolor().reshape(-1)) for stack in stacks]
handles.append(plt.Rectangle((0, 0), 1, 1, color="k"))
plt.legend(handles, labels, loc='upper left')



plt.savefig('qubit_plots.png')

        
    