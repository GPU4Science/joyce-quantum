import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

file_path_d = "exp_1_0_d"
file_path_e = "exp_1_0_e"
# Initialize lists to hold the x and y values
qubits = []
enabled_time = []
enabled_time_raw = []
disabled_time = []

    
with open(file_path_d, 'r') as file:
    for line in file:
        q, d = line.split()  # Split each line into x and y components
        qubits.append(int(q))
        disabled_time.append(float(d))
        
with open(file_path_e, 'r') as file:
    for line in file:
        q, e, r = line.split()  # Split each line into x and y components
        enabled_time.append(float(e))
        enabled_time_raw.append(float(r))
        
overhead = [t-r for t,r in zip(enabled_time, enabled_time_raw)]
        
y = np.vstack([enabled_time_raw,overhead])
        
stacks = plt.stackplot(qubits,y)
plt.plot(qubits, disabled_time, color="g", linewidth=1.3)
plt.yscale("log")
plt.xlabel('qubit numbers')
plt.ylabel('runtime (s)')
plt.title('Random Circuits: NvLink enabled vs disabled')

plt.savefig('exp_1_0.png')