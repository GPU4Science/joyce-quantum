import matplotlib.pyplot as plt
import numpy as np
import math
import copy

# File paths
file_paths = {
    "og": "exp2_og",
    "ws": "exp2_ws",
    "ro": "exp2_ro",
    "cs_0": "exp2_cs0",
    "cs_1": "exp2_cs1",
    "cs_2": "exp2_cs2",
    "pre": "exp2_pre"
}

# Adjust min and max qubits based on your dataset
min_qubits = 10
max_qubits = 30
qubit_range = range(min_qubits, max_qubits+1)

# Initialize dictionaries to hold runtimes for each qubit size
runtimes = {key: {q: [] for q in range(min_qubits, max_qubits + 1)} for key in file_paths}



# Read files and populate runtimes dictionary
for key, file_path in file_paths.items():
    with open(file_path, 'r') as file:
        for line in file:
            q, t = line.split()
            q = int(q)
            t = float(t)
            runtimes[key][q].append(t)

# Calculate speedup for each algorithm compared to the baseline (og)
speedups = {key: {q: [] for q in range(min_qubits, max_qubits + 1)} for key in file_paths if (key != "og" and key != "ro")}
for q in range(min_qubits, max_qubits + 1):
    for run_index in range(10):  # Assuming 10 runs for each qubit size
        og_time = runtimes["og"][q][run_index]
        for key in speedups:
            alg_time = runtimes[key][q][run_index]
            speedup = og_time / alg_time if alg_time != 0 else 0
            speedups[key][q].append(speedup)
            
min_speedups = {key: [min(values) for q, values in speeds.items()] for key, speeds in speedups.items()}
max_speedups = {key: [max(values) for q, values in speeds.items()] for key, speeds in speedups.items()}

overall_avg_speedup = {}
for key, q_speedups in speedups.items():
    all_speedups = [speedup for speeds in q_speedups.values() for speedup in speeds]
    overall_avg_speedup[key] = np.mean(all_speedups)

# Plotting
# Plotting adjustments
n_rows = 2  # This is correct given you have 4 algorithms to plot (ws, cs_0, cs_1, cs_2, pre), assuming 3 columns
fig, axs = plt.subplots(nrows=n_rows, ncols=3, figsize=(20, 5 * n_rows), sharey=True)
axs = axs.flatten()

avg_speedups = {}

for i, (key, q_speedups) in enumerate(speedups.items()):
    min_speedup = min_speedups[key]
    max_speedup = max_speedups[key]
    avg_speedup = [np.mean(q_speedups[q]) for q in qubit_range]  # Correct way to calculate average speedup
    avg_speedups[key]= copy.deepcopy(avg_speedup)

    axs[i].fill_between(qubit_range, min_speedup, max_speedup, alpha=0.3, label=f'{key} range')
    axs[i].axhline(y=1, color='r', linestyle='--', label='Baseline Speedup')  # Add horizontal line at speedup=1
    axs[i].axhline(y=overall_avg_speedup[key], color='g', linestyle='-.', label='Overall Avg. Speedup')
    axs[i].plot(qubit_range, avg_speedup, 'k-', label='Avg. Speedup')  # 'k-' for black solid line
    axs[i].set_title(key)
    axs[i].set_xticks(np.arange(min_qubits, max_qubits + 1, 5))  # Adjust the step as needed for clarity
    axs[i].legend()
    
for key, speed_up in avg_speedups.items():
    axs[-1].plot(qubit_range, speed_up, 'k-', label=key)  # 'k-' for black solid line
axs[-1].set_title(key)
axs[-1].set_xticks(np.arange(min_qubits, max_qubits + 1, 5))  # Adjust the step as needed for clarity
axs[-1].legend()

# Hide unused subplots if there are any
# for ax in axs[i+1:]:
#     ax.set_visible(False)

# Set common labels
fig.supylabel('Speedup')
fig.supxlabel('Qubits')

plt.savefig('exp_2_1.png')