import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

file_path_og = "exp2_og"
file_path_ws = "exp2_ws"
file_path_ro = "exp2_ro"
file_path_cs_0 = "exp2_cs0"
file_path_cs_1 = "exp2_cs1"
file_path_cs_2 = "exp2_cs2"
file_path_pre = "exp2_pre"
# Initialize lists to hold the x and y values
qubits = []
og_time = {}
ws_time = {}
ro_time = {}
cs_0_time = {}
cs_1_time = {}
cs_2_time = {}
pre_time = {}



    
with open(file_path_og, 'r') as file:
    for line in file:
        q, t = line.split()
        q = int(q)
        t = float(t)
        if q not in qubits:
            qubits.append(q) 
        if q in og_time:
            og_time[q] += t
        else:
            og_time[q] = t
og_time_avg = {q: og_time[q] / 10 for q in og_time}
        
with open(file_path_ws, 'r') as file:
    for line in file:
        q, t = line.split() 
        q = int(q)
        t = float(t)
        if q in ws_time:
            ws_time[q] += t
        else:
            ws_time[q] = t
ws_time_avg = {q: ws_time[q] / 10 for q in ws_time}
        
with open(file_path_ro, 'r') as file:
    for line in file:
        q, t = line.split()
        q = int(q)
        t = float(t) 
        if q in ro_time:
            ro_time[q] += t
        else:
            ro_time[q] = t
ro_time_avg = {q: ro_time[q] / 10 for q in ro_time}
        
with open(file_path_cs_0, 'r') as file:
    for line in file:
        q, t = line.split() 
        q = int(q)
        t = float(t)
        if q in cs_0_time:
            cs_0_time[q] += t
        else:
            cs_0_time[q] = t
cs_0_time_avg = {q: cs_0_time[q] / 10 for q in cs_0_time}
        
with open(file_path_cs_1, 'r') as file:
    for line in file:
        q, t = line.split() 
        q = int(q)
        t = float(t)
        if q in cs_1_time:
            cs_1_time[q] += t
        else:
            cs_1_time[q] = t
cs_1_time_avg = {q: cs_1_time[q] / 10 for q in cs_1_time}
        
with open(file_path_cs_2, 'r') as file:
    for line in file:
        q, t = line.split() 
        q = int(q)
        t = float(t)
        if q in cs_2_time:
            cs_2_time[q] += t
        else:
            cs_2_time[q] = t
cs_2_time_avg = {q: cs_2_time[q] / 10 for q in cs_2_time}
        
with open(file_path_pre, 'r') as file:
    for line in file:
        q, t = line.split() 
        q = int(q)
        t = float(t)
        if q in pre_time:
            pre_time[q] += t
        else:
            pre_time[q] = t
pre_time_avg = {q: pre_time[q] / 10 for q in pre_time}

        
ws_su = [og/a for a,og in zip(ws_time_avg.values(), og_time_avg.values())]
ro_su = [og/a for a,og in zip(ro_time_avg.values(), og_time_avg.values())]
cs_0_su = [og/a for a,og in zip(cs_0_time_avg.values(), og_time_avg.values())]
cs_1_su = [og/a for a,og in zip(cs_1_time_avg.values(), og_time_avg.values())]
cs_2_su = [og/a for a,og in zip(cs_2_time_avg.values(), og_time_avg.values())]
pre_su = [og/a for a,og in zip(pre_time_avg.values(), og_time_avg.values())]
    
    
# plt.plot(qubits, og_time, color="g", linewidth=1.3, label="og")
# plt.plot(qubits, ro_time, color="b", linewidth=1.3, label="ro")
# plt.plot(qubits, ws_time, color="y", linewidth=1.3, label="ws")
# plt.plot(qubits, cs_0_time, color="r", linewidth=1.3, label="cs0")
# plt.plot(qubits, cs_1_time, color="k", linewidth=1.3, label="cs1")

# plt.plot(qubits, ro_su, color="b", linewidth=1.3, label="ro")
# plt.plot(qubits, ws_su, color="y", linewidth=1.3, label="ws")
plt.plot(qubits, cs_0_su, color="r", linewidth=1.3, label="cs0")
plt.plot(qubits, cs_1_su, color="k", linewidth=1.3, label="cs1")
plt.plot(qubits, cs_2_su, color="g", linewidth=1.3, label="cs2")
plt.plot(qubits, pre_su, color="b", linewidth=1.3, label="pre")
plt.legend(loc="upper left")
# plt.yscale("log")
plt.xlabel('qubit numbers')
plt.ylabel('runtime (s)')
plt.title('Random Circuits: different swaps')


plt.savefig('exp_2_0.png')