import subprocess
loc = "exp/exp2/"

qubits = range(10,31,1)
batch = range(10)

og = []
for b in batch:
    for q in qubits:
        og.append(f'eog_{q}_{b}.py')

cs0 = []
for b in batch:
    for q in qubits:
        cs0.append(f'ecs0_{q}_{b}.py')
        
cs1 = []
for b in batch:
    for q in qubits:
        cs1.append(f'ecs1_{q}_{b}.py')

cs2 = []
for b in batch:
    for q in qubits:
        cs2.append(f'ecs2_{q}_{b}.py')
        
ro = []
for b in batch:
    for q in qubits:
        ro.append(f'ero_{q}_{b}.py')

pre = []
for b in batch:
    for q in qubits:
        pre.append(f'epre_{q}_{b}.py')

ws = []
for b in batch:
    for q in qubits:
        ws.append(f'ews_{q}_{b}.py')

for script in og:
    subprocess.run(['python3', loc+script])

for script in pre:
    subprocess.run(['python3', loc+script])

for script in ws:
    subprocess.run(['python3', loc+script])
    
for script in ro:
    subprocess.run(['python3', loc+script])

for script in cs0:
    subprocess.run(['python3', loc+script])

for script in cs1:
    subprocess.run(['python3', loc+script])

for script in cs2:
    subprocess.run(['python3', loc+script])