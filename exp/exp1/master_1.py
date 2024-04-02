import subprocess
loc = "exp/exp1/"

qubits = range(10,31,1)

disabled_s = []
for q in qubits:
    disabled_s.append(f'd_{q}.py')

enabled_s = []
for q in qubits:
    enabled_s.append(f'e_{q}.py')

for script in disabled_s:
    subprocess.run(['python3', loc+script])

for script in enabled_s:
    subprocess.run(['python3', loc+script])