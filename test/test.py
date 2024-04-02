import telekinesis
import numpy as np
import ctypes
import time

numQubits = 30;
telekinesis.create_qubits(numQubits)

telekinesis.enableAllPair()

local_target = 0
nvlink_target = numQubits-3
global_target = numQubits-1

rep = 10

startTime_loc = time.time()
for r in range(rep):
    telekinesis.X(local_target)

endTime_loc = time.time()
dur_loc = endTime_loc - startTime_loc
dur_loc /= rep
print(f"Function runtime: {dur_loc}")


startTime_ex = time.time()
for r in range(rep):
    telekinesis.X(nvlink_target)

endTime_ex = time.time()
dur_ex = endTime_ex - startTime_ex
dur_ex /= rep
print(f"Function runtime: {dur_ex}")

startTime_g = time.time()
for r in range(rep):
    telekinesis.X(global_target)

endTime_g = time.time()
dur_g = endTime_g - startTime_g
dur_g /= rep
print(f"Function runtime: {dur_g}")

telekinesis.disableAllPair()


# nvlink_target = numQubits-3

# total_result = [0,0,0,0]

# rep = 20

# for r in range(rep):
#     result = telekinesis.test_comm(nvlink_target)
#     total_result = [ a+b for a,b in zip(total_result,result)]

# total_result = [a/rep for a in total_result]

output_file = open("gate_runtime_figure", "a")
output_file.write(f"{numQubits} {dur_loc} {dur_ex} {dur_g}\n")
output_file.close()
    


# telekinesis.test_bandwidth(6,7)
# telekinesis.test_bandwidth(4,7)
# telekinesis.test_bandwidth(2,7)