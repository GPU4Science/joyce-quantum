import telekinesis
import numpy as np
import ctypes
import time

numQubits = 12;
telekinesis.create_qubits(numQubits)

telekinesis.test_pair(9);


# nvlink_target = numQubits-3

# total_result = [0,0,0,0]

# rep = 20

# for r in range(rep):
#     result = telekinesis.test_comm(nvlink_target)
#     total_result = [ a+b for a,b in zip(total_result,result)]

# total_result = [a/rep for a in total_result]

# output_file = open("gate_runtime_figure", "a")
# output_file.write(f"{numQubits} {dur_loc} {dur_ex} {dur_d}\n")
# output_file.close()
    


# telekinesis.test_bandwidth(6,7)
# telekinesis.test_bandwidth(4,7)
# telekinesis.test_bandwidth(2,7)