import telekinesis
import time

numQubits = 11
telekinesis.create_qubits(11)
telekinesis.enableAllPair()
startTime_loc = time.time()
telekinesis.swap(10,6)
telekinesis.X(8)
telekinesis.X(3)
telekinesis.X(5)
telekinesis.X(9)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(0)
telekinesis.X(10)
telekinesis.X(4)
telekinesis.X(9)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(6)
telekinesis.X(2)
telekinesis.X(10)
telekinesis.X(3)
telekinesis.X(2)
telekinesis.X(6)
telekinesis.X(8)
telekinesis.X(7)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(1)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(0)
telekinesis.X(10)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(7)
telekinesis.X(4)
telekinesis.X(8)
telekinesis.X(3)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(3)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(8)
telekinesis.X(6)
telekinesis.X(10)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(10)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(4)
telekinesis.X(10)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(0)
telekinesis.X(1)
telekinesis.X(6)
telekinesis.X(0)
telekinesis.X(4)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(8)
telekinesis.X(10)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(6)
telekinesis.X(3)
telekinesis.X(3)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(2)
telekinesis.X(8)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(4)
telekinesis.X(4)
telekinesis.X(3)
telekinesis.X(10)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(6)
telekinesis.X(6)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(5)
telekinesis.swap(8,2)
telekinesis.X(8)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(8)
telekinesis.X(6)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(3)
telekinesis.X(10)
telekinesis.X(4)
telekinesis.X(0)
telekinesis.X(3)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(6)
telekinesis.X(0)
telekinesis.X(5)
telekinesis.X(3)
telekinesis.X(3)
telekinesis.X(2)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(8)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(6)
telekinesis.X(1)
telekinesis.X(3)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(3)
telekinesis.X(7)
telekinesis.X(4)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(4)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(0)
telekinesis.X(9)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(0)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.X(5)
telekinesis.X(4)
telekinesis.swap(8,2)
telekinesis.X(8)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(5)
telekinesis.X(4)
telekinesis.X(6)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(7)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(9)
telekinesis.X(5)
telekinesis.X(4)
telekinesis.X(6)
telekinesis.X(5)
telekinesis.X(1)
telekinesis.X(3)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(0)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(3)
telekinesis.X(7)
telekinesis.X(0)
telekinesis.X(4)
telekinesis.X(8)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(1)
telekinesis.X(2)
endTime_loc = time.time()
telekinesis.disableAllPair()
dur_loc = endTime_loc - startTime_loc
print(f"Function runtime: {dur_loc}")
output_file = open("exp2_cs1", "a")
output_file.write(f"{numQubits} {dur_loc}\n")
output_file.close()
