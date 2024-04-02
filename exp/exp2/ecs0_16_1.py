import telekinesis
import time

numQubits = 16
telekinesis.create_qubits(16)
telekinesis.enableAllPair()
startTime_loc = time.time()
telekinesis.swap(14,3)
telekinesis.swap(15,7)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(3)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(6)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(10)
telekinesis.X(12)
telekinesis.X(9)
telekinesis.X(1)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.swap(13,0)
telekinesis.swap(15,1)
telekinesis.X(1)
telekinesis.X(6)
telekinesis.X(11)
telekinesis.X(3)
telekinesis.X(12)
telekinesis.X(14)
telekinesis.X(3)
telekinesis.X(0)
telekinesis.X(3)
telekinesis.X(10)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(11)
telekinesis.X(9)
telekinesis.X(8)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(10)
telekinesis.X(0)
telekinesis.X(0)
telekinesis.X(5)
telekinesis.swap(13,2)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(2)
telekinesis.X(12)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(14)
telekinesis.X(15)
telekinesis.X(11)
telekinesis.X(3)
telekinesis.X(8)
telekinesis.X(7)
telekinesis.X(6)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(6)
telekinesis.X(9)
telekinesis.X(11)
telekinesis.X(8)
telekinesis.X(6)
telekinesis.X(1)
telekinesis.X(11)
telekinesis.X(6)
telekinesis.swap(13,12)
telekinesis.swap(15,10)
telekinesis.X(12)
telekinesis.X(6)
telekinesis.X(12)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(3)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(10)
telekinesis.X(7)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(9)
telekinesis.X(8)
telekinesis.X(8)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(12)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(5)
telekinesis.X(11)
telekinesis.X(5)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(10)
telekinesis.X(9)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(10)
telekinesis.X(12)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.swap(13,3)
telekinesis.swap(14,2)
telekinesis.swap(15,12)
telekinesis.X(3)
telekinesis.X(6)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(4)
telekinesis.X(7)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(0)
telekinesis.X(12)
telekinesis.X(1)
telekinesis.X(9)
telekinesis.X(8)
telekinesis.X(10)
telekinesis.X(7)
telekinesis.X(11)
telekinesis.X(9)
telekinesis.X(10)
telekinesis.X(6)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(6)
telekinesis.X(0)
telekinesis.X(0)
telekinesis.X(5)
telekinesis.X(12)
telekinesis.X(5)
telekinesis.X(4)
telekinesis.X(1)
telekinesis.X(12)
telekinesis.X(12)
telekinesis.X(7)
telekinesis.X(9)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(12)
telekinesis.X(3)
telekinesis.X(0)
telekinesis.X(5)
telekinesis.X(11)
telekinesis.X(7)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.swap(13,11)
telekinesis.swap(14,6)
telekinesis.X(6)
telekinesis.X(12)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(5)
telekinesis.X(9)
telekinesis.X(7)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(11)
telekinesis.X(8)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(9)
telekinesis.X(11)
telekinesis.X(15)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(7)
telekinesis.X(0)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(9)
telekinesis.X(12)
telekinesis.X(14)
telekinesis.X(3)
telekinesis.X(11)
telekinesis.X(4)
telekinesis.X(1)
telekinesis.X(10)
telekinesis.X(4)
telekinesis.X(5)
telekinesis.X(3)
telekinesis.X(4)
endTime_loc = time.time()
telekinesis.disableAllPair()
dur_loc = endTime_loc - startTime_loc
print(f"Function runtime: {dur_loc}")
output_file = open("exp2_cs0", "a")
output_file.write(f"{numQubits} {dur_loc}\n")
output_file.close()
