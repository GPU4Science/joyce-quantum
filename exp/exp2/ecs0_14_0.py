import telekinesis
import time

numQubits = 14
telekinesis.create_qubits(14)
telekinesis.enableAllPair()
startTime_loc = time.time()
telekinesis.X(0)
telekinesis.X(1)
telekinesis.X(3)
telekinesis.X(4)
telekinesis.X(6)
telekinesis.X(12)
telekinesis.X(11)
telekinesis.X(13)
telekinesis.X(7)
telekinesis.X(9)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(1)
telekinesis.X(9)
telekinesis.X(0)
telekinesis.X(10)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(9)
telekinesis.X(2)
telekinesis.X(11)
telekinesis.X(1)
telekinesis.X(6)
telekinesis.X(10)
telekinesis.X(8)
telekinesis.X(5)
telekinesis.X(3)
telekinesis.swap(11,0)
telekinesis.swap(12,9)
telekinesis.swap(13,2)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(4)
telekinesis.X(1)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(3)
telekinesis.X(10)
telekinesis.X(4)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(9)
telekinesis.X(4)
telekinesis.X(9)
telekinesis.X(0)
telekinesis.X(1)
telekinesis.X(2)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(8)
telekinesis.X(8)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(5)
telekinesis.X(2)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(12)
telekinesis.X(1)
telekinesis.X(6)
telekinesis.X(7)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(9)
telekinesis.X(8)
telekinesis.X(11)
telekinesis.X(4)
telekinesis.X(3)
telekinesis.X(8)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.X(4)
telekinesis.X(0)
telekinesis.swap(11,6)
telekinesis.swap(12,8)
telekinesis.swap(13,5)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(8)
telekinesis.X(7)
telekinesis.X(9)
telekinesis.X(8)
telekinesis.X(3)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(0)
telekinesis.X(4)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(4)
telekinesis.X(6)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(6)
telekinesis.X(7)
telekinesis.X(1)
telekinesis.X(3)
telekinesis.X(0)
telekinesis.X(3)
telekinesis.X(8)
telekinesis.X(2)
telekinesis.X(4)
telekinesis.X(9)
telekinesis.swap(11,7)
telekinesis.swap(12,0)
telekinesis.swap(13,8)
telekinesis.X(8)
telekinesis.X(4)
telekinesis.X(5)
telekinesis.X(9)
telekinesis.X(1)
telekinesis.X(10)
telekinesis.X(0)
telekinesis.X(6)
telekinesis.X(9)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(3)
telekinesis.X(8)
telekinesis.X(5)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(0)
telekinesis.swap(12,1)
telekinesis.swap(13,2)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(1)
telekinesis.X(10)
telekinesis.X(8)
telekinesis.X(7)
telekinesis.X(7)
telekinesis.X(4)
telekinesis.X(1)
telekinesis.X(2)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.X(11)
telekinesis.X(3)
telekinesis.X(2)
telekinesis.X(5)
telekinesis.swap(11,10)
telekinesis.swap(13,1)
telekinesis.X(1)
telekinesis.X(12)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(0)
telekinesis.X(8)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.X(3)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(2)
telekinesis.X(6)
telekinesis.X(0)
telekinesis.X(8)
telekinesis.X(10)
telekinesis.X(8)
telekinesis.X(1)
telekinesis.X(9)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(10)
telekinesis.X(10)
telekinesis.swap(12,9)
telekinesis.swap(13,1)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(9)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(9)
telekinesis.X(0)
telekinesis.X(1)
telekinesis.X(9)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(8)
telekinesis.X(2)
telekinesis.X(4)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(8)
telekinesis.X(13)
telekinesis.X(5)
telekinesis.X(8)
telekinesis.X(13)
telekinesis.X(6)
telekinesis.X(5)
telekinesis.X(0)
endTime_loc = time.time()
telekinesis.disableAllPair()
dur_loc = endTime_loc - startTime_loc
print(f"Function runtime: {dur_loc}")
output_file = open("exp2_cs0", "a")
output_file.write(f"{numQubits} {dur_loc}\n")
output_file.close()
