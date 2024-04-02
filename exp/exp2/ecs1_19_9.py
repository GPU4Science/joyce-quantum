import telekinesis
import time

numQubits = 19
telekinesis.create_qubits(19)
telekinesis.enableAllPair()
startTime_loc = time.time()
telekinesis.swap(16,9)
telekinesis.swap(17,14)
telekinesis.swap(18,15)
telekinesis.X(3)
telekinesis.X(12)
telekinesis.X(3)
telekinesis.X(2)
telekinesis.X(9)
telekinesis.X(7)
telekinesis.X(3)
telekinesis.X(1)
telekinesis.X(7)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(11)
telekinesis.X(4)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(11)
telekinesis.X(12)
telekinesis.X(17)
telekinesis.X(18)
telekinesis.X(14)
telekinesis.X(13)
telekinesis.X(5)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(6)
telekinesis.X(9)
telekinesis.X(14)
telekinesis.X(4)
telekinesis.X(9)
telekinesis.X(7)
telekinesis.X(10)
telekinesis.X(18)
telekinesis.X(8)
telekinesis.X(8)
telekinesis.X(0)
telekinesis.X(8)
telekinesis.X(7)
telekinesis.X(18)
telekinesis.X(12)
telekinesis.X(8)
telekinesis.swap(16,7)
telekinesis.X(16)
telekinesis.X(7)
telekinesis.X(10)
telekinesis.X(14)
telekinesis.X(13)
telekinesis.X(15)
telekinesis.X(1)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(10)
telekinesis.X(7)
telekinesis.X(8)
telekinesis.X(12)
telekinesis.X(7)
telekinesis.X(8)
telekinesis.X(5)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(1)
telekinesis.X(6)
telekinesis.X(6)
telekinesis.X(6)
telekinesis.X(16)
telekinesis.X(0)
telekinesis.swap(17,7)
telekinesis.X(7)
telekinesis.X(14)
telekinesis.X(3)
telekinesis.X(2)
telekinesis.X(11)
telekinesis.X(17)
telekinesis.X(6)
telekinesis.X(15)
telekinesis.X(7)
telekinesis.X(11)
telekinesis.X(13)
telekinesis.X(10)
telekinesis.X(14)
telekinesis.X(14)
telekinesis.X(9)
telekinesis.X(8)
telekinesis.X(4)
telekinesis.X(0)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(15)
telekinesis.X(15)
telekinesis.X(12)
telekinesis.X(11)
telekinesis.X(15)
telekinesis.X(13)
telekinesis.X(6)
telekinesis.X(0)
telekinesis.X(11)
telekinesis.X(12)
telekinesis.X(8)
telekinesis.X(14)
telekinesis.X(2)
telekinesis.X(10)
telekinesis.X(2)
telekinesis.X(13)
telekinesis.X(10)
telekinesis.X(6)
telekinesis.swap(18,0)
telekinesis.X(18)
telekinesis.X(1)
telekinesis.X(14)
telekinesis.X(3)
telekinesis.X(17)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(14)
telekinesis.X(2)
telekinesis.X(9)
telekinesis.X(13)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(12)
telekinesis.X(17)
telekinesis.X(2)
telekinesis.X(4)
telekinesis.X(0)
telekinesis.X(11)
telekinesis.X(1)
telekinesis.X(8)
telekinesis.X(15)
telekinesis.X(14)
telekinesis.X(2)
telekinesis.X(6)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(1)
telekinesis.X(11)
telekinesis.X(6)
telekinesis.X(11)
telekinesis.X(3)
telekinesis.X(14)
telekinesis.X(13)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(5)
telekinesis.X(7)
telekinesis.X(6)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(4)
telekinesis.X(12)
telekinesis.X(5)
telekinesis.X(3)
telekinesis.X(1)
telekinesis.X(13)
telekinesis.X(13)
telekinesis.X(3)
telekinesis.X(14)
telekinesis.X(8)
telekinesis.X(3)
telekinesis.X(4)
telekinesis.X(8)
telekinesis.X(3)
telekinesis.X(15)
telekinesis.X(7)
telekinesis.X(6)
telekinesis.X(3)
telekinesis.X(2)
telekinesis.swap(16,10)
telekinesis.X(10)
telekinesis.X(1)
telekinesis.X(2)
telekinesis.X(6)
telekinesis.X(4)
telekinesis.X(10)
telekinesis.X(3)
telekinesis.X(9)
telekinesis.X(2)
telekinesis.X(2)
telekinesis.X(5)
telekinesis.X(6)
telekinesis.X(13)
telekinesis.X(3)
telekinesis.X(15)
telekinesis.X(11)
telekinesis.X(15)
telekinesis.X(15)
telekinesis.X(11)
telekinesis.X(2)
telekinesis.X(8)
telekinesis.X(14)
telekinesis.X(9)
telekinesis.X(14)
telekinesis.X(12)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(3)
telekinesis.X(3)
telekinesis.X(7)
telekinesis.X(5)
telekinesis.X(9)
telekinesis.X(14)
telekinesis.X(0)
telekinesis.X(8)
telekinesis.X(0)
endTime_loc = time.time()
telekinesis.disableAllPair()
dur_loc = endTime_loc - startTime_loc
print(f"Function runtime: {dur_loc}")
output_file = open("exp2_cs1", "a")
output_file.write(f"{numQubits} {dur_loc}\n")
output_file.close()
