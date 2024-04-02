import telekinesis
import time

numQubits = 26
telekinesis.create_qubits(26)
startTime_loc = time.time()
telekinesis.enableAllPair()
startTime_raw = time.time()
telekinesis.X(1)
telekinesis.X(23)
telekinesis.X(20)
telekinesis.X(16)
telekinesis.X(22)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(18)
telekinesis.X(4)
telekinesis.X(7)
telekinesis.X(6)
telekinesis.X(22)
telekinesis.X(14)
telekinesis.X(1)
telekinesis.X(23)
telekinesis.X(1)
telekinesis.X(18)
telekinesis.X(16)
telekinesis.X(1)
telekinesis.X(13)
telekinesis.X(25)
telekinesis.X(17)
telekinesis.X(7)
telekinesis.X(13)
telekinesis.X(11)
telekinesis.X(5)
telekinesis.X(5)
telekinesis.X(1)
telekinesis.X(21)
telekinesis.X(24)
telekinesis.X(13)
telekinesis.X(21)
telekinesis.X(16)
telekinesis.X(3)
telekinesis.X(18)
telekinesis.X(18)
telekinesis.X(2)
telekinesis.X(21)
telekinesis.X(24)
telekinesis.X(17)
telekinesis.X(1)
telekinesis.X(20)
telekinesis.X(23)
telekinesis.X(17)
telekinesis.X(0)
telekinesis.X(3)
telekinesis.X(20)
telekinesis.X(20)
telekinesis.X(9)
telekinesis.X(14)
telekinesis.X(8)
telekinesis.X(10)
telekinesis.X(14)
telekinesis.X(8)
telekinesis.X(19)
telekinesis.X(15)
telekinesis.X(17)
telekinesis.X(8)
telekinesis.X(17)
telekinesis.X(18)
telekinesis.X(24)
telekinesis.X(4)
telekinesis.X(23)
telekinesis.X(5)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(4)
telekinesis.X(23)
telekinesis.X(12)
telekinesis.X(21)
telekinesis.X(5)
telekinesis.X(13)
telekinesis.X(9)
telekinesis.X(1)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(3)
telekinesis.X(10)
telekinesis.X(22)
telekinesis.X(7)
telekinesis.X(9)
telekinesis.X(17)
telekinesis.X(10)
telekinesis.X(19)
telekinesis.X(7)
telekinesis.X(24)
telekinesis.X(7)
telekinesis.X(5)
telekinesis.X(11)
telekinesis.X(10)
telekinesis.X(17)
telekinesis.X(7)
telekinesis.X(9)
telekinesis.X(3)
telekinesis.X(1)
telekinesis.X(15)
telekinesis.X(16)
telekinesis.X(12)
telekinesis.X(25)
telekinesis.X(15)
telekinesis.X(17)
telekinesis.X(10)
telekinesis.X(5)
telekinesis.X(13)
telekinesis.X(22)
telekinesis.X(24)
telekinesis.X(8)
telekinesis.X(7)
telekinesis.X(7)
telekinesis.X(23)
telekinesis.X(7)
telekinesis.X(0)
telekinesis.X(22)
telekinesis.X(16)
telekinesis.X(21)
telekinesis.X(18)
telekinesis.X(13)
telekinesis.X(24)
telekinesis.X(10)
telekinesis.X(0)
telekinesis.X(2)
telekinesis.X(16)
telekinesis.X(8)
telekinesis.X(12)
telekinesis.X(8)
telekinesis.X(22)
telekinesis.X(16)
telekinesis.X(16)
telekinesis.X(13)
telekinesis.X(12)
telekinesis.X(13)
telekinesis.X(14)
telekinesis.X(12)
telekinesis.X(21)
telekinesis.X(10)
telekinesis.X(0)
telekinesis.X(7)
telekinesis.X(23)
telekinesis.X(13)
telekinesis.X(1)
telekinesis.X(25)
telekinesis.X(5)
telekinesis.X(24)
telekinesis.X(11)
telekinesis.X(7)
telekinesis.X(2)
telekinesis.X(23)
telekinesis.X(5)
telekinesis.X(13)
telekinesis.X(10)
telekinesis.X(3)
telekinesis.X(5)
telekinesis.X(4)
telekinesis.X(19)
telekinesis.X(1)
telekinesis.X(6)
telekinesis.X(3)
telekinesis.X(6)
telekinesis.X(3)
telekinesis.X(18)
telekinesis.X(25)
telekinesis.X(0)
telekinesis.X(23)
telekinesis.X(19)
telekinesis.X(11)
telekinesis.X(11)
telekinesis.X(17)
telekinesis.X(25)
telekinesis.X(6)
telekinesis.X(8)
telekinesis.X(16)
telekinesis.X(18)
telekinesis.X(15)
telekinesis.X(20)
telekinesis.X(12)
telekinesis.X(21)
telekinesis.X(22)
telekinesis.X(18)
telekinesis.X(12)
telekinesis.X(12)
telekinesis.X(15)
telekinesis.X(21)
telekinesis.X(16)
telekinesis.X(2)
telekinesis.X(8)
telekinesis.X(20)
telekinesis.X(21)
telekinesis.X(16)
telekinesis.X(17)
telekinesis.X(22)
telekinesis.X(4)
telekinesis.X(19)
telekinesis.X(18)
telekinesis.X(18)
telekinesis.X(23)
telekinesis.X(17)
telekinesis.X(4)
telekinesis.X(2)
telekinesis.X(13)
telekinesis.X(23)
endTime_raw = time.time()
telekinesis.disableAllPair()
endTime_loc = time.time()
dur_loc = endTime_loc - startTime_loc
dur_raw = endTime_raw - startTime_raw
print(f"Function runtime: {dur_loc}")
output_file = open("exp_1_0_e", "a")
output_file.write(f"{numQubits} {dur_loc} {dur_raw}\n")
output_file.close()
