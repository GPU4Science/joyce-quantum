import telekinesis
import math
import numpy as np
import ctypes
import time

# effect |solElem> -> -|solElem> via a 
# multi-controlled phase flip gate 

def applyOracle(numQubits, solElem, r):

    #apply X to transform |111> into |solElem>
    for q in range(numQubits):
        if (((solElem >> q) & 1) == 0):
            #telekinesis.X(q)
            # if r==0 :
            #     telekinesis.X(q)
            #     telekinesis.printStates()
            #     print(q)
            # else:
            telekinesis.X(q)
    
    # if r==1 :
    #     telekinesis.printStates()
    #     print("next is mcz")
        
    #effect |111> -> -|111>    
    controls = list(range(numQubits))
    telekinesis.MCZ(controls)
    
    # if r==1 :
    #     telekinesis.printStates()
    #     print("")
    
    #apply X to transform |solElem> into |111>
    for q in range(numQubits):
        if (((solElem >> q) & 1) == 0):
            telekinesis.X(q)



#  apply 2|+><+|-I by transforming into the Hadamard basis 
#  and effecting 2|0><0|-I. We do this, by observing that 
#    c..cZ = diag{1,..,1,-1} 
#          = I - 2|1..1><1..1|
#  and hence 
#    X..X c..cZ X..X = I - 2|0..0><0..0|
#  which differs from the desired 2|0><0|-I state only by 
#  the irrelevant global phase pi

def applyDiffuser(numQubits) :
    
    #apply H to transform |+> into |0>
    for q in range(numQubits):
        telekinesis.H(q)

    #apply X to transform |11..1> into |00..0>
    for q in range(numQubits):
        telekinesis.X(q)
    
    #effect |11..1> -> -|11..1>
    controls = list(range(numQubits))
    telekinesis.MCZ(controls)
    
    #apply X to transform |00..0> into |11..1>
    for q in range(numQubits):
        telekinesis.X(q)
    
    #apply H to transform |0> into |+>
    for q in range(numQubits):
        telekinesis.H(q)


    
#choose the system size
numQubits = 13;
numElems = int(2 ** numQubits)
numReps = math.ceil(math.pi / 4 * math.sqrt(numElems))
    
print(f"numQubits:{numQubits} , numElems: {numElems}, numReps: {numReps}\n")
    
#choose the element for which to search
solElem = 200
    
#prepare |+>
telekinesis.create_qubits(numQubits)
telekinesis.plus_state()

#telekinesis.printStates()
print("")
#apply Grover's algorithm
for r in range(numReps):
    applyOracle(numQubits, solElem, r)
    # telekinesis.printStates()
    # print("")
    applyDiffuser(numQubits)
    # telekinesis.printStates()
    # print("")
        
    #monitor the probability of the solution state
    print(f"prob of solution |{solElem}> =  {telekinesis.getProbAmp(solElem)}")

    
#free memory 
    # destroyQureg(qureg, env);
    # destroyQuESTEnv(env);