#include "telekinesis2.cuh"
#include <iostream>

int main() {
    // Example of using create_qubits and printStates
    auto qubits = create_qubits<float>(3); // Create a qubit system with 3 qubits
    printStates<float>(std::move(qubits));

    // Note: You will need to manage CUDA device setup and ensure proper memory allocation and deallocation.

    return 0;
}