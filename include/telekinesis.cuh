// telekinesis_utils.h
#ifndef TELEKINESIS_CUH
#define TELEKINESIS_CUH

#include "cudarray.cuh"
#include <cstdint>
#include<iostream>
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

using namespace cudarray_nsp;

class qubits_register {
public:
    // Delete copy constructor and assignment operator to prevent copies
    qubits_register(const qubits_register&) = delete;
    qubits_register& operator=(const qubits_register&) = delete;

    // Getter for the singleton instance
    static complexCudarray<double>& getInstance();

    // Method to initialize the singleton instance (can be called explicitly or from create_qubits)
    static void initialize(int qubit_count, int amp_count, int device_count);

protected:
    qubits_register() = default; // Constructor is protected to prevent instantiation
};
__host__ __device__ int extractBit (int locationOfBitFromRight, uint64_t theEncodedNumber);

__host__ __device__ int64_t flipBit(int64_t number, int bitInd);

__host__ __device__ int isOddParity(uint64_t number, int qb1, int qb2);

__host__ __device__ uint64_t insertZeroBit(uint64_t number, int index);

__host__ __device__ uint64_t insertTwoZeroBits(uint64_t number, int bit1, int bit2);

int64_t getGlobalIdxOfOddParityInPartition(int qb1, int qb2, int device_id, int size_per_device );

void test_pair(int target);

std::vector<double> test_comm(int target);

void test_bandwidth(int gpuid_0, int gpuid_1);

// Initializes a singleton instance for storing qubits.
void create_qubits(int qubit_count);

__global__ void zeroStateKernel(uint64_t size_per_device, double* data_real, double* data_imag, int isDeviceZero);

void zero_state();

__global__ void plusStateKernel(uint64_t size_per_device, double* data_real, double* data_imag, double normFactor);

void plus_state();

int isUpper(int device_id, uint64_t size_per_device, int target);

int getPairedDevice(int device_id, uint64_t size_per_device, int target);

// Prints the states of qubits to standard output.
void printStates();

void enableAllPair();

void disableAllPair();

__global__ void hadamardKernel(double* data_real, double* data_imag, double* swap_data_real, double* swap_data_imag, 
                             int target, int needExchange, int size_per_device, double recRootTwo, int isUpper);

void hadamard(int target);

void hadamard_disabled(int target);

__global__ void pauliXKernel(double* data_real, double* data_imag, double* swap_data_real, double* swap_data_imag, 
                             int target, int needExchange, int size_per_device);
// Applies the Pauli-X gate to a specific target qubit.
void pauliX(int target);

uint64_t getQubitBitMask(int* qubits, int num_qubits);

__global__ void multiControlledPauliZKernel(double* data_real, double* data_imag, int size_per_device, uint64_t ctrl_mask, int device_id);

void multiControlledPauliZ(const std::vector<int>& controls);

__global__ void swapKernel(double* data_real, double* data_imag, double* swap_data_real, double* swap_data_imag, int qb1, int qb2, int needExchange, uint64_t size_per_device, int device_id, int paired_id);

void swap(int qubit_one, int qubit_two);

// Gets the paired device for a given device ID, size per device, and target qubit.

double getProbAmp(uint64_t idx);

// Include the implementation of templates if needed or keep the implementation in corresponding .cpp files.
// Note: For template functions, you might need to include the implementation in the header file or explicitly instantiate templates for expected types.
// namespace cudarray_nsp
 // Only if implementation is separated

#endif // TELEKINESIS_UTILS_H_