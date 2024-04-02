#ifndef RUSHCLUSTER_CUDARRAY_CUH
#define RUSHCLUSTER_CUDARRAY_CUH

#include<iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <chrono>

#define TRUST_MODE 1

namespace cudarray_nsp {

enum eCudaArrayPolicy {
    // Set a given block size and ruond robin visit the array
    BULK_ROUND_ROBIN,
    // Split all data evenly to all devices
    EVENLY
};

enum eCudaDeviceStatus {
    // do not use this device
    DEPRECATED,
    // not allocated
    IDLE,
    // alocated but no data on it
    ALLOCATED,
    // holding data
    BUSY
};

template<typename T>
struct cudarray_device_pointer {
    T *data;
    uint64_t offset;
};

template<typename T>
struct cudarray {

    uint64_t size;
    int device_count;
    eCudaArrayPolicy policy;
    eCudaDeviceStatus *device_status;
    T **data;
    uint64_t *device_offset;
    uint64_t *device_size;
    T **swap_data;

    // Allocate a cudarray with given size and device count
    cudarray(uint64_t size, int device_count) : size(size), device_count(device_count), policy(policy) {
        printf("constructor called\n");
        // printf("creating cudarray\n");
        data = new T *[device_count];
        swap_data = new T *[device_count];
        // Count all devices
        policy = eCudaArrayPolicy::EVENLY;
        device_offset = new uint64_t[device_count];
        device_size = new uint64_t[device_count];
        // printf("checkpoint 2\n");
        for (int i = 0; i < device_count; i++) {
            data[i] = nullptr;
            // printf("checkpoint 2.1\n");
            swap_data[i] = nullptr;
            // printf("checkpoint 2.2\n");
            // device_status[i] = IDLE;
            // printf("checkpoint 2.3\n");
            device_offset[i] = 0;
            // printf("checkpoint 2.4\n");
            device_size[i] = 0;
        }
    }
    // Free all data
    ~cudarray() {
        printf("destructor called\n");
        for (int i = 0; i < device_count; i++) {
            if (data[i] != nullptr) {
                cudaFree(data[i]);
            }
            if (swap_data[i] != nullptr) { // Free swap_data memory
                cudaFree(swap_data[i]);
            }
            //printf("freed %d\n", i);
        }
        delete[] data;
        delete[] device_offset;
        delete[] device_size;
    }

    void reinitialize(uint64_t newSize, int newDeviceCount) {
        clearResources(); // Free existing resources

        // Reinitialize members with new parameters
        size = newSize;
        device_count = newDeviceCount;

        // Allocate new resources
        data = new T*[newDeviceCount];
        swap_data = new T*[newDeviceCount];
        device_offset = new uint64_t[newDeviceCount];
        device_size = new uint64_t[newDeviceCount];
        device_status = new eCudaDeviceStatus[newDeviceCount]; // Assuming this should be allocated too

        for(int i = 0; i < newDeviceCount; i++) {
            data[i] = nullptr;
            swap_data[i] = nullptr;
            //device_status[i] = eCudaDeviceStatus::IDLE;
            device_offset[i] = 0;
            device_size[i] = 0;
        }

        // Additional reinitialization logic as needed...
    }

    void allocate() {
        printf("allocate called\n");
        if (policy == eCudaArrayPolicy::EVENLY) {
            uint64_t size_per_device = size / device_count;
            for (int i = 0; i < device_count; i++) {
                cudaSetDevice(i);
                uint64_t offset = size_per_device * i;
                uint64_t end_offset = offset + size_per_device;
                uint64_t device_array_size = size_per_device;
                if (end_offset > size) {
                    end_offset = size;
                    device_array_size = end_offset - offset;
                }
                // std::cout<< "device_array_size " << device_array_size <<std::endl;
                // std::cout<< "size_per_device " << size_per_device <<std::endl;
                cudaMalloc(&data[i], device_array_size * sizeof(T));
                cudaMalloc(&swap_data[i], size_per_device * sizeof(T));
                int allocationSuccessful = (data[i] && swap_data[i]);
                std::cout<< "allocation " << allocationSuccessful << " at "<< i<<std::endl;
                device_offset[i] = size_per_device * i;
                device_size[i] = device_array_size;
            }
        } else {
            // TODO
        }
    }

    int getDeviceID(uint64_t offset) {
        if (policy == eCudaArrayPolicy::EVENLY) {
            return offset / (size / device_count);
        } else {
            // TODO
        }
        return -1;
    }

    // force inline attribute
    __forceinline__ cudarray_device_pointer<T> getDevicePointer(uint64_t offset) {
        int device_id = getDeviceID(offset);
        cudarray_device_pointer<T> pointer;
        pointer.data = data[device_id];
        pointer.offset = offset - device_offset[device_id];
        return pointer;
    }

    // * Data movement
    int setItem(uint64_t offset, T data) {
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err = cudaMemcpy(pointer.data + pointer.offset, &data, sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return -1;
        }
        return 0;
    }
    T getItem(uint64_t offset) {
        T data;
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err = cudaMemcpy(&data, pointer.data + pointer.offset, sizeof(T), cudaMemcpyDeviceToHost);
        return data;
    }
    int copyFromHostToDevice(T *host_data, uint64_t offset, uint64_t size) {
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err =
            cudaMemcpy(pointer.data + pointer.offset, host_data, size * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return -1;
        }
        return 0;
    }
    int copyFromDeviceToHost(T *host_data, uint64_t offset, uint64_t size) {
        cudarray_device_pointer<T> pointer = getDevicePointer(offset);
        cudaError_t err =
            cudaMemcpy(host_data, pointer.data + pointer.offset, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return -1;
        }
        return 0;
    }

    int copyFromDeviceToDevice(int src_device_id, int dst_device_id) {
        //uint64_t size_per_device = size / device_count;

        // cudaSetDevice(src_device_id);
        // int canAccessPeer = 0;
        // cudaDeviceCanAccessPeer(&canAccessPeer, dst_device_id, src_device_id);
        // if(canAccessPeer){
        //     cudaDeviceEnablePeerAccess(dst_device_id,0);
        // }
        // prepare copy size and pointers
        uint64_t copy_size = device_size[src_device_id] * sizeof(T);
        T* src_ptr = data[src_device_id];
        T* dst_swap_ptr = swap_data[dst_device_id];



        // auto begin = std::chrono::high_resolution_clock::now ();

        // if (canAccessPeer) {
        // // Direct copy from src to dst's swap if peer access is enabled
        cudaSetDevice(dst_device_id);
        cudaMemcpyPeer(dst_swap_ptr, dst_device_id, src_ptr, src_device_id, copy_size);
        // } else {
        // // Otherwise, use an intermediate host buffer or handle the case as needed

        // cudaMemcpyAsync(dst_swap_ptr, src_ptr, copy_size, cudaMemcpyDeviceToDevice);

        // auto end = std::chrono::high_resolution_clock::now ();
        // auto elapsed = std::chrono::duration<double> (end - begin).count ();

         //std::cout<< "copy size: " << copy_size << std::endl;

        // if (canAccessPeer) {
        //     cudaDeviceDisablePeerAccess(dst_device_id);
        // }
        return 0;
    }

private:
    void clearResources() {
        // Assuming data and swap_data pointers are to dynamically allocated arrays
        for (int i = 0; i < device_count; i++) {
            delete[] data[i]; // Correctly free each sub-array
            delete[] swap_data[i];
        }
        delete[] data;
        delete[] swap_data;
        delete[] device_offset;
        delete[] device_size;
        delete[] device_status; // Assuming this should be freed as well

        // Reset pointers to null to avoid dangling references
        data = nullptr;
        swap_data = nullptr;
        device_offset = nullptr;
        device_size = nullptr;
        device_status = nullptr;
    }
};

template<typename T>
struct complexCudarray{
    int qubit_num;
    int device_count;
    uint64_t size;
    uint64_t size_per_device;
    cudarray<T> real; // Cudarray for real part
    cudarray<T> imag; // Cudarray for imaginary part

    complexCudarray(int qubit_count, uint64_t size, int device_count) : qubit_num(qubit_count), device_count(device_count), size(size),
                                                    real(size, device_count), imag(size, device_count) {
        // printf("init complexCudarray\n");
        size_per_device = size / (uint64_t) device_count;
    }

    void allocate() {
        printf("allocating real\n");
        real.allocate();
        printf("allocating imag\n");
        imag.allocate();
    }

    void reinitialize(int qubit_count, uint64_t new_size, int new_device_count) {
        qubit_num = qubit_count;
        device_count = new_device_count;
        size = new_size;
        size_per_device = new_size / (uint64_t)device_count;
        
        // Assuming cudarray<T> has a method to reinitialize or clear and reallocate
        real.reinitialize(new_size, new_device_count);
        imag.reinitialize(new_size, new_device_count);
        
        // Additional reinitialization logic if necessary...
    }

    int targetIsLocal(int targetQubit) {
        // important assumption: each device has two's power number size
        uint64_t pair_offset = 1 << targetQubit;
        return pair_offset < size_per_device;
    }

    void copyToSwap(int dst_id, int src_id) {
        // real.swap_data[dst_id] = real.data[src_id];
        // imag.swap_data[dst_id] = imag.data[src_id];
        real.copyFromDeviceToDevice(src_id, dst_id);
        imag.copyFromDeviceToDevice(src_id, dst_id);
    }

    void enablePeer(int dst_id, int src_id) {
        int can_access_peer_dst_src;
        int can_access_peer_src_dst;
        cudaDeviceCanAccessPeer(&can_access_peer_dst_src, dst_id, src_id);
        cudaDeviceCanAccessPeer(&can_access_peer_src_dst, src_id, dst_id);
        if(can_access_peer_dst_src && can_access_peer_src_dst){
            cudaSetDevice(dst_id);
            cudaDeviceEnablePeerAccess(src_id,0);
            cudaSetDevice(src_id);
            cudaDeviceEnablePeerAccess(dst_id, 0);
        }
    }

    void disablePeer(int dst_id, int src_id) {
        int can_access_peer_dst_src;
        int can_access_peer_src_dst;
        cudaDeviceCanAccessPeer(&can_access_peer_dst_src, dst_id, src_id);
        cudaDeviceCanAccessPeer(&can_access_peer_src_dst, src_id, dst_id);
        if(can_access_peer_dst_src && can_access_peer_src_dst){
            cudaSetDevice(dst_id);
            cudaDeviceDisablePeerAccess(src_id);
            cudaSetDevice(src_id);
            cudaDeviceDisablePeerAccess(dst_id);
        }
    }

    int qubitNumInvalid(int qubit_id){

        //printf("%d\n", size);
        uint64_t offset = 1 << (qubit_id);
        //printf("%d\n", offset);
        if(offset >= size){
            printf("target qubit %d out of bound, size: %d\n", qubit_id, size);
            return 1;
        }
        return 0;;
    }
};
} // namespace cudarray_nsp

#endif