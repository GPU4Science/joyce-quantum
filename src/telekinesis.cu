#include "cudarray.cuh"
#include "telekinesis.cuh"
#include <memory>
#include <cstdint>
#include <sys/types.h>
#include <math.h> 
#include <vector>
#include <chrono>
#include <thread>


using namespace cudarray_nsp;


// Singleton pattern for complexCudarray

    // Getter for the singleton instance
complexCudarray<double>& qubits_register::getInstance() {
        static complexCudarray<double> instance(0, 0, 1); // Default-constructed; you might want to initialize it differently
        return instance;
    }

    // Method to initialize the singleton instance (can be called explicitly or from create_qubits)
void qubits_register::initialize(int qubit_count, int amp_count, int device_count) {
      printf("invoking initialize\n");
        complexCudarray<double>& instance = getInstance();
        // Initialize the instance as needed, e.g., setting amp_count and device_count
        // printf("checkpoint\n");
        instance.reinitialize(qubit_count, amp_count, device_count);
        // printf("about to allocate\n");
        instance.allocate();
    }


__host__ __device__ int extractBit (int locationOfBitFromRight, uint64_t theEncodedNumber) {
    return (theEncodedNumber & ( (uint64_t)1 << locationOfBitFromRight )) >> locationOfBitFromRight;
}

__host__ __device__ int64_t flipBit(int64_t number, int bitInd) {
    return (number ^ ((int64_t)1 << bitInd));
}

__host__ __device__ int isOddParity(uint64_t number, int qb1, int qb2) {
    return extractBit(qb1, number) != extractBit(qb2, number);
}

__host__ __device__ uint64_t insertZeroBit(uint64_t number, int index) {
    uint64_t left, right;
    left = (number >> index) << index;
    right = number - left;
    return (left << 1) ^ right;
}

__host__ __device__ uint64_t insertTwoZeroBits(uint64_t number, int bit1, int bit2) {
    int small = (bit1 < bit2)? bit1 : bit2;
    int big = (bit1 < bit2)? bit2 : bit1;
    return insertZeroBit(insertZeroBit(number, small), big);
}

int64_t getGlobalIdxOfOddParityInPartition(int qb1, int qb2, int device_id, int size_per_device ) {
    int64_t chunkStartInd = size_per_device * device_id;
    int64_t chunkEndInd = chunkStartInd + size_per_device; // exclusive
    int64_t oddParityInd;
    
    if (extractBit(qb1, chunkStartInd) != extractBit(qb2, chunkStartInd))
        return chunkStartInd;
    
    oddParityInd = flipBit(chunkStartInd, qb1);
    if (oddParityInd >= chunkStartInd && oddParityInd < chunkEndInd)
        return oddParityInd;
        
    oddParityInd = flipBit(chunkStartInd, qb2);
    if (oddParityInd >= chunkStartInd && oddParityInd < chunkEndInd)
        return oddParityInd;
    
    return -1;
}

void test_pair(int target){
    complexCudarray<double>& qubits = qubits_register::getInstance();
    std::cout << target << " is local: " << qubits.targetIsLocal(target) << std::endl;
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
      std::cout << device_id << " and " << paired_id << std::endl;
    }

}

std::vector<double> test_comm(int target){

    complexCudarray<double>& qubits = qubits_register::getInstance();
    
    if(!qubits.targetIsLocal(target)){
      auto begin_0 = std::chrono::high_resolution_clock::now ();
      for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
        int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
        //std::cout<< device_id << " and " << paired_id << std::endl;
        qubits.enablePeer(device_id, paired_id);
      }
      auto end_0 = std::chrono::high_resolution_clock::now ();
      auto elapsed_0 = std::chrono::duration<double> (end_0 - begin_0).count ();
      std::cout<< "Time taken for enabling is " << elapsed_0 << std::endl;

      auto begin = std::chrono::high_resolution_clock::now ();

      for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
        int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
        //std::cout<< device_id << " and " << paired_id << std::endl;
        qubits.copyToSwap(device_id, paired_id);
      }

      for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
        cudaSetDevice(device_id);
        cudaDeviceSynchronize();
      }
      auto end = std::chrono::high_resolution_clock::now ();
      auto elapsed = std::chrono::duration<double> (end - begin).count ();

      auto begin_1 = std::chrono::high_resolution_clock::now ();
      for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
        int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
        qubits.disablePeer(device_id, paired_id);
      }
      auto end_1 = std::chrono::high_resolution_clock::now ();
      auto elapsed_1 = std::chrono::duration<double> (end_1 - begin_1).count ();
      std::cout<< "Time taken for disabling is " << elapsed_1 << std::endl;

      auto begin_2 = std::chrono::high_resolution_clock::now ();

      for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
        int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
        //std::cout<< device_id << " and " << paired_id << std::endl;
        qubits.copyToSwap(device_id, paired_id);
      }

      for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
        cudaSetDevice(device_id);
        cudaDeviceSynchronize();
      }

      auto end_2 = std::chrono::high_resolution_clock::now ();
      auto elapsed_2 = std::chrono::duration<double> (end_2 - begin_2).count ();

      std::cout<< "Enabled Time taken for " << target << " is " << elapsed << std::endl;
      std::cout<< "Disabled Time taken for " << target << " is " << elapsed_2 << std::endl;
      std::cout<< "Bandwidth per device " << (2 * qubits.size * sizeof(double))/(1024 * 1024 * 1024) / elapsed / 8 << " gb/s" <<std::endl;
      //std::cout<< "Bandwidth " << (2 * qubits.size_per_device * sizeof(double))/(1024 * 1024 * 1024) / elapsed << " gb/s" <<std::endl;
      //std::cout<< "Bandwidth " << (2 * size)/(1024 * 1024 *1024) / elapsed << " gb/s" <<std::endl;
      std::vector<double> results = {elapsed_0, elapsed_1, elapsed, elapsed_2};
      return results;
      }
}

void test_bandwidth(int gpuid_0, int gpuid_1){
    // GPUs
 
    // Memory Copy Size
    uint32_t size = pow(2, 27); // 2^27 = 124MB
 
    // Allocate Memory
    uint32_t* dev_0;
    cudaSetDevice(gpuid_0);
    cudaMalloc((void**)&dev_0, size);
 
    uint32_t* dev_1;
    cudaSetDevice(gpuid_1);
    cudaMalloc((void**)&dev_1, size);
 
    //Check for peer access between participating GPUs: 
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_0, gpuid_1, can_access_peer_0_1);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_1, gpuid_0, can_access_peer_1_0);
 
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        //Enable P2P Access
        cudaSetDevice(gpuid_0);
        cudaDeviceEnablePeerAccess(gpuid_1, 0);
        cudaSetDevice(gpuid_1);
        cudaDeviceEnablePeerAccess(gpuid_0, 0);
    }
 
    // Init Timing Data
    uint32_t repeat = 10;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Init Stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaStream_t stream1;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
 
    // ~~ Start Test ~~
    // cudaEventRecord(start, stream);
 
    // cudaSetDevice(gpuid_0);
    // for (int i = 0; i < repeat; ++i) {
    //     cudaMemcpyAsync(dev_0, dev_1, size, cudaMemcpyDeviceToDevice, stream);
    // }

    // cudaSetDevice(gpuid_1);
    // for (int i = 0; i < repeat; ++i) {
    //     cudaMemcpyAsync(dev_1, dev_0, size, cudaMemcpyDeviceToDevice, stream1);
    // }
 
    // cudaEventRecord(stop, stream1);

    cudaEventRecord(start);

    auto begin = std::chrono::high_resolution_clock::now ();
 
    cudaSetDevice(gpuid_0);
    for (int i = 0; i < repeat; ++i) {
        cudaMemcpyPeer(dev_0, gpuid_0, dev_1, gpuid_1, size);
    }

    cudaSetDevice(gpuid_0);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now ();
    auto elapsed = std::chrono::duration<double> (end - begin).count ();
    double time_s = elapsed;
 
    cudaEventRecord(stop);

    cudaStreamSynchronize(stream);

    cudaStreamSynchronize(stream1);
    // ~~ End of Test ~~
 
    // Check Timing & Performance
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    // double time_s = time_ms / 1e3;

    double gb = size * repeat / (double)1e9;
    double bandwidth =  gb / time_s;
 
    printf("Seconds: %f\n", time_s);
    printf("Bidirectional Bandwidth: %f (GB/s)\n", bandwidth);
 
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        // Shutdown P2P Settings
        cudaSetDevice(gpuid_0);
        cudaDeviceDisablePeerAccess(gpuid_1);
        cudaSetDevice(gpuid_1);
        cudaDeviceDisablePeerAccess(gpuid_0);
    }
 
    // Clean Up
    cudaFree(dev_0);
    cudaFree(dev_1);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}



void create_qubits(int qubit_count){
  //printf("invoking creating_qubits\n");
  int device_count = 8; //TODO make this adpative
  int amp_count = 1 << qubit_count;
  qubits_register::initialize(qubit_count, amp_count, device_count);
};

__global__ void zeroStateKernel(uint64_t size_per_device, double* data_real, double* data_imag, int isDeviceZero){
    uint64_t index = blockIdx.x*blockDim.x + threadIdx.x;

    // initialise the state to |0000..0000>

    if (index>=size_per_device) return;
    data_real[index] = 0.0;
    data_imag[index] = 0.0;

    if (index==0 && isDeviceZero){
        // zero state |0000..0000> has probability 1
        data_real[0] = 1.0;
        data_imag[0] = 0.0;
    }
}

void zero_state(){
  complexCudarray<double>& qubits = qubits_register::getInstance();
  dim3 dimBlock(1024, 1, 1);
  dim3 dimGridLocal(ceil((double) (qubits.size_per_device >> 1) / dimBlock.x)); 
  for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
    cudaSetDevice(device_id);
    zeroStateKernel <<<dimBlock, dimGridLocal>>> (qubits.size_per_device, qubits.real.data[device_id], qubits.imag.data[device_id], (device_id==0));
  }
}

__global__ void plusStateKernel(uint64_t size_per_device, double* data_real, double* data_imag, double normFactor){
    uint64_t index = blockIdx.x*blockDim.x + threadIdx.x;
    // initialise the state to |0000..0000>
    if (index>=size_per_device) return;

    data_real[index] = normFactor;
    data_imag[index] = 0.0;
}

void plus_state(){
  complexCudarray<double>& qubits = qubits_register::getInstance();
  dim3 dimBlock(1024, 1, 1);
  dim3 dimGridLocal(ceil((double) (qubits.size_per_device >> 1) / dimBlock.x));
  double probFactor = 1.0/sqrt((double)((qubits.size)));
  printf("total_size : %d\n", qubits.size); 
  printf("normFactor : %f\n", probFactor);
  for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
    cudaSetDevice(device_id);
    plusStateKernel <<<dimBlock, dimGridLocal>>> (qubits.size_per_device, qubits.real.data[device_id], qubits.imag.data[device_id], probFactor);
  }
}

int isUpper(int device_id, uint64_t size_per_device, int target)
{       
  u_int64_t pair_offset = 1 << target;
  u_int64_t position = (device_id * size_per_device) % (pair_offset << 1);
  return position < pair_offset;
}

int getPairedDevice(int device_id, uint64_t size_per_device, int target){
  u_int64_t pair_offset = 1 << target;
  int device_is_upper = isUpper(device_id, size_per_device, target);
  int device_offset = pair_offset / size_per_device;
  if(device_is_upper) {
    return device_id + device_offset;
  } else {
    return device_id - device_offset;
  }
};


void printStates() {
    // Access the singleton instance
    complexCudarray<double>& qubits = qubits_register::getInstance();

    // Determine the number of qubit states to print
    uint64_t printSize = std::min(qubits.size, static_cast<uint64_t>(4));

    // Calculate the number of qubits to fetch per device, adjusting for the total print size
    uint64_t sizePerDevice = std::max(static_cast<uint64_t>(1), printSize / qubits.device_count); // Ensure at least one qubit per device

    // Allocate host memory for the real and imaginary parts
    double *host_real = new double[printSize];
    double *host_imag = new double[printSize];

    // Initialize variables to track the number of qubits copied
    int qubitsCopied = 0;
    uint64_t remainingQubits = printSize;

    for (int device_id = 0; device_id < qubits.device_count && remainingQubits > 0; device_id++) {
        // Calculate the number of qubits to copy from this device
        int qubitsToCopy = std::min(sizePerDevice, remainingQubits);

        // Adjust the source offset based on qubits already copied
        int sourceOffset = device_id * qubits.size_per_device;

        // Copy from device to host memory
        qubits.real.copyFromDeviceToHost(host_real + qubitsCopied, sourceOffset, qubitsToCopy);
        qubits.imag.copyFromDeviceToHost(host_imag + qubitsCopied, sourceOffset, qubitsToCopy);

        // Update counters
        qubitsCopied += qubitsToCopy;
        remainingQubits -= qubitsToCopy;
    }

    // Print the qubit states
    for (int i = 0; i < printSize; i++) {
        printf("%f + %fi\n", host_real[i], host_imag[i]);
    }

    // Free the allocated host memory
    delete[] host_real;
    delete[] host_imag;
}

void enableAllPair(){
   complexCudarray<double>& qubits = qubits_register::getInstance();
  for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      for(int paired_id = 0; paired_id < qubits.device_count; paired_id++ ) {
      if(device_id == paired_id) continue;
      cudaSetDevice(device_id);
      qubits.enablePeer(device_id,paired_id);
      }
    }
}

void disableAllPair(){
   complexCudarray<double>& qubits = qubits_register::getInstance();
  for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      for(int paired_id = 0; paired_id < qubits.device_count; paired_id++ ) {
      if(device_id == paired_id) continue;
      cudaSetDevice(device_id);
      qubits.disablePeer(device_id,paired_id);
      }
    }

}

__global__ void hadamardKernel(double* data_real, double* data_imag, double* swap_data_real, double* swap_data_imag, 
                             int target, int needExchange, int size_per_device, double recRootTwo, int isUpper){
  // a task is a single pair of values to be updated
  uint64_t task_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total_task, pair_offset, pair_id, upper_idx, lower_idx;
  double realBufferUp,realBufferLo, imagBufferUp, imagBufferLo;
  if(needExchange){
    total_task = size_per_device;
    if(task_id < total_task){

    int sign;

      if(isUpper){
        realBufferUp = data_real[task_id];
        imagBufferUp = data_imag[task_id];
        realBufferLo = swap_data_real[task_id];
        imagBufferLo = swap_data_imag[task_id];
        sign = 1;
      } else {
        realBufferUp = swap_data_real[task_id];
        imagBufferUp = swap_data_imag[task_id];
        realBufferLo = data_real[task_id];
        imagBufferLo = data_imag[task_id];
        sign = -1;
      }

      data_real[task_id] = recRootTwo * (realBufferUp + sign * realBufferLo);
      data_imag[task_id] = recRootTwo * (imagBufferUp + sign * imagBufferLo);

    }
  } else { // local version
    total_task = size_per_device >> 1;  //halved for local

    if(task_id < total_task){
      pair_offset = 1 << target;
      pair_id = task_id / pair_offset; // c feature take floor value
      upper_idx = pair_id * (pair_offset << 1) + task_id % pair_offset;
      lower_idx = upper_idx + pair_offset;

      //create buffer
      
      realBufferUp = data_real[upper_idx];
      imagBufferUp = data_imag[upper_idx];
      realBufferLo = data_real[lower_idx];
      imagBufferLo = data_imag[lower_idx];

      data_real[upper_idx] = recRootTwo * ( realBufferUp + realBufferLo);
      data_imag[upper_idx] = recRootTwo * ( imagBufferUp + imagBufferLo);

      data_real[lower_idx] = recRootTwo * ( realBufferUp - realBufferLo);
      data_imag[lower_idx] = recRootTwo * ( imagBufferUp - imagBufferLo);
    }
  }
};

void hadamard(int target){

  complexCudarray<double>& qubits = qubits_register::getInstance();
  if(qubits.qubitNumInvalid(target) == 1){
    //printf("aborting\n");
    return;
  }
  // determine if local, this is done by checking if the target qubit is in the local range
  // local range: spacing offset is 2^n where n is target qubit number(starting from 0, hence 0 is the smallest qubit number)
  double recRootTwo = 1/sqrt(2);
  dim3 dimBlock(1024, 1, 1);
  if(qubits.targetIsLocal(target)){ // local version
    //printf("invoking local\n");
    dim3 dimGridLocal(ceil((double) (qubits.size_per_device >> 1) / dimBlock.x));  // half the calculation because of in-place update
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      hadamardKernel <<<dimBlock, dimGridLocal>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                                    qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                                    target, 0, qubits.size_per_device, recRootTwo, 0);
    }

  } else {
    //printf("invoking exchange\n");
    dim3 dimGridExchange(ceil((double) (qubits.size_per_device) / dimBlock.x));
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
      qubits.copyToSwap(device_id, paired_id);
    }
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      int deviceIsUpper = isUpper(device_id, qubits.size_per_device, target);
      hadamardKernel <<<dimBlock, dimGridExchange>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                                    qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                                    target, 1, qubits.size_per_device, recRootTwo, deviceIsUpper);

    }
  }
  // in each pair of amplitudes, there exist an upper and a lower, this needs to be determined
};

void hadamard_disabled(int target){

  complexCudarray<double>& qubits = qubits_register::getInstance();
  if(qubits.qubitNumInvalid(target) == 1){
    //printf("aborting\n");
    return;
  }
  // determine if local, this is done by checking if the target qubit is in the local range
  // local range: spacing offset is 2^n where n is target qubit number(starting from 0, hence 0 is the smallest qubit number)
  double recRootTwo = 1/sqrt(2);
  dim3 dimBlock(1024, 1, 1);
  if(qubits.targetIsLocal(target)){ // local version
    //printf("invoking local\n");
    dim3 dimGridLocal(ceil((double) (qubits.size_per_device >> 1) / dimBlock.x));  // half the calculation because of in-place update
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      hadamardKernel <<<dimBlock, dimGridLocal>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                                    qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                                    target, 0, qubits.size_per_device, recRootTwo, 0);
    }

  } else {
    //printf("invoking exchange\n");
    dim3 dimGridExchange(ceil((double) (qubits.size_per_device) / dimBlock.x));
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
      qubits.copyToSwap(device_id, paired_id);
    }
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      int deviceIsUpper = isUpper(device_id, qubits.size_per_device, target);
      hadamardKernel <<<dimBlock, dimGridExchange>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                                    qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                                    target, 1, qubits.size_per_device, recRootTwo, deviceIsUpper);

    }
  }
  // in each pair of amplitudes, there exist an upper and a lower, this needs to be determined
};

__global__ void pauliXKernel(double* data_real, double* data_imag, double* swap_data_real, double* swap_data_imag, 
                             int target, int needExchange, int size_per_device){
  // a task is a single pair of values to be updated
  uint64_t task_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total_task, pair_offset, pair_id, upper_idx, lower_idx;
  if(needExchange){
    total_task = size_per_device;
    
    if(task_id < total_task){
      data_real[task_id] = swap_data_real[task_id];
      data_imag[task_id] = swap_data_imag[task_id];
    }
  } else { // local version
    total_task = size_per_device >> 1;  //halved for local

    if(task_id < total_task){
      pair_offset = 1 << target;
      pair_id = task_id / pair_offset; // c feature take floor value
      upper_idx = pair_id * (pair_offset << 1) + task_id % pair_offset;
      lower_idx = upper_idx + pair_offset;

      //create buffer
      double realBuffer, imagBuffer;
      realBuffer = data_real[upper_idx];
      imagBuffer = data_imag[upper_idx];

      data_real[upper_idx] = data_real[lower_idx];
      data_imag[upper_idx] = data_imag[lower_idx];

      data_real[lower_idx] = realBuffer;
      data_imag[lower_idx] = imagBuffer;
    }
  }
};

void pauliX(int target){

  complexCudarray<double>& qubits = qubits_register::getInstance();
  if(qubits.qubitNumInvalid(target) == 1){
    printf("aborting\n");
    return;
  }
  // determine if local, this is done by checking if the target qubit is in the local range
  // local range: spacing offset is 2^n where n is target qubit number(starting from 0, hence 0 is the smallest qubit number)
  dim3 dimBlock(1024, 1, 1);
  if(qubits.targetIsLocal(target)){ // local version
    // if (print) {
    //   printf("invoking local\n");
    // }
    dim3 dimGridLocal(ceil((double) (qubits.size_per_device >> 1) / dimBlock.x));  // half the calculation because of in-place update
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      cudaDeviceSynchronize();
    }
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      pauliXKernel <<<dimBlock, dimGridLocal>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                                    qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                                    target, 0, qubits.size_per_device);
    }

  } else {
      // if (print) {
      //   printf("invoking exchange\n");
      // }
    dim3 dimGridExchange(ceil((double) (qubits.size_per_device) / dimBlock.x));
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      cudaDeviceSynchronize();
    }
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      int paired_id = getPairedDevice(device_id, qubits.size_per_device, target);
      // if (print) {
      //   printf("copy %d to %d\n", paired_id, device_id);
      // }
      qubits.copyToSwap(device_id, paired_id);
    }
      // if (print) {
      //   printf("finished exchange\n");
      // }
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      pauliXKernel <<<dimBlock, dimGridExchange>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                                    qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                                    target, 1, qubits.size_per_device);

    }
  }
  // in each pair of amplitudes, there exist an upper and a lower, this needs to be determined
};

uint64_t getQubitBitMask(int* qubits, int num_qubits) {
    
    uint64_t mask=0; 
    for (int i=0; i< num_qubits; i++)
        mask = mask | ((uint64_t) 1 << qubits[i]);
        
    return mask;
};

__global__ void multiControlledPauliZKernel(double* data_real, double* data_imag, int size_per_device, uint64_t ctrl_mask, int device_id){
  uint64_t task_id = blockIdx.x * blockDim.x + threadIdx.x + device_id * size_per_device ;
  uint64_t task_start, task_end, id;
 // local version
  task_start = size_per_device * device_id;
  task_end = task_start + size_per_device;

  if(task_id >= task_start && task_id < task_end){
    //printf("device_id: %d  task_id: %d\n", device_id, task_id );
    if(ctrl_mask == (ctrl_mask & task_id)){
      //printf("task_id: %d, masked\n", task_id );
      id = task_id - device_id*size_per_device;
      data_real[id] = - data_real[id];
      data_imag[id] = - data_imag[id];
    }
  }
};


void multiControlledPauliZ(const std::vector<int>& controls){
  complexCudarray<double>& qubits = qubits_register::getInstance(); 
  dim3 dimBlock(1024, 1, 1);
  dim3 dimGridLocal(ceil((double) (qubits.size_per_device >> 1) / dimBlock.x));

  uint64_t ctrl_mask = getQubitBitMask(const_cast<int*>(controls.data()), controls.size());
  //printf("ctrl_mask: %x\n", ctrl_mask);

  for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
    cudaSetDevice(device_id);
    multiControlledPauliZKernel <<<dimBlock, dimGridLocal>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                                        qubits.size_per_device, ctrl_mask, device_id);
  }
};

__global__ void swapKernel(double* data_real, double* data_imag, double* swap_data_real, double* swap_data_imag, int qb1, int qb2, int needExchange, uint64_t size_per_device, int device_id, int paired_id){
  uint64_t task_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total_task;
  if(needExchange){
    total_task = size_per_device;
    uint64_t globalInd, pairGlobalInd, pairLocalInd;
    uint64_t globalStartInd = device_id * size_per_device;
    uint64_t pairGlobalStartInd = paired_id * size_per_device;

    if(task_id < total_task){
      //printf("task_id: %d\n", task_id);
      globalInd = globalStartInd + task_id;
      if (isOddParity(globalInd, qb1, qb2)) {
                
        pairGlobalInd = (uint64_t) flipBit(flipBit(globalInd, qb1), qb2);
        pairLocalInd = pairGlobalInd - pairGlobalStartInd;
        // printf("task_id: %"PRIu64", globalInd: %"PRIu64", globalStartInd: %"PRIu64", pairGlobalInd: %"PRIu64", pariLocalInd: %"PRIu64"\n", task_id, globalInd, globalStartInd ,pairGlobalInd, pairLocalInd);
        //printf("task_id: %d, globalInd: %d, globalStartInd: %d, pairGlobalInd: %d, pariLocalInd: %d\n", task_id, globalInd, globalStartInd ,pairGlobalInd, pairLocalInd);
        //double buff = swap_data_real[pairLocalInd];
        //printf("device_id: %d, task_id: %"PRIu64", pairLocalInd: %"PRIu64", swap_data_real: %f\n", device_id,task_id, pairLocalInd,buff);
        data_real[task_id] = swap_data_real[pairLocalInd];
        data_imag[task_id] = swap_data_imag[pairLocalInd];
      }
    }
  } else { // local version
    total_task = size_per_device >> 2;  //each iteration updates 2 amps and skips 2 amps

    if(task_id < total_task){
      //printf("task_id: %d\n", task_id);
      int64_t ind00, ind01, ind10;
      double re01, re10, im01, im10;
  
      // determine ind00 of |..0..0..>, |..0..1..> and |..1..0..>
      ind00 = insertTwoZeroBits(task_id, qb1, qb2);
      ind01 = (uint64_t)flipBit(ind00, qb1);
      ind10 = (uint64_t)flipBit(ind00, qb2);
      // printf("task_id: %"PRIu64", ind00: %"PRIu64", ind01: %"PRIu64", ind10: %"PRIu64"\n", task_id, ind00, ind01, ind10);
      // printf("task_id: %d, ind00: %d, ind01: %d, ind10: %d\n", task_id, ind00, ind01, ind10);
      // extract statevec amplitudes 
      re01 = data_real[ind01]; im01 = data_imag[ind01];
      re10 = data_real[ind10]; im10 = data_imag[ind10];

      // swap 01 and 10 amps
      data_real[ind01] = re10; data_real[ind10] = re01;
      data_imag[ind01] = im10; data_imag[ind10] = im01;
    }
  }
};


void swap(int qubit_one, int qubit_two){
  //printf("invoking swap\n");
  complexCudarray<double>& qubits = qubits_register::getInstance(); 
  dim3 dimBlock(1024, 1, 1);

  int larger_target = (qubit_one > qubit_two)? qubit_one : qubit_two;
  //printf("larger target: %d\n", larger_target);
  if(qubits.targetIsLocal(larger_target)){
    //printf("invoking local\n");
    dim3 dimGridLocal(ceil((double) (qubits.size_per_device >> 2) / dimBlock.x));
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      swapKernel <<<dimBlock, dimGridLocal>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                         qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                         qubit_one, qubit_two, 0, qubits.size_per_device,
                                         device_id, 0);
    }
  } else {
    // printf("invoking exchange\n");
    int paired_ids[qubits.device_count];
    dim3 dimGridExchange(ceil((double) (qubits.size_per_device) / dimBlock.x));
    for(int device_id = 0; device_id < qubits.device_count; device_id++ ) {
      cudaSetDevice(device_id);
      cudaDeviceSynchronize();
    }
    for(int device_id = 0; device_id < qubits.device_count; device_id++) {
      int64_t oddParityGlobalInd = getGlobalIdxOfOddParityInPartition(qubit_one, qubit_two, device_id ,qubits.size_per_device);
      if (oddParityGlobalInd == -1) {continue;}
      int paired_id = flipBit(flipBit(oddParityGlobalInd, qubit_one), qubit_two) / qubits.size_per_device;
      // printf("device_id: %d, paired_id: %d\n", device_id, paired_id );
      qubits.copyToSwap(device_id, paired_id);
      paired_ids[device_id] = paired_id;
    }
    for(int device_id = 0; device_id < qubits.device_count; device_id++){
      cudaSetDevice(device_id);
      swapKernel <<<dimBlock, dimGridExchange>>> (qubits.real.data[device_id], qubits.imag.data[device_id],
                                            qubits.real.swap_data[device_id], qubits.imag.swap_data[device_id],
                                            qubit_one, qubit_two, 1, qubits.size_per_device,
                                            device_id, paired_ids[device_id]);
    }
  }
};


double getProbAmp(uint64_t idx){
  complexCudarray<double>& qubits = qubits_register::getInstance();
  double realAmp, imagAmp;
  qubits.real.copyFromDeviceToHost(&realAmp, idx, 1);
  qubits.imag.copyFromDeviceToHost(&imagAmp, idx, 1);
  return realAmp*realAmp + imagAmp*imagAmp;
}

PYBIND11_MODULE(telekinesis, m) {

    m.def("get_instance", []() -> complexCudarray<double>& {
        return qubits_register::getInstance();
    }, pybind11::return_value_policy::reference);

    m.def("test_pair", &test_pair, "testing pair", pybind11::arg("target"));
    m.def("test_comm", &test_comm, "testing communication", pybind11::arg("target"));
    m.def("test_bandwidth", &test_bandwidth, "testing bandwidth", pybind11::arg("gpuid_0"), pybind11::arg("gpuid_1"));

    m.def("create_qubits", &create_qubits, "Create qubit cudarray", pybind11::arg("qubit_count"));
    m.def("zero_state", &zero_state, "init qubits to |0>");
    m.def("plus_state", &plus_state, "init qubits to |+>");
    m.def("printStates", &printStates, "Print states of the qubits");
    m.def("enableAllPair", &enableAllPair, "enable peer access");
    m.def("disableAllPair", &disableAllPair, "disale peer access");
    m.def("H", &hadamard, "hadamard", pybind11::arg("target"));
    m.def("H_d", &hadamard_disabled, "hadamard_disabled", pybind11::arg("target"));
    m.def("X", &pauliX, "pauliX", pybind11::arg("target"));
    m.def("MCZ", &multiControlledPauliZ, "multiControlledPauliZ", pybind11::arg("controls"));
    m.def("swap", &swap, "swap", pybind11::arg("qubit_one"), pybind11::arg("qubit_two"));
    m.def("getProbAmp", &getProbAmp, "get probability amplitude of a state", pybind11::arg("index"));
    // Add more bindings as needed
};

