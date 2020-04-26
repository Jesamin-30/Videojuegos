#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

//DEVICE

__global__ void kernelVector_suma_constante(float* array, int _size, int _constant){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size){
        array[idx] = array[idx]+_constant;
    }
}


//HOST
int main(){
    int size = 1000000;
    float* arr = new float[size];
    float* arr_DEVICE= NULL;

    for (int index = 0; index < size; index++){
        arr[index] = index;
    }
    
    cudaMalloc((void**)&arr_DEVICE,size * sizeof(float));
    cudaMemcpy(arr_DEVICE, arr,size * sizeof(float), cudaMemcpyHostToDevice);

    kernelVector_suma_constante <<< ceil(size/512),512>>>(arr_DEVICE,size,65);
    
    cudaMemcpy(arr,arr_DEVICE,size * sizeof (float), cudaMemcpyDeviceToHost);
    for ( int index = 0; index<100; index++){
        cout<<arr[index]<<endl;
    }
    
    cudaFree(arr_DEVICE);
    delete[] arr;
}