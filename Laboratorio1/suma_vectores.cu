#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

//DEVICE

__global__ void kernelSuma_Vectores(float* array_A, float* array_B, int _size){
    int idx= blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<_size){
        array_A[idx] = array_A[idx] + array_B[idx];
    }
}


//HOST
int main(){
    int size= 1000000;
    float* array_A= new float[size];
    float* array_B= new float[size];

    float* array_A_DEVICE=NULL;
    float* array_B_DEVICE=NULL;

    for (int index = 0; index < size ; index++){
        array_A[index]= index;
        array_B[index]= index;
    }

    cudaMalloc((void**)&array_A_DEVICE,size*sizeof(float));
    cudaMalloc((void**)&array_B_DEVICE,size*sizeof(float));

    cudaMemcpy(array_A_DEVICE,array_A,size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(array_B_DEVICE,array_B,size*sizeof(float),cudaMemcpyHostToDevice);

    kernelSuma_Vectores<<<ceil(size/512),512>>>(array_A_DEVICE,array_B_DEVICE,size);
    
    cudaMemcpy(array_A,array_A_DEVICE,size*sizeof(float),cudaMemcpyDeviceToHost);
    for( int index=0 ; index< 100 ; index++){
        cout<<array_A[index]<< endl;
    }
    cudaFree(array_A_DEVICE);
    cudaFree(array_B_DEVICE);

    delete[] array_A;
    delete[] array_B;
}
