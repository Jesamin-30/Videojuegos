#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
const int DIMBLOCKX=32;
//DEVICE


__global__ void kernelSum_Column_Matrix(float* matrix, float* array, int tam){
    __shared__ float shareMatrix[DIMBLOCKX];

    float value=0;
    int col=blockIdx.x;
    int step= tam/blockDim.x;
    int posIni= col*tam+threadIdx.x*step;
    for(int i=0;i<step;i++){
        value=value+matrix[posIni+i];
    }
    
    shareMatrix[threadIdx.x]=value;
    __syncthreads();

    if(threadIdx.x==0){
        for(int j=1;j<blockDim.x;j++){
            shareMatrix[0]=shareMatrix[0]+shareMatrix[j];
        }
        array[blockIdx.x]=shareMatrix[0];
    }
}

//HOST
int main(){
    int row=512;   
    int col=512;

    float* matrix= (float*) malloc(sizeof(float)*row*col);
    float* matrix_DEVICE= NULL;
    float* array_DEVICE= NULL;

    float* array=new float[col];
    for(int i=0;i<row;i++){
        for(int j=0; j<col;j++){
            matrix[i*col+j]=j;
        }
    }

    cudaMalloc((void**)&matrix_DEVICE,sizeof(float)*row*col);
    cudaMalloc((void**)&array_DEVICE, col*sizeof(float));

    cudaMemcpy(matrix_DEVICE,matrix,sizeof(float)*row*col,cudaMemcpyHostToDevice);
    dim3 dimGrid(col,1);
    dim3 dimBlock(row/DIMBLOCKX,1);
    
    kernelSum_Column_Matrix<<< dimGrid , dimBlock >>>(matrix_DEVICE,array_DEVICE,col);
    
    cudaMemcpy(array,array_DEVICE,sizeof(float)*col,cudaMemcpyDeviceToHost);
    for( int index = 0; index<col ; index++){
		cout<<array[index]<<"  ";
    }
    
    cudaFree(matrix_DEVICE);
    cudaFree(array_DEVICE);

    delete[] array;
    delete[] matrix;
    
}