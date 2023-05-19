%%cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 16
#define A_HEIGHT 500
#define A_WIDTH 400
#define B_WIDTH 300

#define USECPSEC 1000000ULL

__host__ unsigned long long myCPUTimer(unsigned long long start=0){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

__host__ void printMatrix(int *A, int n, int m) {
    
   for (int i = 0; i < n; i ++) {
        
        for (int j = 0; j < m; j ++) {
            
            printf("%d ", A[i * m + j]);

        }

        printf("\n");

    }
    
}

__host__ void matrix_mult(int *A, int *B, int *C, int m, int n, int k) {
    
    for (int i = 0; i < m; i ++) {
        
        for (int j = 0; j < k; j ++) {
            
            int temp = 0;

            for (int l = 0; l < n; l ++) {
                temp += A[i * n + l] * B[l * k + j];
            }

            C[i * k + j] = temp;

        }

    }

}

__global__ void matrix_mult_kernel(int *A, int *B, int *C, int m, int n, int k) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int temp = 0;
        
        for (int i = 0; i < n; i ++) {
            temp += A[row * n + i] * B[i * k + col];
        }
        
        C[row * k + col] = temp;
        
    }

}

__global__ void matrix_mult_tiles(int *A, int *B, int *C, int m, int n, int k) {
    
    __shared__ int t_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int t_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
 
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    int temp = 0;

    for (int p = 0; p < gridDim.x; ++p) 
    {
        if(row < m && (p * BLOCK_SIZE + tx) < n)
        {
            t_A[ty][tx] = A[row * n + p * BLOCK_SIZE + tx];
        }
        else
        {
            t_A[ty][tx] = 0;
        }

        if((p * BLOCK_SIZE + ty) < n && col < k)
        {
            t_B[ty][tx] = B[(p * BLOCK_SIZE + ty) * k + col];
        }  
        else
        {
            t_B[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            temp += t_A[ty][k] * t_B[k][tx];
        }
        __syncthreads();
    }
    if(row < m && col < k)
    {
        C[row * k + col] = temp;
        
    }
 
}

int main() {

    int *A;
    int *B;
    int *C;
    int *D;

    int *d_A, *d_B, *d_C;

    cudaMallocHost((void **) &A, sizeof(int) * A_HEIGHT * A_WIDTH);
    cudaMallocHost((void **) &B, sizeof(int) * A_WIDTH * B_WIDTH);
    cudaMallocHost((void **) &C, sizeof(int) * A_HEIGHT * B_WIDTH);
    cudaMallocHost((void **) &D, sizeof(int) * A_HEIGHT * B_WIDTH);

    for (int i = 0; i < A_HEIGHT; i++) {
        for (int j = 0; j < A_WIDTH; j++) {
            A[i * A_WIDTH + j] = rand() % 9 + 1;
        }
    }

    for (int i = 0; i < A_WIDTH; i++) {
        for (int j = 0; j < B_WIDTH; j++) {
            B[i * B_WIDTH + j] = rand() % 9 + 1;
        }
    }

    cudaMalloc((void **) &d_A, sizeof(int) * A_HEIGHT * A_WIDTH);
    cudaMalloc((void **) &d_B, sizeof(int) * A_WIDTH * B_WIDTH);
    cudaMalloc((void **) &d_C, sizeof(int) * A_HEIGHT * B_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_A, A, sizeof(int) * A_HEIGHT * A_WIDTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * A_WIDTH * B_WIDTH, cudaMemcpyHostToDevice);

    dim3 DimGrid((B_WIDTH - 1) / BLOCK_SIZE + 1, (A_HEIGHT - 1) / BLOCK_SIZE + 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start);
    clock_t t;
    t = clock();
    matrix_mult_tiles<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, A_HEIGHT, A_WIDTH, B_WIDTH);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, sizeof(int) * A_HEIGHT * B_WIDTH, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printMatrix(C, A_HEIGHT, B_WIDTH);

    printf("Time: %f\n", time_taken);
   
    t = clock();
    matrix_mult(A, B, D, A_HEIGHT, A_WIDTH, B_WIDTH);
    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC;
    

    printMatrix(D, A_HEIGHT, B_WIDTH);
    printf("Time: %f\n", time_taken);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
