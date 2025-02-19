#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "mmio.h"
#include "math.h"
#include <cuda_runtime.h>
#include <iostream>
#define PARALLEL 8196
#define OMEGA 32

# define EPSILON 1
# define recycle 10

struct hbp_timer {
    cudaEvent_t start_event, stop_event;

    void start() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
        cudaDeviceSynchronize();
    }

    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
};

void calculate_and_print_performance(double ms, int number_of_nonzeroes)
{
    printf("Your calculations took %.2lf ms to run.\n", ms);
    printf("Number of operations %d, PERFORMANCE %lf GFlops\n",
           2 * number_of_nonzeroes,
           (2 * number_of_nonzeroes) / ms * 1e-6);
}

void calculate_and_print_speed(double ms, int number_of_nonzeroes)
{
    printf("GBytes transferred to processor %lf - %lf, speed %lf - %lf GB/s\n",
           number_of_nonzeroes * sizeof(double) * 1e-9,
           (2 * number_of_nonzeroes) * sizeof(double) * 1e-9,
           number_of_nonzeroes * sizeof(double) / (ms) * 1e-6,
           (2 * number_of_nonzeroes) * sizeof(double) / (ms) * 1e-6);
}

void compute_using_cpu(double *data, double *vect, int *ptr, int *cols, int number_of_rows, int number_of_nonzeroes, double **result)
{
    int i;
    struct timespec start_time;
    struct timespec end_time;

    clock_gettime(1, &start_time);

    for (i = 0; i < number_of_rows; ++i)
    {
        int j;

        for (j = ptr[i]; j < ptr[i+1]; ++j)
        {
            (*result)[i] += data[j] * vect[cols[j]];
        }
    }

    clock_gettime(1, &end_time);
    double ms = (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000 + (double)(end_time.tv_sec - start_time.tv_sec) * 1000;

    printf("\nCPU calculations\n");
    calculate_and_print_performance(ms, number_of_nonzeroes);
}

bool read_size_of_matrices_from_file(FILE *file, int *number_of_rows, int *number_of_columns, int *number_of_nonzeroes)
{
    MM_typecode matcode;
    
    if (file == NULL)
    {
        return false;
    }
    
    if (mm_read_banner(file, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return false;
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return false;
    }

    /* find out size of sparse matrix */
    if (mm_read_mtx_crd_size(file, number_of_rows, number_of_columns, number_of_nonzeroes) != 0) 
    {
        return false;
    }
    
    return true;
}

__global__
void csr(const int *ptr, const int *col, const double *data, const double *vect, double *output, const int N)
{
    int i;
    for (i = blockDim.x * blockIdx.x +threadIdx.x; i < N; i += PARALLEL)
    {
        double sum = 0;
        int j;
        //#pragma unroll
        for (j = ptr[i]; j < ptr[i+1]; ++j)
        {
            sum += data[j] * vect[col[j]];
        }
        output[i] = sum;
    }
}

bool check(double *result_cpu, double *result_gpu,int number_of_rows)
{
    for (int i = 0; i < number_of_rows; ++i)
    {
        if (fabs(result_cpu[i] - result_gpu[i]) > EPSILON)
        {
            printf("wrong value at index %d: expected %lf - calculated %f\n", i, result_cpu[i], result_gpu[i]);
            return false;
        }
    }
    free(result_cpu);
    free(result_gpu);
    return true;
}

int main(int argc, char *argv[])
{
    int number_of_rows;
    int number_of_columns;
    int number_of_nonzeroes;
    int i;
    FILE *file;

    //device
    int *d_csr_ptr;
    int *d_csr_cols;
    double *d_csr_data;
    const char *filename = "databases/ASIC_320k.mtx";

    /* prepare data for calculations */
    
    file = fopen(filename, "r");
    
    if (file == NULL) 
    {
        perror(filename);
        return 0;
    }
    
    if (read_size_of_matrices_from_file(file, &number_of_rows, &number_of_columns, &number_of_nonzeroes) == false)
    {
        fclose(file);
        return 0;
    }
    int *csr_ptr;
    int *csr_cols;
    double *csr_data;

    csr_ptr  = (int *)malloc((number_of_rows + 1) * sizeof(int));
    csr_cols = (int *)malloc(number_of_nonzeroes * sizeof(int));
    csr_data = (double *)malloc(number_of_nonzeroes * sizeof(double));

    for(i=0;i<number_of_rows+1;++i)
    {
        csr_ptr[i]=0;
    }

    int pervious_row = 0;
    
    for (i = 0; i < number_of_nonzeroes; ++i)
    {
        int current_row;
        fscanf(file, "%d %d %lg\n", &csr_cols[i], &current_row, &csr_data[i]);
        csr_cols[i]--; 

        if(pervious_row != current_row)
        {
            for(int j = pervious_row;j<current_row;++j)
            {
                csr_ptr[j] = i;
            }
        pervious_row = current_row;
        }
    }
    // for (i = 0; i < number_of_nonzeroes; ++i)
    // {
    //     int current_row;
    //     fscanf(file, "%d %d \n", &csr_cols[i], &current_row);
    //     // adjust from 1-based to 0-based
    //     if(i % 2 == 0)
    //         csr_data[i] = 3.1415926535e-12;
    //     else
    //         csr_data[i] = i;
    //     csr_cols[i]--; 

    //     if(pervious_row != current_row)
    //     {
    //         for(int j = pervious_row;j<current_row;++j)
    //         {
    //             csr_ptr[j] = i;
    //         }
    //     pervious_row = current_row;
    //     }
    // }

    fclose(file);
    printf("read data finish \n");

    cudaMalloc((void **)&d_csr_ptr,(number_of_rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_csr_cols,number_of_nonzeroes * sizeof(int));
    cudaMalloc((void **)&d_csr_data,number_of_nonzeroes * sizeof(double));
    cudaMemcpy(d_csr_ptr,csr_ptr,(number_of_rows + 1) * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_cols,csr_cols,number_of_nonzeroes * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_data,csr_data,number_of_nonzeroes * sizeof(double),cudaMemcpyHostToDevice);

    double *vect;    
    double *output_cpu;    
    vect = (double*)malloc(sizeof(double) * number_of_columns);
    for (i = 0; i < number_of_columns; ++i) 
    {
        vect[i] = 1;
    }
    output_cpu = (double*)malloc(sizeof(double) * number_of_rows);
    compute_using_cpu(csr_data, vect, csr_ptr, csr_cols, number_of_rows, number_of_nonzeroes, &output_cpu);
    printf("CPU compute finish!\n");

    double *d_output;
    cudaMalloc((void **)&d_output,number_of_rows * sizeof(double));

    double *d_vect;
    cudaMalloc((void **)&d_vect,number_of_columns * sizeof(double));
    cudaMemcpy(d_vect,vect,number_of_columns * sizeof(double),cudaMemcpyHostToDevice);

    double *csr_output;
    csr_output = (double *)malloc(number_of_rows * sizeof(double));
    for (i = 0; i < number_of_rows; ++i) 
    {
        csr_output[i] = 0;
    }

    csr<<<PARALLEL/OMEGA,32>>>(d_csr_ptr, d_csr_cols, d_csr_data, d_vect, d_output, number_of_rows);
    double csr_ms = 0.0;
    for(int k = 0; k < recycle; ++k)
    {
        cudaMemcpy(d_output,csr_output,number_of_rows * sizeof(double),cudaMemcpyHostToDevice);
        hbp_timer spmv_time;
        spmv_time.start();
        csr<<<PARALLEL/OMEGA,32>>>(d_csr_ptr, d_csr_cols, d_csr_data, d_vect, d_output, number_of_rows);
        csr_ms += spmv_time.stop();
    }
    cudaMemcpy(csr_output,d_output,number_of_rows * sizeof(double),cudaMemcpyDeviceToHost);

    check(output_cpu,csr_output,number_of_rows);
    cudaFree(d_csr_ptr);
    cudaFree(d_csr_cols);
    cudaFree(d_csr_data);
    cudaFree(d_vect);
    cudaFree(d_output);
    printf("csr_spmv finish! spmv time : %lf recycle: %d \n",csr_ms,recycle);
}
