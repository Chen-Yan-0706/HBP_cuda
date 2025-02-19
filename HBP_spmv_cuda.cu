#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "mmio.h"
#include "math.h"
#include <cuda_runtime.h>
#include <iostream>
#include "error.cuh"
#define BLOCK_ROW_M 512
#define BLOCK_COL_N 4096
#define HBP_M 512
#define HBP_N 4096
#define PARALLEL 8192
#define OMEGA 32
#define BUFFER_SIZE 1024
#define SUB_N 10
# define EPSILON 1
# define recycle 10
using namespace std;


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

__device__
void hbp_hash(int *output_hash,const int a,int end,int basic_row,int row)                                                               
{
    int i;
    if(a >= end)
    {        
        for(i = 1;(output_hash[basic_row + end - i] != -1);++i)
        {
            ;
        }
        output_hash[basic_row + end - i] = row;
    }
    else
    {
        for(i = 0;i < end;++i)
        {
            if(a + i < end && output_hash[basic_row + a + i] == -1)
            {
                break;
            }
            else if(a - i >= 0 && output_hash[basic_row + a - i] == -1)
            {
                i = -i;
                break;
            }
        }
        output_hash[basic_row + a + i] = row;
    }
}

__device__
int binary_search(const int *old_cols,int begin,int end,int target)                                                                                        
{                                                                                                                                                                                                                                                                
    int median; 
    int start = begin;
    int stop = end;                                                                                                                                                                                                                                                                            
    while (start <= stop)                                                                                                                     
    {                                                                                                                                         
        median = (start + stop + 1) / 2;                                                                                                                                                                                                                                                          
        if (old_cols[median] <= target)                                                                                                          
            start = median + 1;                                                                                                               
        else                                                                                                                                  
            stop = median - 1;                                                                                                                
    }                                                                                                                                                                                                                                                                                    
    return start;                                                                                                                             
}                                                                                                                                             
                                                                                                                                                
__global__
void Hash_partition(const int* old_ptr,
                    const int* old_cols,
                    const double* old_data,
                    const int number_of_rows,
                    const int number_of_nonzeroes,
                    const int block_col_num,
                    const int block_row_num,
                    double sparsity,
                    int* new_cols,
                    double* new_data,
                    int* output_hash,
                    int* add_sign,
                    int* zero_row,
                    int* beginptr)
{
    for(int id = blockIdx.x * blockDim.x + threadIdx.x;id < block_col_num * block_row_num;id+=PARALLEL)
    {
        int block_m = id / block_col_num;
        int block_n = id % block_col_num;
        int i;
        int j;
        int begin_col = block_n * HBP_N;
        int begin_ptr = block_m * HBP_M;
        int begin_nnz[HBP_M];
        int num_per_row[HBP_M];
        int currow_nnz;
        int count_nnz = 0;
        int end_row = (block_m >= block_row_num - 1)?number_of_rows % HBP_M:HBP_M ;
        int basic_beginptr = block_m * (HBP_M/OMEGA) * block_col_num + block_n * ((end_row%OMEGA == 0)?end_row/OMEGA:end_row/OMEGA + 1) + 1;
        int basic_ptr = block_m * HBP_M * block_col_num + block_n * end_row + 1;
        for(i = 0;i < end_row;++i)
        {
            num_per_row[i] = 0;
            currow_nnz = old_ptr[begin_ptr + i +1] - old_ptr[begin_ptr + i];
            if(currow_nnz == 0)
            {
                begin_nnz[i] = old_ptr[begin_ptr + i];
            }
            else
            {
                j = binary_search(old_cols,old_ptr[begin_ptr + i],old_ptr[begin_ptr + i +1] -1,begin_col - 1);
                begin_nnz[i] = j;
                count_nnz += j - old_ptr[begin_ptr + i];
                for(;(old_cols[j] < begin_col + HBP_N) && (j < old_ptr[begin_ptr + i +1]);++j)
                {
                    ++num_per_row[i];
                }
            }
        }

        for(i = 0;i < end_row;++i)
        {
            //int a = ((num_per_row[i] >> 3) << 8) + (num_per_row[i] / 6) * 100;
            // int a = ((num_per_row[i]>> 2) << 9); 
            int a = ((num_per_row[i]>> 4) << 8);   
            // int b;
            // if(num_per_row[i] < 11)
            //     b = 0;
            // else
            //     b = num_per_row[i] - 11;
            // int a = ((b >> 2) << 7) + (b << 3);
            //int a = ((num_per_row[i] / 200) << 8) + ((num_per_row[i] / 100)<<7) + ((num_per_row[i] / 50)<<6);
            hbp_hash(output_hash,a,end_row,basic_ptr - 1,i);
        }
        int basic_ptr_num = old_ptr[begin_ptr];
         
        for(i = 0;i < end_row / OMEGA;++i)
        {
            int count_zerorow = 0;
            j = OMEGA;
            for(int k =0;k < OMEGA;++k)
            {
                if(num_per_row[output_hash[basic_ptr -1 + i * OMEGA + k]] == 0)
                {
                    num_per_row[output_hash[basic_ptr -1 + i * OMEGA + k]] = -1;
                    --j;
                    zero_row[basic_ptr + i * OMEGA + k - 1] = -1;
                    count_zerorow++;
                }
                else
                {
                    zero_row[basic_ptr + i * OMEGA + k - 1] = count_zerorow;
                }
            }
            if(j == 0)
            {
                beginptr[basic_beginptr + i] = basic_ptr_num + count_nnz;
            }
            else
            {
                while(j > 0)
                {
                    for(int k = 0;k < OMEGA;++k)
                    {
                        int hash_row = output_hash[basic_ptr - 1 + i * OMEGA + k];
                        if(num_per_row[hash_row] > 0)
                        {
                            new_cols[basic_ptr_num + count_nnz] = old_cols[begin_nnz[hash_row]];
                            new_data[basic_ptr_num + count_nnz] = old_data[begin_nnz[hash_row]];
                            add_sign[basic_ptr_num + count_nnz] = j;
                            --num_per_row[hash_row];
                            begin_nnz[hash_row]++;
                            ++count_nnz;
                        }
                        if(num_per_row[hash_row] == 0)
                        {
                            add_sign[basic_ptr_num + count_nnz - 1] = -1;
                            num_per_row[hash_row] = -1;
                            --j;
                        }
                    }
                }
                beginptr[basic_beginptr + i] = basic_ptr_num + count_nnz;
            }
        }
        if(end_row % OMEGA)
        {
            int end = end_row % OMEGA;
            j = end;
            int count_zerorow = 0;
            for(int k =0;k < end;++k)
            {
                if(num_per_row[output_hash[basic_ptr -1 + (end_row / OMEGA) * OMEGA + k]] == 0)
                {
                    num_per_row[output_hash[basic_ptr -1 + (end_row / OMEGA) * OMEGA + k]] = -1;
                    --j;
                    zero_row[basic_ptr + (end_row / OMEGA) * OMEGA + k - 1] = -1;
                    count_zerorow++;
                }
                else
                {
                    zero_row[basic_ptr + (end_row / OMEGA) * OMEGA + k - 1] = count_zerorow;
                }
            }

            while(j > 0)
            {
                for(int k = 0;k < end;++k)
                {
                    int hash_row = output_hash[basic_ptr - 1 + (end_row / OMEGA) * OMEGA + k];
                    if(num_per_row[hash_row] > 0)
                    {
                        new_cols[basic_ptr_num + count_nnz] = old_cols[begin_nnz[hash_row]];
                        new_data[basic_ptr_num + count_nnz] = old_data[begin_nnz[hash_row]];
                        add_sign[basic_ptr_num + count_nnz] = j;
                        --num_per_row[hash_row];
                        begin_nnz[hash_row]++;
                        ++count_nnz;
                    }
                    if(num_per_row[hash_row] == 0)
                    {
                        add_sign[basic_ptr_num + count_nnz - 1] = -1;
                        num_per_row[hash_row] = -1;
                        --j;
                    }
                }
            }
            
        beginptr[basic_beginptr + end_row / OMEGA] = basic_ptr_num + count_nnz;
        }
    }
}

__device__
double basiccore(const int     *cols,
                 const double  *data,
                 const double  *vect,
                 const int     *add_sign,
                 int beginj)
{
    int j = beginj;
    double sum = 0;
    for (int r_add_sign = add_sign[j];r_add_sign > 0; r_add_sign = add_sign[j])
    {
        sum += data[j] * vect[cols[j]];
        j += r_add_sign;
    }
    sum += data[j] * vect[cols[j]];
    return sum;
}

__device__
double block_basiccore(const int     *cols,
                       const double  *data,
                       const double  *vect,
                       const int     *add_sign,
                       const int beginj)
{
    int j = beginj;
    double sum = 0;
    for (int r_add_sign = add_sign[j];r_add_sign > 0; r_add_sign = add_sign[j])
    {
        sum += data[j] * vect[cols[j] % HBP_N];
        j += r_add_sign;
    }
    sum += data[j] * vect[cols[j] % HBP_N];
    return sum;
}
__device__
void spmv(const int* cols,
          const double* data,
          const double* vect,
          const int* output_hash,
          double *output,
          int block_m,
          int block_n,
          int begin,
          int end_row,
          int q,
          const int* add_sign,
          const int* zero_row,
          const int *beginptr,
          int block_col_num)
{
    int h = (end_row % OMEGA == 0)?end_row / OMEGA:end_row / OMEGA + 1;
    for (int i = q; i < end_row; i += OMEGA)
    {
        if(zero_row[begin + i] == -1)
        {output[begin + output_hash[begin + i]] = 0;}
        else
        {
            int beginj = beginptr[block_m * (HBP_M / OMEGA) * block_col_num + block_n * h + i/OMEGA] + q - zero_row[begin + i];
            output[begin + output_hash[begin + i]] = basiccore(cols,
                                                                data,
                                                                vect,
                                                                add_sign,
                                                                beginj);               
        }
    }
}
__device__
void block_spmv(const int* cols,
          const double* data,
          const double* vect,
          const int* output_hash,
          double *output,
          int block_m,
          int block_n,
          int begin,
          int end_row,
          int q,
          const int* add_sign,
          const int* zero_row,
          const int *beginptr,
          int block_col_num)
{
    int h = (end_row % OMEGA == 0)?end_row / OMEGA:end_row / OMEGA + 1;
    for (int i = q; i < end_row; i += OMEGA)
    {
        if(zero_row[begin + i] == -1)
        {output[begin + output_hash[begin + i]] = 0;}
        else
        {
            int beginj = beginptr[block_m * (HBP_M / OMEGA) * block_col_num + block_n * h + i/OMEGA] + q - zero_row[begin + i];
            output[begin + output_hash[begin + i]] = block_basiccore(cols,
                                                                    data,
                                                                    vect,
                                                                    add_sign,
                                                                    beginj);                                                               
        }
    }
}

__global__
void basic_spmv(const int* cols,
                const double* data,
                const double* vect,
                const int* output_hash,
                double* output,
                const int block_col_num,
                const int block_row_num,
                const int number_of_rows,
                const int blocknum_per_thread,
                const int res,
                const int* add_sign,
                const int* zero_row,
                const int* beginptr,
                int *d_res)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = p % OMEGA;
    int pblock_m = ((p / OMEGA) / block_col_num) * blocknum_per_thread;
    int block_end = (pblock_m == blocknum_per_thread * (block_row_num /blocknum_per_thread))?block_row_num % blocknum_per_thread:blocknum_per_thread;
    int block_n = (p / OMEGA) % block_col_num;
    int block_m;
    int end_row;
    int begin;
    __shared__ int res_line;
    __shared__ double block_vect[HBP_N];
    for(int i = 0;i < HBP_N / OMEGA;++i)
    {
        block_vect[i * OMEGA + q] = vect[block_n * HBP_N + i * OMEGA + q];
    }
    for(int i = 0;i < block_end; ++i)
    {
        block_m = pblock_m + i;
        end_row = (block_m >= block_row_num - 1)?number_of_rows % HBP_M:HBP_M;
        begin = block_m * HBP_M * block_col_num + block_n * end_row;
        block_spmv(cols,
             data,
             block_vect,
             output_hash,
             output,
             block_m,
             block_n,
             begin,
             end_row,
             q,
             add_sign,
             zero_row,
             beginptr,
             block_col_num);
    }

    if(q == 0)
    {
        res_line = atomicSub(&d_res[0],SUB_N) - 1; 
    }
    __syncthreads();

    while(res_line >= 0)
    {
        int compete_blocknum;
        if (res_line >= SUB_N - 1) 
            compete_blocknum = SUB_N - 1;
        else
            compete_blocknum = res_line;

        while(compete_blocknum >= 0)
        {
            int predict = (res_line - compete_blocknum) - blocknum_per_thread * (block_col_num-((PARALLEL / OMEGA) % block_col_num));
            if(predict < 0)
            {
                block_m = blocknum_per_thread * ((PARALLEL / OMEGA) / block_col_num) + (res_line - compete_blocknum) / (block_col_num-((PARALLEL / OMEGA) % block_col_num));
                block_n = ((PARALLEL / OMEGA) % block_col_num) + (res_line - compete_blocknum) % (block_col_num-((PARALLEL / OMEGA) % block_col_num));
            }
            else
            {
                block_m = blocknum_per_thread * ceil((double)(PARALLEL / OMEGA)/ (double)block_col_num) + predict / block_col_num;
                block_n = predict % block_col_num;
            }
            end_row = (block_m >= block_row_num - 1)?number_of_rows % HBP_M:HBP_M;
            begin = block_m * HBP_M * block_col_num + block_n * end_row;
            spmv(cols,
                data,
                vect,
                output_hash,
                output,
                block_m,
                block_n,
                begin,
                end_row,
                q,
                add_sign,
                zero_row,
                beginptr,
                block_col_num);
            --compete_blocknum;
        }
        if(q == 0)
        {
            res_line = atomicSub(&d_res[0],SUB_N) - 1;
        }
        __syncthreads(); 
    }
}

// __global__
// void basic_spmv(const int* cols,
//                 const double* data,
//                 const double* vect,
//                 const int* output_hash,
//                 double* output,
//                 const int block_col_num,
//                 const int block_row_num,
//                 const int number_of_rows,
//                 const int blocknum_per_thread,
//                 const int res,
//                 const int* add_sign,
//                 const int* zero_row,
//                 const int* beginptr,
//                 int *d_res,
//                 int col_extension)
// {
//     int p = blockIdx.x * blockDim.x + threadIdx.x;
//     int q = threadIdx.x;
//     int pblock_m = ((p / OMEGA) / block_col_num) * blocknum_per_thread;
//     int block_end = (pblock_m == blocknum_per_thread * (block_row_num /blocknum_per_thread))?block_row_num % blocknum_per_thread:blocknum_per_thread;
//     int block_n = (p / OMEGA * col_extension) % block_col_num;
//     int block_m;
//     int end_row;
//     int begin;
    
//     __shared__ int res_line;
//     __shared__ double block_vect[HBP_N];
//     for(int id = 0;id < col_extension;++id)
//     {
//         for(int i = 0;i < HBP_N / OMEGA;++i)
//         {
//             block_vect[i * OMEGA + q] = vect[block_n * HBP_N + i * OMEGA + q];
//         }
//         for(int i = 0;i < block_end; ++i)
//         {
//             block_m = pblock_m + i;
//             end_row = (block_m >= block_row_num - 1)?number_of_rows % HBP_M:HBP_M;
//             begin = block_m * HBP_M * block_col_num + block_n * end_row;
//             block_spmv(cols,
//                 data,
//                 block_vect,
//                 output_hash,
//                 output,
//                 block_m,
//                 block_n,
//                 begin,
//                 end_row,
//                 q,
//                 add_sign,
//                 zero_row,
//                 beginptr,
//                 block_col_num);
//             // spmv(cols,
//             //     data,
//             //     vect,
//             //     output_hash,
//             //     output,
//             //     block_m,
//             //     block_n,
//             //     begin,
//             //     end_row,
//             //     q,
//             //     add_sign,
//             //     zero_row,
//             //     beginptr,
//             //     block_col_num);
//         }
//         block_n++;
//         __syncwarp();
//     }

//     if(q == 0)
//     {
//         res_line = atomicSub(&d_res[0],SUB_N) - 1; 
//     }
//     __syncthreads();

//     while(res_line >= 0)
//     {
//         int compete_blocknum;
//         if (res_line >= SUB_N - 1) 
//             compete_blocknum = SUB_N - 1;
//         else
//             compete_blocknum = res_line;

//         while(compete_blocknum >= 0)
//         {
//             int predict = (res_line - compete_blocknum) - blocknum_per_thread * (block_col_num-((PARALLEL / OMEGA) * col_extension % block_col_num));
//             if(predict < 0)
//             {
//                 block_m = blocknum_per_thread * ((PARALLEL / OMEGA) * col_extension / block_col_num) + (res_line - compete_blocknum) / (block_col_num-((PARALLEL / OMEGA) * col_extension % block_col_num));
//                 block_n = ((PARALLEL / OMEGA) * col_extension % block_col_num) + (res_line - compete_blocknum) % (block_col_num-((PARALLEL / OMEGA) * col_extension % block_col_num));
//             }
//             else
//             {
//                 block_m = blocknum_per_thread * ceil((double)(PARALLEL / OMEGA)/ (double)block_col_num) + predict / block_col_num;
//                 block_n = predict % block_col_num;
//             }
//             end_row = (block_m >= block_row_num - 1)?number_of_rows % HBP_M:HBP_M;
//             begin = block_m * HBP_M * block_col_num + block_n * end_row;
//             spmv(cols,
//                 data,
//                 vect,
//                 output_hash,
//                 output,
//                 block_m,
//                 block_n,
//                 begin,
//                 end_row,
//                 q,
//                 add_sign,
//                 zero_row,
//                 beginptr,
//                 block_col_num);
//             --compete_blocknum;
//         }
//         if(q == 0)
//         {
//             res_line = atomicSub(&d_res[0],SUB_N) - 1;
//         }
//         __syncthreads(); 
//     }
// }

__global__
void basic_combine(const double* data,double *output,const int block_col_num,const int block_row_num,const int number_of_rows)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    while(p < number_of_rows)
    {
        double sum = 0;
        int write_loc = (p / HBP_M) * HBP_M *block_col_num + p % HBP_M;
        int end_row = ((p / HBP_M)>=block_row_num - 1)?number_of_rows % HBP_M:HBP_M;
        for (int j = 0; j < block_col_num;  ++j)
        { 
            sum += data[write_loc];
            write_loc += end_row;
        }
        output[p] = sum;
        p += PARALLEL;        
    }
}

bool check(double *result_cpu, double *result_gpu,int number_of_rows)
{
    for (int i = 0; i < number_of_rows; ++i)
    {
        if (fabs(result_cpu[i] - result_gpu[i]) > EPSILON)
        {
            printf("wrong value at index %d: expected %f - calculated %f\n", i, result_cpu[i], result_gpu[i]);
            return false;
        }
    }
    free(result_cpu);
    free(result_gpu);
    return true;
}

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
    int *d_HBP_cols;
    int *d_HBP_addsign;
    int *d_HBP_zero_row;
    double *d_HBP_data;
    int* d_HBP_output_hash;
    int *d_HBP_beginptr;
    double sparsity;
    const char *filename = "databases/ASIC_320k.mtx";
    struct timespec start_time;
    struct timespec end_time;

    /* get GPU device properties*/

    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&devprop,0);
    std::cout << "global memary size : " << devprop.totalGlobalMem / 1024 / 1024 << "MB" << endl;
    std::cout << "shared memory per block : " << devprop.sharedMemPerBlock / 1024.0<< "KB" << endl;
    std::cout << "max threads per block : " << devprop.maxThreadsPerBlock << endl;
    std::cout << "max threads per processor : " << devprop.maxThreadsPerMultiProcessor << endl;

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

    //ptr[number_of_rows] = number_of_nonzeroes-1;
    int pervious_row = 0;
    
    for (i = 0; i < number_of_nonzeroes; ++i)
    {
        int current_row;
        fscanf(file, "%d %d %lg\n", &csr_cols[i], &current_row, &csr_data[i]);
        // adjust from 1-based to 0-based
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
    sparsity = number_of_nonzeroes / (number_of_rows * number_of_columns);
    printf("read data finish \n");

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

    //2D partition

    int block_col_num = ceil((double)number_of_columns/(double)HBP_N);//列块数
    int block_row_num = ceil((double)number_of_rows/(double)HBP_M);//行块数

    printf("block_col_num = %d , block_row_num = %d \n",block_col_num,block_row_num);

    // int col_extension = 1;
    // int threadnum_per_col = ceil((double) (PARALLEL/OMEGA) /(double)block_col_num);
    // if(threadnum_per_col == 1)
    // {
    //     col_extension = block_col_num / (PARALLEL / OMEGA);
    // }
    // int blocknum_per_thread = block_row_num / threadnum_per_col;
    // int parallel_blocknum = (threadnum_per_col == 1)?blocknum_per_thread:((PARALLEL / OMEGA) / block_col_num) * blocknum_per_thread;
    // printf("threadnum_per_col = %d,blocknum_per_thread = %d,parallel_blocknum = %d \n",threadnum_per_col,blocknum_per_thread,parallel_blocknum);
    // int res;
    // if((block_row_num - parallel_blocknum >= blocknum_per_thread) || threadnum_per_col == 1)
    //     res = block_col_num * block_row_num - (PARALLEL / OMEGA) * blocknum_per_thread * col_extension;
    // else
    //     res = block_col_num * block_row_num - parallel_blocknum * block_col_num - ((PARALLEL / OMEGA) % block_col_num) * (block_row_num - parallel_blocknum);

    // printf("col_extension = %d \n",col_extension);

    int threadnum_per_col = ceil((double) (PARALLEL/OMEGA) /(double)block_col_num);
    int blocknum_per_thread = block_row_num / threadnum_per_col;
    int parallel_blocknum = ((PARALLEL / OMEGA) / block_col_num == 0)?blocknum_per_thread:((PARALLEL / OMEGA) / block_col_num) * blocknum_per_thread;
    printf("threadnum_per_col = %d,blocknum_per_thread = %d,parallel_blocknum = %d \n",threadnum_per_col,blocknum_per_thread,parallel_blocknum);
    int res;
    if((block_row_num - parallel_blocknum >= blocknum_per_thread) || threadnum_per_col == 1)
        res = block_col_num * block_row_num - (PARALLEL / OMEGA) * blocknum_per_thread;
    else
        res = block_col_num * block_row_num - parallel_blocknum * block_col_num - ((PARALLEL / OMEGA) % block_col_num) * (block_row_num - parallel_blocknum);
    
    printf("res = %d \n",res);
    printf("readdata finish!\n");
    int memsize_HBP_beginptr = block_col_num * ceil((double)number_of_rows/(double)OMEGA) + 1;
    int *HBP_output_hash;
    HBP_output_hash = (int *)malloc(block_col_num * number_of_rows * sizeof(int));   
    for(i= 0; i < block_col_num * number_of_rows; ++i)
    {
        HBP_output_hash[i] = -1;
    }
    CHECK(cudaMalloc((void **)&d_csr_ptr,(number_of_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_csr_cols,number_of_nonzeroes * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_csr_data,number_of_nonzeroes * sizeof(double)));
    CHECK(cudaMemcpy(d_csr_ptr,csr_ptr,(number_of_rows + 1) * sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr_cols,csr_cols,number_of_nonzeroes * sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr_data,csr_data,number_of_nonzeroes * sizeof(double),cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void **)&d_HBP_cols,number_of_nonzeroes * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_HBP_data,number_of_nonzeroes * sizeof(double)));
    CHECK(cudaMalloc((void **)&d_HBP_addsign,number_of_nonzeroes * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_HBP_zero_row,block_col_num * number_of_rows * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_HBP_output_hash,block_col_num * number_of_rows * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_HBP_beginptr,memsize_HBP_beginptr * sizeof(int)));
    CHECK(cudaMemcpy(d_HBP_output_hash,HBP_output_hash,block_col_num * number_of_rows * sizeof(int),cudaMemcpyHostToDevice));

    Hash_partition<<<PARALLEL/OMEGA,32>>>(d_csr_ptr,d_csr_cols,d_csr_data,
                                number_of_rows,
                                number_of_nonzeroes,
                                block_col_num,
                                block_row_num,
                                sparsity,
                                d_HBP_cols,
                                d_HBP_data,
                                d_HBP_output_hash,
                                d_HBP_addsign,
                                d_HBP_zero_row,
                                d_HBP_beginptr);
    double partition_ms = 0.0;
    
    for(int k = 0; k < 1; ++k)
    {
        cudaMemcpy(d_HBP_output_hash,HBP_output_hash,block_col_num * number_of_rows * sizeof(int),cudaMemcpyHostToDevice);
        //clock_gettime(1, &start_time);
        hbp_timer partition_time;
        partition_time.start();
        Hash_partition<<<PARALLEL/OMEGA,32>>>(d_csr_ptr,d_csr_cols,d_csr_data,
                                number_of_rows,
                                number_of_nonzeroes,
                                block_col_num,
                                block_row_num,
                                sparsity,
                                d_HBP_cols,
                                d_HBP_data,
                                d_HBP_output_hash,
                                d_HBP_addsign,
                                d_HBP_zero_row,
                                d_HBP_beginptr);
        //clock_gettime(1, &end_time);
        partition_ms += partition_time.stop();
        //partition_ms += (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000 + (double)(end_time.tv_sec - start_time.tv_sec) * 1000;
        //printf("CSR->HBP %d / %d\n",k,recycle);
    }
    free(csr_ptr);
    free(csr_cols);
    free(csr_data);
    cudaFree(d_csr_ptr);
    cudaFree(d_csr_cols);
    cudaFree(d_csr_data);
    printf("CSR->HBP finish!\n");

    double *d_HBP_output;

    int *d_res;
    int *h_res;
    h_res = (int*)malloc(sizeof(int) * 1);
    h_res[0] = res;
    CHECK(cudaMalloc((void **)&d_res,1 * sizeof(int)));
    CHECK(cudaMemcpy(d_res,h_res,1 * sizeof(int),cudaMemcpyHostToDevice));

    double *d_mid_output;
    CHECK(cudaMalloc((void **)&d_mid_output,block_col_num * number_of_rows * sizeof(double)));

    double *d_vect;
    CHECK(cudaMalloc((void **)&d_vect,number_of_columns * sizeof(double)));
    CHECK(cudaMemcpy(d_vect,vect,number_of_columns * sizeof(double),cudaMemcpyHostToDevice));

    // basic_spmv<<<32,32>>>(d_HBP_cols,d_HBP_data,d_vect,
    //                             d_HBP_output_hash,
    //                             d_mid_output,
    //                             block_col_num,
    //                             block_row_num,
    //                             number_of_rows,
    //                             blocknum_per_thread,
    //                             res,
    //                             d_HBP_addsign,
    //                             d_HBP_zero_row,
    //                             d_HBP_beginptr,
    //                             d_res,
    //                             col_extension);
    basic_spmv<<<PARALLEL/OMEGA,32>>>(d_HBP_cols,d_HBP_data,d_vect,
                                d_HBP_output_hash,
                                d_mid_output,
                                block_col_num,
                                block_row_num,
                                number_of_rows,
                                blocknum_per_thread,
                                res,
                                d_HBP_addsign,
                                d_HBP_zero_row,
                                d_HBP_beginptr,
                                d_res);
    double spmv_ms = 0.0;
    for(int k = 0; k < recycle; ++k)
    {
        hbp_timer spmv_time;
        spmv_time.start();
        basic_spmv<<<PARALLEL/OMEGA,32>>>(d_HBP_cols,d_HBP_data,d_vect,
                                    d_HBP_output_hash,
                                    d_mid_output,
                                    block_col_num,
                                    block_row_num,
                                    number_of_rows,
                                    blocknum_per_thread,
                                    res,
                                    d_HBP_addsign,
                                    d_HBP_zero_row,
                                    d_HBP_beginptr,
                                    d_res);
        spmv_ms += spmv_time.stop();
    }
    //cudaMemcpy(mid_output,d_mid_output,block_col_num * number_of_rows * sizeof(double),cudaMemcpyDeviceToHost);
    printf("HBP_spmv finish!\n");
    cudaFree(d_HBP_cols);
    cudaFree(d_HBP_data);
    cudaFree(d_HBP_addsign);
    cudaFree(d_HBP_zero_row);
    cudaFree(d_HBP_output_hash);
    cudaFree(d_HBP_beginptr);
    cudaFree(d_res);
    cudaFree(d_vect);


    double *HBP_output;
    HBP_output = (double *)malloc(number_of_rows * sizeof(double));
    cudaMalloc((void **)&d_HBP_output,number_of_rows * sizeof(double));
    basic_combine<<<PARALLEL/OMEGA,32>>>(d_mid_output,d_HBP_output,block_col_num,block_row_num,number_of_rows);
    double combine_ms = 0.0;
    for(int k = 0; k < recycle; ++k)
    {
        hbp_timer combine_time;
        combine_time.start();
        basic_combine<<<PARALLEL/OMEGA,32>>>(d_mid_output,d_HBP_output,block_col_num,block_row_num,number_of_rows);
        combine_ms += combine_time.stop();
        //printf("HBP_combine %d / %d\n",k,recycle);
    }
    cudaFree(d_mid_output);

    cudaMemcpy(HBP_output,d_HBP_output,number_of_rows * sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(d_HBP_output);
    printf("CSR->HBP finish! partition time : %lf recycle : %d \n",partition_ms,recycle);
    printf("HBP_spmv finish! spmv time : %lf\n",spmv_ms);
    printf("HBP_combine finish! combine time : %lf\n",combine_ms);
    if(check(output_cpu,HBP_output,number_of_rows))
    {printf("HBP result is true! \n");}

    return 1;
}
