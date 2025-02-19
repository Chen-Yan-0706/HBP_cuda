# HBP_cuda

-----

Here we provide an example of SpMV using the HBP format.
'HBP_spmv_cuda.cu' provides the conversion of CSR format to HBP format and SpMV. We set the number of threads to 8192 and the number of SpMV recycles to 10, which can balance out the calculated speed deviation due to accidental factors.

-----

## how to run this example

-----

The matrix data used for testing is placed in databases, and in this example, ASIC_320k matrices are provided, please decompress them before testing.

### compilation

'nvcc -o csr_spmv ./csr_spmv.cu'

'nvcc -o HBP_spmv_cuda ./HBP_spmv_cuda.cu'

### run

'./csr_spmv'

'./HBP_spmv_cuda'
