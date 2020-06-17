#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cusolverDn.h>
#include <cuComplex.h>

void Tensor_to_cuComplex(torch::Tensor real, torch::Tensor imag, cuComplex *dst, int len);
void cuComplex_to_Tensor(torch::Tensor real, torch::Tensor imag, cuComplex *dst, int len);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> tcs_Rsolve(
    torch::Tensor input_A,
    torch::Tensor input_b,
    int pivot_on
){
    CHECK_INPUT(input_A);
    CHECK_INPUT(input_b);
    // pivot_on 1:True 0:False

    const int m = input_A.size(0);
    const int nrhs = input_b.size(0);
    // m means matrix_size
    // nrhs = k
    
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    float *d_work = NULL; /* device workspace for getrf */

/* step 1: create cusolver handle, bind a stream */
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;

    cusolverDnHandle_t cusolverH = NULL;
    auto stream = at::cuda::getCurrentCUDAStream();

    status = cusolverDnCreate(&cusolverH);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnSetStream(cusolverH, stream);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: copy to device */
    auto d_A = input_A.data_ptr<float>();
    auto d_B = input_b.data_ptr<float>();

    cudaStat1 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
    AT_CHECK(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMalloc ((void**)&d_info, sizeof(int));
    AT_CHECK(cudaSuccess == cudaStat1);

/* step 3: query working space of getrf */
    // status = cusolverDnDgetrf_bufferSize(
    status = cusolverDnSgetrf_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        m,
        &lwork);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    AT_CHECK(cudaSuccess == cudaStat1);

/* step 4: LU factorization */
    if (pivot_on){
        // status = cusolverDnDgetrf(
        status = cusolverDnSgetrf(
            cusolverH,
            m,
            m,
            d_A,
            m,
            d_work,
            d_Ipiv,
            d_info);
    }else{
        // status = cusolverDnDgetrf(
        status = cusolverDnSgetrf(
            cusolverH,
            m,
            m,
            d_A,
            m,
            d_work,
            NULL,
            d_info);
    }
    // cudaStat1 = cudaDeviceSynchronize();
    cudaStat1 = cudaStreamSynchronize(stream);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    AT_CHECK(cudaSuccess == cudaStat1);
 
/*
 * step 5: solve A*X = B 
 *       | 1 |       | -0.3333 |
 *   B = | 2 |,  X = |  0.6667 |
 *       | 3 |       |  0      |
 *
 */
    if (pivot_on){
        status = cusolverDnSgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            nrhs, /* nrhs */
            d_A,
            m,
            d_Ipiv,
            d_B,
            m,
            d_info);
    }else{
        status = cusolverDnSgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            nrhs, /* nrhs */
            d_A,
            m,
            NULL,
            d_B,
            m,
            d_info);
    }
    cudaStat1 = cudaStreamSynchronize(stream);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    AT_CHECK(cudaSuccess == cudaStat1);

/* free resources */
    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);

    return {input_b,input_A};
}

std::vector<torch::Tensor> tcs_Csolve(
    torch::Tensor input_A_r,
    torch::Tensor input_A_i,
    torch::Tensor input_b_r,
    torch::Tensor input_b_i,
    int pivot_on
){
    CHECK_INPUT(input_A_r);
    CHECK_INPUT(input_A_i);
    CHECK_INPUT(input_b_r);
    CHECK_INPUT(input_b_i);
    // pivot_on 1:True 0:False

    const int m = input_A_r.size(0);
    const int nrhs = input_b_r.size(0);
    const int A_len = m*m;
    const int b_len = nrhs*m;
    
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    cuComplex *d_work = NULL; /* device workspace for getrf */

/* step 1: create cusolver handle, bind a stream */
    cusolverDnHandle_t cusolverH = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    status = cusolverDnCreate(&cusolverH);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    cudaStream_t stream = NULL;
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    AT_CHECK(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: copy A to device */
    cuComplex *d_A = NULL;
    cuComplex *d_B = NULL;
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(cuComplex) * A_len);
    AT_CHECK(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMalloc ((void**)&d_B, sizeof(cuComplex) * b_len);
    AT_CHECK(cudaSuccess == cudaStat1);
    // implement here tensor -> cuComplex
    Tensor_to_cuComplex(input_A_r, input_A_i, d_A, A_len);
    Tensor_to_cuComplex(input_b_r, input_b_i, d_B, b_len);

    cudaStat1 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
    AT_CHECK(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMalloc ((void**)&d_info, sizeof(int));
    AT_CHECK(cudaSuccess == cudaStat1);

/* step 3: query working space of getrf */
    // status = cusolverDnDgetrf_bufferSize(
    status = cusolverDnCgetrf_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        m,
        &lwork);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuComplex)*lwork);
    AT_CHECK(cudaSuccess == cudaStat1);

/* step 4: LU factorization */
    if (pivot_on){
        // status = cusolverDnDgetrf(
        status = cusolverDnCgetrf(
            cusolverH,
            m,
            m,
            d_A,
            m,
            d_work,
            d_Ipiv,
            d_info);
    }else{
        // status = cusolverDnDgetrf(
        status = cusolverDnCgetrf(
            cusolverH,
            m,
            m,
            d_A,
            m,
            d_work,
            NULL,
            d_info);
    }
    // cudaStat1 = cudaDeviceSynchronize();
    cudaStat1 = cudaStreamSynchronize(stream);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    AT_CHECK(cudaSuccess == cudaStat1);
 
/*
 * step 5: solve A*X = B 
 *       | 1 |       | -0.3333 |
 *   B = | 2 |,  X = |  0.6667 |
 *       | 3 |       |  0      |
 *
 */
    if (pivot_on){
        status = cusolverDnCgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            nrhs, /* nrhs */
            d_A,
            m,
            d_Ipiv,
            d_B,
            m,
            d_info);
    }else{
        status = cusolverDnCgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            nrhs, /* nrhs */
            d_A,
            m,
            NULL,
            d_B,
            m,
            d_info);
    }
    // cudaStat1 = cudaDeviceSynchronize();
    cudaStat1 = cudaStreamSynchronize(stream);
    AT_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    AT_CHECK(cudaSuccess == cudaStat1);

    auto x_r = torch::empty_like(input_b_r);
    auto x_i = torch::empty_like(input_b_i);
    auto lu_r = torch::empty_like(input_A_r);
    auto lu_i = torch::empty_like(input_A_i);
    // implement here cuComplex -> tensor
    cuComplex_to_Tensor(x_r, x_i, d_B, b_len);
    cuComplex_to_Tensor(lu_r, lu_i, d_A, A_len);

/* free resources */
    if (d_A ) cudaFree(d_A);
    if (d_B ) cudaFree(d_B);
    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);

    return {x_r,x_i,lu_r,lu_i};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("Rsolve", &tcs_Rsolve, "TorchComplexSolve Real matrix solve (cuSolver) for debug");
    m.def("Csolve", &tcs_Csolve, "TorchComplexSolve Complex matrix solve (cuSolver)");
}