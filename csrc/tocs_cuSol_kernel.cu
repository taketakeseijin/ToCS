#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ref. https://discuss.pytorch.org/t/aten-transfer-tensor-to-cucomplex-cuda-gpu-array-sigsegv-error/18586/5

__global__
void moveto_cu(float* real, float* imag, cuComplex* dst, int len)
{
	const int i = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
	
	if(i >= len)
		return;
	dst[i].x = real[i];
	dst[i].y = imag[i];
}
__global__
void moveto_10(float* real, float* imag, cuComplex* dst, int len)
{
	const int i = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;

	if(i >= len)
		return;
    real[i] = dst[i].x;
    imag[i] = dst[i].y;
}


void Tensor_to_cuComplex(torch::Tensor real, torch::Tensor imag, cuComplex* dst, int len)
{
	float* realPtr;
	float* imagPtr;
	realPtr = real.data_ptr<float>();
	imagPtr = imag.data_ptr<float>();

	// max threads differ frome device
	const int threads = 1024;
	const int blocks = 65535;
	int by = len / threads + 1;
	const int bx = by / blocks + 1;
	if (by > blocks) by = blocks;

	const dim3 grid(bx,by);
    auto stream = at::cuda::getCurrentCUDAStream();

	moveto_cu<<<grid,threads,0,stream>>>(realPtr, imagPtr, dst,len);
}
void cuComplex_to_Tensor(torch::Tensor real, torch::Tensor imag, cuComplex* dst, int len)
{
	float* realPtr;
	float* imagPtr;
	realPtr = real.data_ptr<float>();
	imagPtr = imag.data_ptr<float>();

	// max threads differ frome device
	const int threads = 1024;
	const int blocks = 65535;
	int by = len / threads + 1;
	const int bx = by / blocks + 1;
	if (by > blocks) by = blocks;

	const dim3 grid(bx,by);
    auto stream = at::cuda::getCurrentCUDAStream();

	moveto_10<<<grid,threads,0,stream>>>(realPtr, imagPtr, dst,len);
}