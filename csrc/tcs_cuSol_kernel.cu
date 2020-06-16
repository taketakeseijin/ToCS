#include <torch/types.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ref. https://discuss.pytorch.org/t/aten-transfer-tensor-to-cucomplex-cuda-gpu-array-sigsegv-error/18586/5

__global__
void moveto_cu(float* real, float* imag, cuComplex* dst, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= len)
		return;
	dst[i].x = real[i];
	dst[i].y = imag[i];
}
__global__
void moveto_10(float* real, float* imag, cuComplex* dst, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
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

	moveto_cu<<<1,len>>>(realPtr, imagPtr, dst,len);
}
void cuComplex_to_Tensor(torch::Tensor real, torch::Tensor imag, cuComplex* dst, int len)
{
	float* realPtr;
	float* imagPtr;
	realPtr = real.data_ptr<float>();
	imagPtr = imag.data_ptr<float>();

	moveto_10<<<1,len>>>(realPtr, imagPtr, dst,len);
}