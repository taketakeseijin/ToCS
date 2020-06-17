
from concurrent import futures

from scipy import linalg
import numpy as np

def complex_solver_cpu_kernel(np_complex_A,np_complex_b,np_complex_x,kernel_id):
    # Ax = b
    np_complex_x[kernel_id] = linalg.solve(np_complex_A[kernel_id],np_complex_b[kernel_id])

def tensor_to_complex_np(tensor_r,tensor_i):
    return tensor_r.detach().cpu().numpy() + 1j*tensor_i.detach().cpu().numpy()

def batch_complex_solver(tensor_A_r,tensor_A_i,tensor_b_r,tensor_b_i,max_workers=4):
    # A [...,n,n]
    # b [...,n,k]
    batch_np_complex_A = tensor_to_complex_np(tensor_A_r,tensor_A_i)
    batch_np_complex_b = tensor_to_complex_np(tensor_b_r,tensor_b_i)

    original_A_shape = batch_np_complex_A.shape
    original_b_shape = batch_np_complex_b.shape
    batch_np_complex_A = batch_np_complex_A.reshape(-1,original_A_shape[-2],original_A_shape[-1])
    batch_np_complex_b = batch_np_complex_b.reshape(-1,original_b_shape[-2],original_b_shape[-1])

    # create output area
    batch_np_complex_x = np.empty_like(batch_np_complex_b)

    batch_size = batch_np_complex_A.shape[0]

    # multi threadings
    future_list = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for kid in range(batch_size):
            future = executor.submit(
                fn=complex_solver_cpu_kernel,
                np_complex_A = batch_np_complex_A,
                np_complex_b = batch_np_complex_b,
                np_complex_x = batch_np_complex_x,
                kernel_id = kid)
            future_list.append(future)
        _ = futures.as_completed(fs=future_list)

    return batch_np_complex_x.reshape(original_b_shape)


