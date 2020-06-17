import torch
import tocs_cpu as cpu
import tocs_cuSol as cuSol

def check_tensor(tensor,dim):
    if dim == tensor.dim():
        return True
    return False

def torch_complex_solve(tensor_A_r,tensor_A_i,tensor_b_r,tensor_b_i,use_cuSol=True,pivot_on=True):
    assert check_tensor(tensor_A_r,2), "tesor_A size is not correct"
    assert check_tensor(tensor_A_i,2), "tesor_A size is not correct"
    assert check_tensor(tensor_b_r,2), "tesor_b size is not correct"
    assert check_tensor(tensor_b_i,2), "tesor_b size is not correct"

    if use_cuSol:
        return cuSol.single_Csolve(
            tensor_A_r,
            tensor_A_i,
            tensor_b_r,
            tensor_b_i,
            pivot_on=pivot_on)
    else:
        return cpu.batch_complex_solver(
            tensor_A_r[None],
            tensor_A_i[None],
            tensor_b_r[None],
            tensor_b_i[None])

def torch_real_solve(tensor_A,tensor_b,use_cuSol=True,pivot_on=True):
    assert check_tensor(tensor_A,2), "tesor_A size is not correct"
    assert check_tensor(tensor_b,2), "tesor_b size is not correct"
    if use_cuSol:
        return cuSol.single_Rsolve(
            tensor_A,
            tensor_b,
            pivot_on=pivot_on)
    else:
        return torch.solve(tensor_b,tensor_A)

    
