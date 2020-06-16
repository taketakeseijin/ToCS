import TCS_cuSol #build from cpp with cuSolver
import torch

def single_Rsolve(tensor_A,tensor_b,pivot_on=True):
    # tensor_A  [m1,m2]  -> input_A     [m2,m1]
    # tensor_b  [m,k]    -> input_b     [k,m]
    # tensor_x  [k,m]    -> output_x    [m,k]
    # tensor_lu [m2,m1]  -> output_lu   [m1,m2]
    # m = m1 = m2

    # Caution! Create new continuous tensor, because TCS_cuSol.Rsolve is destructive method and suppose continuous tensor.
    input_A = tensor_A.T.clone()
    input_b = tensor_b.T.clone()
    # .T.clone creates new transposed continuous tensor

    if pivot_on:
        tensor_x,tensor_lu = TCS_cuSol.Rsolve(input_A,input_b,1)
        return tensor_x.T,tensor_lu.T
    else:
        tensor_x,tensor_lu = TCS_cuSol.Rsolve(input_A,input_b,0)
        return tensor_x.T,tensor_lu.T

def single_Csolve(tensor_A_r,tensor_A_i,tensor_b_r,tensor_b_i,pivot_on=True):
    # tensor_A  [m1,m2]  -> input_A     [m2,m1]
    # tensor_b  [m,k]    -> input_b     [k,m]
    # tensor_x  [k,m]    -> output_x    [m,k]
    # tensor_lu [m2,m1]  -> output_lu   [m1,m2]
    # m = m1 = m2

    # Caution! Input continuous tensor.
    input_A_r = tensor_A_r.T.clone()
    input_A_i = tensor_A_i.T.clone()
    input_b_r = tensor_b_r.T.clone()
    input_b_i = tensor_b_i.T.clone()
    # .T.clone creates new transposed continuous tensor

    if pivot_on:
        tensor_x_r,tensor_x_i,tensor_lu_r,tensor_lu_i = TCS_cuSol.Csolve(input_A_r,input_A_i,input_b_r,input_b_i,1)
        return tensor_x_r.T,tensor_x_i.T,tensor_lu_r.T,tensor_lu_i.T
    else:
        tensor_x_r,tensor_x_i,tensor_lu_r,tensor_lu_i = TCS_cuSol.Csolve(input_A_r,input_A_i,input_b_r,input_b_i,0)
        return tensor_x_r.T,tensor_x_i.T,tensor_lu_r.T,tensor_lu_i.T