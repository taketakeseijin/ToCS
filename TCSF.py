# Torch Complex Solver Functional
import torch
import tcs_cuSol as cuSol
from torch.autograd import Function

class Csolver(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        tensor_A_r,
        tensor_A_i,
        tensor_b_r,
        tensor_b_i,):

        return CsolverFunction.apply(
            tensor_A_r,
            tensor_A_i,
            tensor_b_r,
            tensor_b_i,)

class CsolverFunction(Function):

    @staticmethod
    def forward(
        ctx,
        tensor_A_r,
        tensor_A_i,
        tensor_b_r,
        tensor_b_i,):
        tensor_x_r, tensor_x_i, _, _ = cuSol.single_Csolve(
            tensor_A_r,
            tensor_A_i,
            tensor_b_r,
            tensor_b_i,
        )
        ctx.save_for_backward(
            tensor_A_r,
            tensor_A_i,
            tensor_x_r,
            tensor_x_i
        )
        return tensor_x_r, tensor_x_i

    @staticmethod
    def backward(
        ctx,
        grad_x_r,
        grad_x_i):
        tensor_A_r, tensor_A_i, tensor_x_r, tensor_x_i = ctx.saved_tensors
        grad_A_r = grad_A_i = grad_b_r = grad_b_i = None
        if any(ctx.needs_input_grad):
            grad_b_r, grad_b_i, _, _ = cuSol.single_Csolve(
                tensor_A_r.T,
                tensor_A_i.T,
                grad_x_r,
                grad_x_i,
            )
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_A_r = - (torch.einsum("nk,mk->nm", grad_b_r, tensor_x_r) - torch.einsum("nk,mk->nm", grad_b_i, tensor_x_i))
            grad_A_i = - (torch.einsum("nk,mk->nm", grad_b_r, tensor_x_i) + torch.einsum("nk,mk->nm", grad_b_i, tensor_x_r))
        return grad_A_r, grad_A_i, grad_b_r, grad_b_i
