# Torch Complex Solver Functional
import torch
import tocs_cuSol as cuSol
import tocs_cpu as cpu
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

def Batch_Csolve(
            tensor_A_r,
            tensor_A_i,
            tensor_b_r,
            tensor_b_i,
            device):
    if device == "cpu":
        return cpu.batch_complex_solver(
            tensor_A_r,
            tensor_A_i,
            tensor_b_r,
            tensor_b_i,
        )
    else:
        return cuSol.Batch_Csolve(
            tensor_A_r,
            tensor_A_i,
            tensor_b_r,
            tensor_b_i,
        )

class CsolverFunction(Function):

    @staticmethod
    def forward(
        ctx,
        tensor_A_r,
        tensor_A_i,
        tensor_b_r,
        tensor_b_i,):
        assert tensor_A_r.is_cuda() == tensor_A_i.is_cuda() == tensor_b_r.is_cuda() == tensor_b_i.is_cuda(), "is_cuda not match" 
        ctx.device = "cuda" if tensor_A_r.is_cuda() else "cpu"       

        tensor_x_r, tensor_x_i = Batch_Csolve(
            tensor_A_r,
            tensor_A_i,
            tensor_b_r,
            tensor_b_i,
            ctx.device,
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
        batch_size = tensor_A_r.shape[0]
        if any(ctx.needs_input_grad):
            grad_b_r, grad_b_i = Batch_Csolve(
                tensor_A_r.transpose(-1,-2),
                tensor_A_i.transpose(-1,-2),
                grad_x_r.expand(batch_size,-1,-1),
                grad_x_i.expand(batch_size,-1,-1),
                ctx.device,
            )
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_A_r = - (torch.einsum("bnk,bmk->bnm", grad_b_r, tensor_x_r) - torch.einsum("bnk,bmk->bnm", grad_b_i, tensor_x_i))
            grad_A_i = - (torch.einsum("bnk,bmk->bnm", grad_b_r, tensor_x_i) + torch.einsum("bnk,bmk->bnm", grad_b_i, tensor_x_r))
        return grad_A_r, grad_A_i, grad_b_r, grad_b_i

Csolve = CsolverFunction.apply
