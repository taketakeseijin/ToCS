# TCS
TCS means Torch Complex Solver.  
This repository gives you the complex version of "torch.solve".  
With TCS, you can solve Ax = b such that A,b,x are complex.  
TCS works in both cpu and gpu.

## Build
For use in gpu, you need to build.  
But just run below.

```sh
python setup_cuSol.py install
```

## How to use
Use as a torch.nn.Module

```python
import TCSF
tcsf_module = TCSF.Csolver()
~
x_r,x_i = tcsf_module(A_r,A_i,b_r,b_i)
```

Use as a torch.nn.functional. Just a replace of torch.solve

```python
import TCSF
~
x_r,x_i = TCSF.CsolverFunction.apply(A_r,A_i,b_r,b_i)
```
_r, _i means real, imaginary part respectively.
## What's inside?
For cpu tensor, scipy.linalg.solve works. Batch is parallely computed.  
For gpu tensor, cuSolver works. Batch is parallely computerd.

## build tested environment
ubuntu 18  
python 3.8  
pytorch 1.4  
cuda 10  
