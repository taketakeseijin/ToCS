# TCS
TCS means Torch Complex Solver.  
This repository gives you the complex version of "torch.solve".  
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
```

Use as a torch.nn.functional. Just a replace of torch.solve

```python
import TCSF
~
x_r,x_i = TCSF.CsolverFunction.apply(A_r,A_i,b_r,b_i)
```
## What's inside?
For cpu tensor, scipy.linalg.solve works. Batch is parallely computed.  
For gpu tensor, cuSolver works. Batch is parallely computerd.
