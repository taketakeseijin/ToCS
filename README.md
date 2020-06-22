# ToCS
ToCS means **To**rch **C**omplex **S**olver.  
This repository gives you the **complex** version of **"torch.solve"**.  
With ToCS, you can solve **Ax = b** such that **A,b,x are complex**.  
TCS works on **both CPUs and GPUs**.

## Build
When you want to use **ToCS on GPUs**, you need to build the c++ & cuda extension code.  
Run the below line once. The line enables ToCS to use cuSolver.

```sh
python setup_cuSol.py install
```

## How to use
Usage as a _torch.nn.functional_. **Replace of "torch.solve"**

```python
import tocs
~
# x = torch.solve(b,A)
x_r,x_i = tocs.Csolve(b_r,b_i,A_r,A_i)
```
\_r, \_i mean real, imaginary part of the tensor respectively.
## What's inside?
On CPUs, [scipy.linalg.solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html#scipy.linalg.solve) works.  
On GPUs, [cuSolver's Dense linear solver](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-linearsolver-reference) works.   
On both CPUs and GPUs, **Batch is parallely computed**.

## Build tested environment
- ubuntu 18  
- python 3.8  
- pytorch 1.4  
- cuda tool kit 10  
- GeForce GTX 1080 Ti  
