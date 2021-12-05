# Create file integral.py
import math
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

fun = lambda x: 3*x**2 - 2.5*x + 1
a = -3
b = 1
iterations = 10000000

delta = (b - a) / iterations
chunk_size = (iterations)//size + 1
chunk_head = chunk_size * rank
chunk_tail = chunk_size * (rank + 1)
if chunk_tail > iterations:
    chunk_tail = iterations

fun_approx = 0
for i in range(chunk_head,chunk_tail):
    fun_approx += fun(a+delta*i)

fun_approx = comm.reduce(fun_approx, op = MPI.SUM, root = 0)

if rank == 0:
    fun_approx += fun(a) / 2. + fun(b) / 2.
    print(f"{fun_approx*delta}")
