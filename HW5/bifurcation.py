# Create file bifurcation.py
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

xs_amount = 100000
r = np.linspace(0, 4.0, num=xs_amount)

chunk_size = xs_amount//size + 1
chunk_head = chunk_size * rank
chunk_tail = chunk_size * (rank + 1)
if chunk_tail > xs_amount:
    chunk_tail = xs_amount
    
x = np.random.rand((chunk_tail-chunk_head))
for i in range(1, 10000):
    inverted_x = 1 - x
    updated_x = np.multiply(x,r[chunk_head:chunk_tail])
    updated_x = np.multiply(updated_x, inverted_x)
    x = updated_x
print(f"process {rank}: from {chunk_head} to {chunk_tail}")
  
all_x_patched = comm.gather((chunk_head, x),root=0)

if rank == 0:
    all_x = []
    for x_patch in sorted(all_x_patched):
        all_x = np.append(all_x, x_patch[1])
    plt.figure(figsize=(12, 8))
    plt.title(r'Logistic map: $x_{n+1} = r x_{n} (1-x_{n}).$ ')
    plt.ylabel('x')
    plt.xlabel('r')
    plt.scatter(r, all_x, s=0.1)
    #plt.show()
    plt.savefig('bifurcation.png')
