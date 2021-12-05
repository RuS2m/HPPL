# Create file life.py
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np

def initialize(pixels):
    indices = [[1, 1],
            [2, 2], [2, 25],
            [3, 13], [3, 14], [3, 21], [3, 22], [3, 35], [3, 36],
            [4, 12], [4, 16], [4, 21], [4, 22], [4, 35], [4, 36],
            [5, 1], [5, 2], [5, 11], [5, 17], [5, 21], [5, 22],
            [6, 1], [6, 2], [6, 11], [6, 17], [6, 18], [6, 23],
            [7, 11], [7, 17], [7, 25],
            [8, 12], [8, 16],
            [9, 13], [9, 14]]
    
    for i in indices:
        pixels[i[0]][i[1]] = 1

    return pixels

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

pixels = np.zeros([45, 45])
initialize(pixels)

chunk_size = 45//size + 1
chunk_head = chunk_size * rank
chunk_tail = chunk_size * (rank + 1)
if chunk_tail > 45:
    chunk_tail = 45

pixels_cur = pixels[:,chunk_head:chunk_tail]
step = 0
height = pixels_cur.shape[0]
width = pixels_cur.shape[1]

if size != 1:
        it_to_sendr = rank+1 if rank != size-1 else 0
        it_to_sendl = rank-1 if rank != 0 else size-1
        comm.send(pixels_cur[:,0], dest=it_to_sendl)
        comm.send(pixels_cur[:,width-1], dest=it_to_sendr)
        left_ghost = comm.recv(source=it_to_sendl)
        right_ghost = comm.recv(source=it_to_sendr)
        pixels_cur = np.concatenate((np.array([left_ghost]).T, pixels_cur), axis=1)
        pixels_cur = np.concatenate((pixels_cur, np.array([right_ghost]).T), axis=1)

height = pixels_cur.shape[0]
width = pixels_cur.shape[1]
        
step = 0
while sum(sum(pixels_cur)) != 0 and step != 100:
    new_pixels = pixels_cur.copy()
    for i in range(height):
        for j in range(width):
            total = int((pixels[i, (j-1) % width] + pixels[i, (j+1) % width] +
                             pixels[(i-1) % height, j] + pixels[(i+1) % height, j] +
                             pixels[(i-1) % height, (j-1) % width] + pixels[(i-1) % height, (j+1) % width] +
                             pixels[(i+1) % height, (j-1) % width] + pixels[(i+1) % height, (j+1) % width]))
 
            if pixels[i, j] == 1:
                if (total < 2) or (total > 3):
                    new_pixels[i, j] = 0
            else:
                if total == 3:
                    new_pixels[i, j] = 1
    pixels_cur = new_pixels
    step += 1
    if rank == 0:
        print(f"step {step}", end='\r')
