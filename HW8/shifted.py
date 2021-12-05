# Create file shifted.py
import cv2
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpi4py import MPI
import os
from PIL import Image
import random

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

img = cv2.imread('minecraft.jpg')
height = img.shape[0]

for roll in range(height):
    if rank == 0 and roll == 0:
        img = cv2.imread('minecraft.jpg')
        image = img.copy()
        split = np.array_split(image, size, axis=0)
    if rank != 0:
        split = None

    split_part = comm.scatter(split, root=0)
    split_part = np.roll(split_part, 1, axis=0)
    
    if size != 1:
        it_to_send = rank+1 if rank != size-1 else 0
        it_to_recv = rank-1 if rank != 0 else size-1
        comm.send(split_part[0], dest=it_to_send)
        split_part_first = comm.recv(source=it_to_recv)
        split_part[0] = split_part_first
    
    new_split = comm.gather(split_part,root=0)
    
    if rank == 0:
        print("[%-50s] %d%%" % ('='* int(roll * 50 / (height-1)) , int(roll * 100 / (height-1))), end='\r')
        cv2.imwrite('sminecraft' + '{0:05}'.format(roll) + '.jpg', np.concatenate(new_split))
        split = new_split.copy()

if rank == 0:
    fp_in = "sminecraft*.jpg"
    fp_out = "shifted_minecraft.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=30, loop=0)
    
    for filePath in sorted(glob.glob(fp_in)):
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    print(f"shifted pictures gif could be found in file {fp_out}")
