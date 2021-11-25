# Create file spectrogram.py
import math
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

steps_in_t = 3**9+6
t = np.linspace(-20*2*math.pi, 20*2*math.pi, steps_in_t)
y=np.sin(t)*np.exp(-t**2/2/20**2)
y=y+np.sin(3*t)*np.exp(-(t-5*2*math.pi)**2/2/20**2)
y=y+np.sin(5.5*t)*np.exp(-(t-10*2*math.pi)**2/2/5**2)
y=y+np.sin(4*t)*np.exp(-(t-7*2*math.pi)**2/2/5**2)
    
nwindowsteps = 10000
window_positions= np.linspace(-20*2*math.pi, 20*2*math.pi, nwindowsteps)
window_width = 2.0*2*math.pi

chunk_size = len(window_positions)//size + 1
chunk_head = chunk_size * rank
chunk_tail = chunk_size * (rank + 1)
if chunk_tail > len(window_positions):
    chunk_tail = len(window_positions)

specgram = np.empty([len(t), chunk_tail-chunk_head])
for i, w in enumerate(window_positions[chunk_head:chunk_tail]):
    window_funtion = np.exp(-(t-w)**2/2/window_width**2)
    y_window = y * window_funtion
    specgram[:,i] = abs(np.fft.fft(y_window))

print(f"process {rank}: from {chunk_head} to {chunk_tail}")
  
all_specgram_patched = comm.gather((chunk_head, specgram),root=0)

if rank == 0:
    print(f"all shape: {specgram.shape}")
    for specgram_patch in sorted(all_specgram_patched)[1:]:
        specgram = np.concatenate((specgram, specgram_patch[1]), axis=1)
    
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    w=np.fft.fftfreq(len(y), d=(t[1]-t[0])/2/math.pi)
    im = ax.imshow(specgram, aspect='auto', origin='lower', extent=[min(t)/2/math.pi, max(t)/2/math.pi, y[0], 2 * w[int(len(w)/2)-1]])
    fig.set_size_inches(9, 7, forward=True)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Time, cycles')
    ax.set_ylabel('Frequency')
    ax.title.set_text('Spectrogram')
#     plt.show()
    plt.savefig('spectrogram.png')
