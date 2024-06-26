import time
import MDAnalysis as mda
import numpy as np
from sharedmemoryreader.sharedmemoryreader import SharedMemoryReader, transfer_to_shared_memory
from multiprocessing import Pool
import sys
import click
import warnings
warnings.filterwarnings("ignore")

TOP = 'yiip_equilibrium/YiiP_system.pdb'
TRAJ = 'yiip_equilibrium/YiiP_system_90ns_center.xtc'

from MDAnalysis.analysis.rms import rmsd
def calculate_rmsd(protein, i, j):
    protein.universe.trajectory[i]
    pos_1 = protein.positions
    protein.universe.trajectory[j]
    pos_2 = protein.positions
    return rmsd(pos_1, pos_2, superposition=True, center=True)

@click.command()
@click.option('--n_frames', default=None, help='Number of frames to calculate pairwise RMSD')
@click.option('--n_cores', default=12, help='Number of cores to use')
def main(n_frames: int, n_cores: int):
    if n_frames is None:
        raise ValueError('Please provide the number of frames to calculate pairwise RMSD')
    n_frames = int(n_frames)

    p = Pool(processes=n_cores)
    u = mda.Universe(TOP, TRAJ)
    transfer_to_shared_memory(u, stop=n_frames, verbose=True)

    pairwise_rmsd = np.zeros((u.trajectory.n_frames, u.trajectory.n_frames))

    protein = u.select_atoms('protein and name CA')

    start_time = time.time()
    output = p.starmap(calculate_rmsd, [(protein, i, j) for i in range(u.trajectory.n_frames) for j in range(i, u.trajectory.n_frames)])
    end_time = time.time()  # End timing

    for i in range(u.trajectory.n_frames):
        for j in range(i, u.trajectory.n_frames):
            pairwise_rmsd[i][j] = output.pop(0)
            pairwise_rmsd[j][i] = pairwise_rmsd[i][j]
    
    np.save(f'pairwise_rmsd_parallel_{n_frames}.npy', pairwise_rmsd)
    print('Pairwise RMSD calculation is done!')

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

    # save to log
    with open('log.txt', 'a') as f:
        f.write(f"Elapsed time: {elapsed_time} seconds with {n_cores} processors with {n_frames} frames\n")

if __name__ ==  '__main__':
    main()