import time
import MDAnalysis as mda
import numpy as np
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
def main(n_frames: int):
    if n_frames is None:
        raise ValueError('Please provide the number of frames to calculate pairwise RMSD')
    n_frames = int(n_frames)

    u = mda.Universe(TOP, TRAJ)
    u.transfer_to_memory(stop=n_frames, verbose=True)

    pairwise_rmsd = np.zeros((u.trajectory.n_frames, u.trajectory.n_frames))

    protein = u.select_atoms('protein and name CA')

    start_time = time.time()
    for i in range(u.trajectory.n_frames):
        for j in range(i, u.trajectory.n_frames):
            pairwise_rmsd[i][j] = calculate_rmsd(protein, i, j)
    end_time = time.time()  # End timing

    np.save('pairwise_rmsd.npy', pairwise_rmsd)
    print('Pairwise RMSD calculation is done!')

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

    # save to log
    with open('log_serial.txt', 'a') as f:
        f.write(f"Elapsed time: {elapsed_time} seconds with 1 processors with {n_frames} frames\n")

if __name__ ==  '__main__':
    main()