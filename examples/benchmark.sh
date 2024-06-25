# benchmark for n_frames, n_cores
n_frames=(128 256 512 1024 2048)
n_cores=(1 2 4 6 12)

for n_frame in ${n_frames[@]}; do
    for n_core in ${n_cores[@]}; do
        echo "n_frame: $n_frame, n_core: $n_core"
        python bench_shared_mem.py --n_frames $n_frame --n_cores $n_core
    done
    python bench_serial.py --n_frames $n_frame
done