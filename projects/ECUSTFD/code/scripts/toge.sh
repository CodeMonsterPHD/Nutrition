#!/bin/bash
for seed in 0 1 2 3;do
{
  python main.py --CUDA_DEVICE 0 --batch_size 64 --output_dim 4096 --found_lr 1e-4 --segment 10 --epoch 30 --alpha 1 --beta 1 --threshold 0.1 --seed $seed &
  #python main.py --CUDA_DEVICE 2 --batch_size 64 --output_dim 4096 --found_lr 1e-4 --segment 10 --epoch 30 --alpha 1 --beta 1 --threshold 0.2 --seed 0 &
wait
}
done

nohup sh *.sh > myout.file 2>&1 &
