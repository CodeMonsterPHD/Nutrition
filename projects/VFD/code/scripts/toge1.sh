#!/bin/bash
for threshold in 0.01 0.05 0.08 0.12 0.15 0.2;do
{
  python main.py --CUDA_DEVICE 1 --batch_size 64 --output_dim 4096 --found_lr 1e-4 --segment 10 --epoch 30 --alpha 1 --beta 1 --threshold $threshold --seed 1 &
  #python main.py --CUDA_DEVICE 2 --batch_size 64 --output_dim 4096 --found_lr 1e-4 --segment 10 --epoch 30 --alpha 1 --beta 1 --threshold 0.2 --seed 0 &
wait
}
done
