#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset=nih --co_lambda=0.9 --adjust_lr=0 --nih_img_size=128 --forget_rate=0.1 > ../Results/joint_nih_multiLabel_resnet34_img128_forget0.1_withdecay.txt &