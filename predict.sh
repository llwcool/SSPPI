#!/bin/bash
nohup python -u net_evalue.py 3 1 100 > net_3.log 2>&1 &
#3 是窗口大小
#1 使用GPU则输入卡号，否则输入CPU表示使用CPU
#100 batch_size
#需要预测的pdbid放在data/eval.txt中
#模型参数在work目录下
