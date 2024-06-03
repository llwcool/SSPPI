from generate_matrix import getdict
import os
from net import start_train
import time
import datetime
import sys

distance = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
length = [5, 8, 12, 15, 20, 25, 32, 38, 46, 53, 60, 69]
dict_length_distance = dict(zip(distance, length))

dict_seq, dict_pdb = getdict()

with open(os.path.join('data', "train.txt"), "r", encoding="utf-8") as f:
    info_train = f.readlines()

with open(os.path.join('data', "eval.txt"), "r", encoding="utf-8") as f:
    info_eval = f.readlines()
allL = [19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
gpu = 0
batch_size = 1024
epo = 100
for dis in distance[::-1]:
    L = 1
    log_file = open(f"log_{L}.txt", "a+")
    start_train_time = datetime.datetime.now()
    print(f"Start train time: {start_train_time}")
    log_file.write(f"Start train time: {start_train_time}\n")
    start = time.time()
    current_time = datetime.datetime.now()
    print(f"Start time: {current_time}, Begin training model with the window_length {L} and the struct_distance {dis}")
    log_file.write(f"Start time: {current_time}, Begin training model with the window_length {L} and the struct_distance {dis}\n")
    start_train(dict_length_distance, dict_seq, dict_pdb, L, gpu, batch_size,\
            dis, info_train, info_eval, epo)
    end = time.time()
    current_time = datetime.datetime.now()
    print(f"End time: {current_time}, Stop training model with the window_length {L} and the struct_distance {dis}")
    print(f"Training time: {end - start}, when the window_length is {L} and the struct_distance is {dis}")
    log_file.write(f"End time: {current_time}, Stop training model with the window_length {L} and the struct_distance {dis}\n")
    log_file.write(f"Training time: {end - start}, when the window_length is {L} and the struct_distance is {dis}\n")
    end_train_time = datetime.datetime.now()
    log_file.write(f"End train time: {end_train_time}\n")
    log_file.write(f"Training time: {end_train_time - start_train_time}\n")
    print(f"End train time: {end_train_time}")
    print(f"Training time: {end_train_time - start_train_time}")
    log_file.flush()
    log_file.close()
