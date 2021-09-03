import numpy as np 
import torch as th
import json
import os
from os import path
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("--source", action="store_true")
parser.add_argument("--target", action="store_true")
args = parser.parse_args()

with open(path.join("configs","IOConfigs.json")) as fn:
    io_configs = json.load(fn)

a_order = io_configs["a_order"]
b_order = io_configs["b_order"]
c_order = io_configs["c_order"]
d_order = io_configs["d_order"]


if args.source:
    set_dir_path = path.join("data","source","set")
    stat_dir_path = path.join("data","source","stat")

    a_trn_path = path.join(set_dir_path,"a_trn.bin")
    # a_val_path = path.join(set_dir_path,"a_val.bin")
    # a_tst_path = path.join(set_dir_path,"a_tst.bin")

    # b_trn_path = path.join(set_dir_path,"b_trn.bin")
    # b_val_path = path.join(set_dir_path,"b_val.bin")
    # b_tst_path = path.join(set_dir_path,"b_tst.bin")

    c_trn_path = path.join(set_dir_path,"c_trn.bin")
    # c_val_path = path.join(set_dir_path,"c_val.bin")
    # c_tst_path = path.join(set_dir_path,"c_tst.bin")

    # d_trn_path = path.join(set_dir_path,"d_trn.bin")
    # d_val_path = path.join(set_dir_path,"d_val.bin")
    # d_tst_path = path.join(set_dir_path,"d_tst.bin")

    a_trn_len = int(os.stat( a_trn_path ).st_size / 4 / a_order)
    # a_val_len = int(os.stat( a_val_path ).st_size / 4 / a_order)
    # a_tst_len = int(os.stat( a_tst_path ).st_size / 4 / a_order)

    # b_trn_len = int(os.stat( b_trn_path ).st_size / 4 / b_order)
    # b_val_len = int(os.stat( b_val_path ).st_size / 4 / b_order)
    # b_tst_len = int(os.stat( b_tst_path ).st_size / 4 / b_order)

    c_trn_len = int(os.stat( c_trn_path ).st_size / 4 / c_order)
    # c_val_len = int(os.stat( c_val_path ).st_size / 4 / c_order)
    # c_tst_len = int(os.stat( c_tst_path ).st_size / 4 / c_order)

    # d_trn_len = int(os.stat( d_trn_path ).st_size / 4 / d_order)
    # d_val_len = int(os.stat( d_val_path ).st_size / 4 / d_order)
    # d_tst_len = int(os.stat( d_tst_path ).st_size / 4 / d_order)
    a_trn_len = int(os.stat( a_trn_path ).st_size / 4 / a_order)
    c_trn_len = int(os.stat( c_trn_path ).st_size / 4 / c_order)
    print(a_trn_len, c_trn_len)
    c_trn = th.from_numpy( np.memmap( c_trn_path , dtype="float32", mode="r", shape=(c_trn_len, c_order) ) ) 
    c_trn_mean = th.mean(c_trn, 0)
    c_trn_var = th.var(c_trn, 0)
    c_trn_std = th.std(c_trn, 0)
    th.save(c_trn_mean, path.join(stat_dir_path, "c_trn_mean.pt"))
    th.save(c_trn_var, path.join(stat_dir_path, "c_trn_var.pt"))
    th.save(c_trn_std, path.join(stat_dir_path, "c_trn_std.pt"))



if args.target:
    set_dir_path = "data/target/set"
    stat_dir_path = path.join("data","target","stat")

    a_trn_path = path.join(set_dir_path,"a_trn.bin")
    c_trn_path = path.join(set_dir_path,"c_trn.bin")


    
    c_trn = th.from_numpy( np.memmap( c_trn_path , dtype="float32", mode="r", shape=(c_trn_len, c_order) ) ) 
    c_trn_mean = th.mean(c_trn, 0)
    c_trn_var = th.var(c_trn, 0)
    c_trn_std = th.std(c_trn, 0)
    
    th.save(c_trn_mean, path.join(stat_dir_path, "c_trn_mean.pt"))
    th.save(c_trn_var, path.join(stat_dir_path, "c_trn_var.pt"))
    th.save(c_trn_std, path.join(stat_dir_path, "c_trn_std.pt"))
