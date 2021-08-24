from os import path
import os
import torch as th
import numpy as np
import json



if __name__ == '__main__':

    io_config_path = os.path.join("configs","IOConfigs.json")
    with open(io_config_path,'r') as fin:
        io_config = json.load(fin)

    a_order = io_config["a_order"]
    b_order = io_config["b_order"]
    c_order = io_config["c_order"]
    d_order = io_config["d_order"]

    set_dir_path = path.join("data","set")
    stat_dir_path = path.join("data","stat")

    a_trn_path = path.join(set_dir_path,"a_trn.bin")
    a_val_path = path.join(set_dir_path,"a_val.bin")
    a_tst_path = path.join(set_dir_path,"a_tst.bin")

    b_trn_path = path.join(set_dir_path,"b_trn.bin")
    b_val_path = path.join(set_dir_path,"b_val.bin")
    b_tst_path = path.join(set_dir_path,"b_tst.bin")

    c_trn_path = path.join(set_dir_path,"c_trn.bin")
    c_val_path = path.join(set_dir_path,"c_val.bin")
    c_tst_path = path.join(set_dir_path,"c_tst.bin")

    d_trn_path = path.join(set_dir_path,"d_trn.bin")
    d_val_path = path.join(set_dir_path,"d_val.bin")
    d_tst_path = path.join(set_dir_path,"d_tst.bin")

    a_trn_len = int(os.stat( a_trn_path ).st_size / 4 / a_order)
    a_val_len = int(os.stat( a_val_path ).st_size / 4 / a_order)
    a_tst_len = int(os.stat( a_tst_path ).st_size / 4 / a_order)

    b_trn_len = int(os.stat( b_trn_path ).st_size / 4 / b_order)
    b_val_len = int(os.stat( b_val_path ).st_size / 4 / b_order)
    b_tst_len = int(os.stat( b_tst_path ).st_size / 4 / b_order)

    c_trn_len = int(os.stat( c_trn_path ).st_size / 4 / c_order) ## /4 float32
    c_val_len = int(os.stat( c_val_path ).st_size / 4 / c_order)
    c_tst_len = int(os.stat( c_tst_path ).st_size / 4 / c_order)

    d_trn_len = int(os.stat( d_trn_path ).st_size / 4 / d_order)
    d_val_len = int(os.stat( d_val_path ).st_size / 4 / d_order)
    d_tst_len = int(os.stat( d_tst_path ).st_size / 4 / d_order)

    print("a_trn_len : ",a_trn_len)
    print("a_val_len : ",a_val_len)
    print("a_tst_len : ",a_tst_len)
    print("b_trn_len : ",b_trn_len)
    print("b_val_len : ",b_val_len)
    print("b_tst_len : ",b_tst_len)
    print("c_trn_len : ",c_trn_len)
    print("c_val_len : ",c_val_len)
    print("c_tst_len : ",c_tst_len)
    print("d_trn_len : ",d_trn_len)
    print("d_val_len : ",d_val_len)
    print("d_tst_len : ",d_tst_len)

    c_trn = th.from_numpy( np.memmap( c_trn_path , dtype="float32", mode="r", shape=(c_trn_len, c_order) ) )
    print("c_trn :",c_trn.shape)
    c_trn_mean = th.mean(c_trn, 0)
    c_trn_var = th.var(c_trn, 0)
    c_trn_std = th.std(c_trn, 0)
    print(c_trn_mean.shape)
    print(c_trn_var.shape)
    print(c_trn_std.shape)

    th.save(c_trn_mean, path.join(stat_dir_path, "c_trn_mean.pt"))
    th.save(c_trn_var, path.join(stat_dir_path, "c_trn_var.pt"))
    th.save(c_trn_std, path.join(stat_dir_path, "c_trn_std.pt"))
