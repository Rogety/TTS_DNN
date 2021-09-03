import os
import sys
import json
import numpy as np 
import torch as th
import torch.utils.data as data
import torch.optim as optim
import argparse as ap

from os import path

import tts

parser = ap.ArgumentParser()
parser.add_argument("--source", action="store_true")
parser.add_argument("--target", action="store_true")
args = parser.parse_args()


with open(path.join("configs","Configs.json")) as configs_file:
    configs = json.load(configs_file)

with open(path.join("configs","IOConfigs.json")) as io_configs_file:
    io_configs = json.load(io_configs_file)

a_order = io_configs["a_order"]
b_order = io_configs["b_order"]
c_order = io_configs["c_order"]
d_order = io_configs["d_order"]

mgc_order = configs["mgc_order"]
lf0_order = configs["lf0_order"]
hidden_size = configs["hidden_size"]
use_cuda = configs["use_cuda"]
sampling_rate = configs["sampling_rate"]
frame_shift = configs["frame_shift"]

model_dir_path = path.join("model")


if args.source:
    a_dir_path = path.join("data","gen","set","a")
    stat_dir_path = path.join("data", "source", "stat")
    gen_dir_path = path.join("gen","source", "from_a")
    var_path = path.join(stat_dir_path, "c_trn_var.pt")
    state_dict_path = path.join(model_dir_path, "source", "am.state_dict")
    device = th.device("cuda" if use_cuda else "cpu")

    variance = th.load(var_path)
    model = tts.AcousticModelV3( a_order, hidden_size, c_order, variance=variance).to(device)
    state_dict = th.load( state_dict_path )
    model.load_state_dict(state_dict)

    with th.no_grad():
        # 去掉 voiced/unvoiced
        var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
        for filename in os.listdir(a_dir_path):
            if filename.endswith(".bin"):
                base = path.splitext(filename)[0]
                print(base)
                a_path = path.join(a_dir_path, "{}.bin".format(base) )
                a = tts.load_bin( a_path, 'float32').view(-1,a_order).to(device)
                 # c_: lf0 mgc dlf0 dmgc ddlf0 ddmgc uv
                c_ = model(a)
                lf0 = c_[:,0]
                uv = ( c_[:, c_order-1] > 0.4 ).cpu()

                c =  c_[:, 0:c_order-1].view(-1, 3, lf0_order+mgc_order)
                
                lf0_m = c[:,:,0:lf0_order].view(-1, 3*lf0_order).cpu()
                mgc_m = c[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).cpu()
                lf0_v = var_lf0_mgc[:,:,0:lf0_order].view(-1, 3*lf0_order).expand_as(lf0_m)
                mgc_v = var_lf0_mgc[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).expand_as( mgc_m )

                lf0_mlpg = th.cat( (lf0_m,lf0_v), dim=1 )
                mgc_mlpg = th.cat( (mgc_m,mgc_v), dim=1 )
                
                tts.savebin(lf0_mlpg, os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)))
                tts.savebin(mgc_mlpg, os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)))
                tts.savebin(uv.float(), os.path.join(gen_dir_path, '{}.uv'.format(base)))

if args.target:
    a_dir_path = path.join("data","gen","set","a")
    source_stat_dir_path = path.join("data", "source", "stat")
    stat_dir_path = path.join("data", "target", "stat")
    gen_dir_path = path.join("gen","target", "from_a")
    source_var_path = path.join(source_stat_dir_path, "c_trn_var.pt")
    var_path = path.join(stat_dir_path, "c_trn_var.pt")
    source_state_dict_path = path.join(model_dir_path, "source", "am.state_dict")
    state_dict_path = path.join(model_dir_path, "target", "am.state_dict")
    device = th.device("cuda" if use_cuda else "cpu")

    #source_variance = th.load(source_var_path)
    #source_model = tts.AcousticModelV3( a_order, hidden_size, c_order, variance=source_variance).to(device)
    #source_state_dict = th.load(source_state_dict_path)
    #source_model.load_state_dict(source_state_dict)
    variance = th.load(var_path)
    #model = tts.AcousticModelAdapt( source_model , variance=variance).to(device)
    model = tts.AcousticModelV3( a_order, hidden_size, c_order, variance=variance).to(device)
    state_dict = th.load(state_dict_path)
    model.load_state_dict(state_dict)

    with th.no_grad():
        # 去掉 voiced/unvoiced
        var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
        for filename in os.listdir(a_dir_path):
            if filename.endswith(".bin"):
                base = path.splitext(filename)[0]
                print(base)
                a_path = path.join(a_dir_path, "{}.bin".format(base) )
                a = tts.load_bin( a_path, 'float32').view(-1,a_order).to(device)
                 # c_: lf0 mgc dlf0 dmgc ddlf0 ddmgc uv
                c_ = model(a)
                lf0 = c_[:,0]
                uv = ( c_[:, c_order-1] > 0.4 ).cpu()

                c =  c_[:, 0:c_order-1].view(-1, 3, lf0_order+mgc_order)
                
                lf0_m = c[:,:,0:lf0_order].view(-1, 3*lf0_order).cpu()
                mgc_m = c[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).cpu()
                lf0_v = var_lf0_mgc[:,:,0:lf0_order].view(-1, 3*lf0_order).expand_as(lf0_m)
                mgc_v = var_lf0_mgc[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).expand_as( mgc_m )

                lf0_mlpg = th.cat( (lf0_m,lf0_v), dim=1 )
                mgc_mlpg = th.cat( (mgc_m,mgc_v), dim=1 )
                
                tts.savebin(lf0_mlpg, os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)))
                tts.savebin(mgc_mlpg, os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)))
                tts.savebin(uv.float(), os.path.join(gen_dir_path, '{}.uv'.format(base)))
