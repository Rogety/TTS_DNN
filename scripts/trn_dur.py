import os
import sys
import json
import numpy as np
import torch as th
import torch.utils.data as data
import torch.optim as optim
import argparse as ap
import torch.nn.functional as fn
from torch.utils.tensorboard import SummaryWriter
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
d_class = io_configs["d_class"]

max_epoch = configs["max_epoch"]
batch_size = configs["batch_size"]
use_cuda = configs["use_cuda"]
hidden_size = configs["hidden_size"]

model_dir_path = path.join("model")


def train(model, trn_dl, device, optimizer ):
    model.train()
    for i, (b, d) in enumerate(trn_dl):
        b, d = b.to(device), d.to(device=device, dtype=th.float)
        b.requires_grad_()
        optimizer.zero_grad()
        d_ = model.forward(b)
        loss = fn.mse_loss(d_, d)
        #loss = -th.mean( tts.log_gaussian_density(0.0, m_, model.variance, c) )
        loss.backward()
        optimizer.step()

def eval(model, trn_dl, val_dl, tst_dl, device):
    model.eval()
    trn_loss = 0.0
    val_loss = 0.0
    tst_loss = 0.0

    with th.no_grad():
        #i = 0
        for i, (b, d) in enumerate( trn_dl ):
            b, d = b.to(device), d.to(device=device, dtype=th.float)
            b.requires_grad_()
            d_ = model.forward(b)
            loss = fn.mse_loss(d_, d, reduction='sum').item()
            trn_loss += loss
        #i = 0
        for i,(b, d) in enumerate( val_dl ):
            b, d = b.to(device), d.to(device=device, dtype=th.float)
            b.requires_grad_()
            d_ = model.forward(b)
            loss = fn.mse_loss(d_, d, reduction='sum').item()
            val_loss += loss

        for i,(b, d) in enumerate( tst_dl ):
            b, d = b.to(device), d.to(device=device, dtype=th.float)
            b.requires_grad_()
            d_ = model.forward(b)
            loss = fn.mse_loss(d_, d, reduction='sum').item()
            tst_loss += loss

    trn_loss /= len(trn_dl.dataset)
    val_loss /= len(val_dl.dataset)
    tst_loss /= len(tst_dl.dataset)

    return trn_loss, val_loss, tst_loss

if args.source:
    # paths of set and variance
    PATH_to_log_dir = "run/dur/"
    writer = SummaryWriter(PATH_to_log_dir)
    set_dir_path = path.join("data", "source", "set")
    stat_dir_path = path.join("data", "source", "stat")
    b_trn_path = path.join(set_dir_path, "b_trn.bin")
    b_val_path = path.join(set_dir_path, "b_val.bin")
    b_tst_path = path.join(set_dir_path, "b_tst.bin")
    d_trn_path = path.join(set_dir_path, "d_trn.bin")
    d_val_path = path.join(set_dir_path, "d_val.bin")
    d_tst_path = path.join(set_dir_path, "d_tst.bin")
    state_dict_path = path.join(model_dir_path, "source", "dur.state_dict")

    # make the dataset
    trn_set = tts.make_dataset(b_trn_path, b_order, 'float32', d_trn_path, d_order, 'int32')
    val_set = tts.make_dataset(b_val_path, b_order, 'float32', d_val_path, d_order, 'int32')
    tst_set = tts.make_dataset(b_tst_path, b_order, 'float32', d_tst_path, d_order, 'int32')

    #
    trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")

    model = tts.DurationModelV2( b_order, hidden_size ).to(device)
    #state_dict = th.load(state_dict_path)
    #model.load_state_dict(state_dict)


    optimizer = optim.Adam( model.parameters() , lr=0.0001 )

    print("+==================================================================+")
    print("|  Start Training Duration Model of Source                         |")
    print("+=========+===============+===============+===============+========+")
    print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
    print("+=========+===============+===============+===============+========+")
    sys.stdout.flush()
    min_trn_loss, min_val_loss, min_tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device)
    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()

    for epoch in range(1, max_epoch+1):
        train(model, trn_dl_for_trainig, device, optimizer)
        trn_loss, val_loss, tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device)
        save = False
        if val_loss < min_val_loss:
            save = True
            min_val_loss = val_loss
            th.save(model.state_dict(), state_dict_path)

        if save == True:
            print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, "*"))
        else:
            print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, " "))

        print("+---------+---------------+---------------+---------------+--------+")
        sys.stdout.flush()


if args.target:
# paths of set and variance
    set_dir_path = path.join("data", "target", "set")
    stat_dir_path = path.join("data", "target", "stat")
    b_trn_path = path.join(set_dir_path, "b_trn.bin")
    b_val_path = path.join(set_dir_path, "b_val.bin")
    b_tst_path = path.join(set_dir_path, "b_tst.bin")
    d_trn_path = path.join(set_dir_path, "d_trn.bin")
    d_val_path = path.join(set_dir_path, "d_val.bin")
    d_tst_path = path.join(set_dir_path, "d_tst.bin")
    source_state_dict_path = path.join(model_dir_path, "source", "dur.state_dict")
    target_state_dict_path = path.join(model_dir_path, "target", "dur.state_dict")
    # make the dataset
    trn_set = tts.make_dataset(b_trn_path, b_order, 'float32', d_trn_path, d_order, 'int32')
    val_set = tts.make_dataset(b_val_path, b_order, 'float32', d_val_path, d_order, 'int32')
    tst_set = tts.make_dataset(b_tst_path, b_order, 'float32', d_tst_path, d_order, 'int32')

    #
    trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")

    model = tts.DurationModelV2( b_order, hidden_size ).to(device)
    state_dict = th.load(source_state_dict_path)
    model.load_state_dict(state_dict)


    param = []
    for name, value in model.named_parameters():
        #print(name)
        if name.endswith('bias'):
            param.append(value)

    optimizer = optim.Adam( param , lr=0.001 )

    print("+==================================================================+")
    print("|  Start Training Duration Model of Target                         |")
    print("+=========+===============+===============+===============+========+")
    print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
    print("+=========+===============+===============+===============+========+")
    sys.stdout.flush()
    min_trn_loss, min_val_loss, min_tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device)
    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()

    for epoch in range(1, max_epoch+1):
        train(model, trn_dl_for_trainig, device, optimizer)
        trn_loss, val_loss, tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device)
        save = False
        if val_loss < min_val_loss:
            save = True
            min_val_loss = val_loss
            th.save(model.state_dict(), target_state_dict_path)

        if save == True:
            print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, "*"))
        else:
            print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, " "))

        print("+---------+---------------+---------------+---------------+--------+")
        writer.add_scalar('Training/Loss', trn_loss, epoch)
        writer.add_scalar('Testing/Loss', tst_loss, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.flush()
        sys.stdout.flush()
