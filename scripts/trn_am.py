import os
import sys
import json
import numpy as np
import torch as th
import torch.utils.data as data
import torch.optim as optim
import argparse as ap
#from torch.utils.tensorboard import SummaryWriter
from os import path
from tensorboardX import SummaryWriter
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

max_epoch = configs["max_epoch"]
batch_size = configs["batch_size"]
use_cuda = configs["use_cuda"]
hidden_size = configs["hidden_size"]

model_dir_path = path.join("model")


def train(model, trn_dl, device, optimizer ):
    model.train()
    for i, (a, c) in enumerate(trn_dl):
        a, c = a.to(device), c.to(device)
        a.requires_grad_()
        optimizer.zero_grad()
        m_ = model.forward(a)
        loss = -th.mean( tts.log_gaussian_density(0.0, m_, model.variance, c) )
        loss.backward()
        optimizer.step()

def eval(model, trn_dl, val_dl, tst_dl, device):
    model.eval()
    trn_loss = 0.0
    val_loss = 0.0
    tst_loss = 0.0
    with th.no_grad():
        #i = 0
        for i, (a, c) in enumerate( trn_dl ):
            a, c = a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, model.variance, c) ).item()
            trn_loss += loss

        #i = 0
        for i,(a,c) in enumerate( val_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, model.variance, c) ).item()
            val_loss += loss

        for i,(a,c) in enumerate( tst_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, model.variance, c) ).item()
            tst_loss += loss

    trn_loss /= len(trn_dl.dataset)
    val_loss /= len(val_dl.dataset)
    tst_loss /= len(tst_dl.dataset)
    return trn_loss, val_loss, tst_loss

if args.source:

    # PATH_to_log_dir = "run/am/"
    # writer = SummaryWriter(PATH_to_log_dir)
    writer = SummaryWriter()
    # paths of set and variance
    set_dir_path = path.join("data", "source", "set")
    stat_dir_path = path.join("data", "source", "stat")
    a_trn_path = path.join(set_dir_path, "a_trn.bin")
    a_val_path = path.join(set_dir_path, "a_val.bin")
    a_tst_path = path.join(set_dir_path, "a_tst.bin")
    c_trn_path = path.join(set_dir_path, "c_trn.bin")
    c_val_path = path.join(set_dir_path, "c_val.bin")
    c_tst_path = path.join(set_dir_path, "c_tst.bin")
    var_path = path.join(stat_dir_path, "c_trn_var.pt")
    state_dict_path = path.join(model_dir_path, "source", "am.state_dict")

    # make the dataset
    trn_set = tts.make_dataset(a_trn_path, a_order, 'float32', c_trn_path, c_order, 'float32')
    val_set = tts.make_dataset(a_val_path, a_order, 'float32', c_val_path, c_order, 'float32')
    tst_set = tts.make_dataset(a_tst_path, a_order, 'float32', c_tst_path, c_order, 'float32')

    #
    trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")

    variance = th.load(var_path)
    model = tts.AcousticModelV3( a_order, hidden_size, c_order, variance=variance).to(device)
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) )

    print("+==================================================================+")
    print("|  Start Training Acoustic Model of the Source                     |")
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
    source_stat_dir_path = path.join("data", "source", "stat")
    stat_dir_path = path.join("data", "target", "stat")
    a_trn_path = path.join(set_dir_path, "a_trn.bin")
    a_val_path = path.join(set_dir_path, "a_val.bin")
    a_tst_path = path.join(set_dir_path, "a_tst.bin")
    c_trn_path = path.join(set_dir_path, "c_trn.bin")
    c_val_path = path.join(set_dir_path, "c_val.bin")
    c_tst_path = path.join(set_dir_path, "c_tst.bin")
    source_var_path = path.join(source_stat_dir_path, "c_trn_var.pt")
    var_path = path.join(stat_dir_path, "c_trn_var.pt")
    source_state_dict_path = path.join(model_dir_path, "source", "am.state_dict")
    trans_state_dict_path = path.join(model_dir_path, "target", "am_trans.state_dict")
    state_dict_path = path.join(model_dir_path, "target", "am.state_dict")

    # make the dataset
    trn_set = tts.make_dataset(a_trn_path, a_order, 'float32', c_trn_path, c_order, 'float32')
    val_set = tts.make_dataset(a_val_path, a_order, 'float32', c_val_path, c_order, 'float32')
    tst_set = tts.make_dataset(a_tst_path, a_order, 'float32', c_tst_path, c_order, 'float32')

    trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=3, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=3, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=3, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=3, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")

    source_variance = th.load(source_var_path)
    source_model = tts.AcousticModelV3( a_order, hidden_size, c_order, variance=source_variance).to(device)
    source_state_dict = th.load(source_state_dict_path)
    source_model.load_state_dict(source_state_dict)
    variance = th.load(var_path)
    model = tts.AcousticModelAdapt( source_model , variance=variance).to(device)

    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) )
    print("+==================================================================+")
    print("|  Start Training Transfer Layer of the Target                     |")
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
            th.save(model.state_dict(), trans_state_dict_path)

        if save == True:
            print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, "*"))
        else:
            print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, " "))

        print("+---------+---------------+---------------+---------------+--------+")
        sys.stdout.flush()

    # start fine tune
    model = model.to_acoustic_model_v3().to(device)
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) )
    print("+==================================================================+")
    print("|  Start Tuning the Acoustic Model of the Target                   |")
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
        writer.add_scalar('Training/Loss', trn_loss, epoch)
        writer.add_scalar('Testing/Loss', tst_loss, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.flush()
        sys.stdout.flush()
