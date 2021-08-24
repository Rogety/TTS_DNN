import os
import sys
import json
import numpy as np
import torch as th
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as fn
import torch.optim as optim
import argparse as ap
import tts

with open(os.path.join("configs","Configs.json")) as configs_file:
    configs = json.load(configs_file)

with open(os.path.join("configs","IOConfigs.json")) as configs_file:
    io_configs = json.load(configs_file)

a_order = io_configs["a_order"] # 1395
b_order = io_configs["b_order"]  # 1391
c_order = io_configs["c_order"] # 79
d_order = io_configs["d_order"] # 1
max_epoch = configs["max_epoch"]
batch_size = configs["batch_size"]
use_cuda = configs["use_cuda"]
hidden_size = configs["hidden_size"]
model_dir_path = os.path.join("model")
batch_size_lstm = 16

set_dir_path = os.path.join("data", "set")
stat_dir_path = os.path.join("data", "stat")
a_trn_path = os.path.join(set_dir_path, "a_trn.bin")
a_val_path = os.path.join(set_dir_path, "a_val.bin")
a_tst_path = os.path.join(set_dir_path, "a_tst.bin")
c_trn_path = os.path.join(set_dir_path, "c_trn.bin")
c_val_path = os.path.join(set_dir_path, "c_val.bin")
c_tst_path = os.path.join(set_dir_path, "c_tst.bin")
b_trn_path = os.path.join(set_dir_path, "b_trn.bin")
b_val_path = os.path.join(set_dir_path, "b_val.bin")
b_tst_path = os.path.join(set_dir_path, "b_tst.bin")
d_trn_path = os.path.join(set_dir_path, "d_trn.bin")
d_val_path = os.path.join(set_dir_path, "d_val.bin")
d_tst_path = os.path.join(set_dir_path, "d_tst.bin")
var_path = os.path.join(stat_dir_path, "c_trn_var.pt")

a_trn_path_lstm = os.path.join(set_dir_path, "lstm_a" , "trn")
a_val_path_lstm = os.path.join(set_dir_path, "lstm_a" , "val")
a_tst_path_lstm = os.path.join(set_dir_path, "lstm_a" , "tst")
c_trn_path_lstm = os.path.join(set_dir_path, "lstm_c" , "trn")
c_val_path_lstm = os.path.join(set_dir_path, "lstm_c" , "val")
c_tst_path_lstm = os.path.join(set_dir_path, "lstm_c" , "tst")
b_trn_path_lstm = os.path.join(set_dir_path, "lstm_b" , "trn")
b_val_path_lstm = os.path.join(set_dir_path, "lstm_b" , "val")
b_tst_path_lstm = os.path.join(set_dir_path, "lstm_b" , "tst")
d_trn_path_lstm = os.path.join(set_dir_path, "lstm_d" , "trn")
d_val_path_lstm = os.path.join(set_dir_path, "lstm_d" , "val")
d_tst_path_lstm = os.path.join(set_dir_path, "lstm_d" , "tst")


def train(model, trn_dl, device, optimizer, train_model, train_type):
    model.train()
    if train_model == 0:
        variance = model.variance * 0.1
        for i, (a, c) in enumerate(trn_dl):
            a, c = a.to(device), c.to(device)
            a.requires_grad_()
            optimizer.zero_grad()
            if train_type == 0:
                m_ = model.forward(a)
            elif train_type == 1:
                m_ = th.squeeze(model.forward(a))
                m_ = th.flatten(m_, start_dim=0, end_dim=1)
                c = th.flatten(c, start_dim=0, end_dim=1)
            loss = -th.mean( tts.log_gaussian_density(0.0, m_,variance , c) )
            loss.backward()
            optimizer.step()

    if train_model == 1:
        for i, (b, d) in enumerate(trn_dl):
            b, d = b.to(device), d.to(device=device, dtype=th.float)
            b.requires_grad_()
            optimizer.zero_grad()
            if train_type == 0:
                d_ = model.forward(b)
            elif train_type == 1:
                d_ = model.forward(b)
                d_ = th.flatten(d_, start_dim=0, end_dim=1)
                d = th.flatten(d, start_dim=0, end_dim=1)
            loss = fn.mse_loss(d_, d)
            loss.backward()
            optimizer.step()

def eval(model, trn_dl, val_dl, tst_dl, device, train_model, train_type):
    model.eval()
    trn_loss = 0.0
    val_loss = 0.0
    tst_loss = 0.0

    if train_model == 0:
        variance = model.variance * 0.1
        with th.no_grad():
            for i, (a, c) in enumerate( trn_dl ):
                a, c = a.to(device), c.to(device)
                if train_type == 0:
                    m_ = model.forward(a)
                elif train_type == 1:
                    m_ = th.squeeze(model.forward(a))
                    m_ = th.flatten(m_, start_dim=0, end_dim=1)
                    c = th.flatten(c, start_dim=0, end_dim=1)
                loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
                trn_loss += loss
            for i,(a,c) in enumerate( val_dl ):
                a,c= a.to(device), c.to(device)
                if train_type == 0:
                    m_ = model.forward(a)
                elif train_type == 1:
                    m_ = th.squeeze(model.forward(a))
                    m_ = th.flatten(m_, start_dim=0, end_dim=1)
                    c = th.flatten(c, start_dim=0, end_dim=1)
                loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
                val_loss += loss
            for i,(a,c) in enumerate( tst_dl ):
                a,c= a.to(device), c.to(device)
                if train_type == 0:
                    m_ = model.forward(a)
                elif train_type == 1:
                    m_ = th.squeeze(model.forward(a))
                    m_ = th.flatten(m_, start_dim=0, end_dim=1)
                    c = th.flatten(c, start_dim=0, end_dim=1)
                loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
                tst_loss += loss

        trn_loss /= len(trn_dl.dataset)
        val_loss /= len(val_dl.dataset)
        tst_loss /= len(tst_dl.dataset)

        trn_loss /= 1070
        val_loss /= 1070
        tst_loss /= 1070


    if train_model == 1:
        with th.no_grad():
            #i = 0
            for i, (b, d) in enumerate( trn_dl ):
                b, d = b.to(device), d.to(device=device, dtype=th.float)
                b.requires_grad_()
                if train_type == 0:
                    d_ = model.forward(b)
                elif train_type == 1:
                    d_ = model.forward(b)
                    d_ = th.flatten(d_, start_dim=0, end_dim=1)
                    d = th.flatten(d, start_dim=0, end_dim=1)
                loss = fn.mse_loss(d_, d, reduction='sum').item()
                trn_loss += loss
            #i = 0
            for i,(b, d) in enumerate( val_dl ):
                b, d = b.to(device), d.to(device=device, dtype=th.float)
                b.requires_grad_()
                if train_type == 0:
                    d_ = model.forward(b)
                elif train_type == 1:
                    d_ = model.forward(b)
                    d_ = th.flatten(d_, start_dim=0, end_dim=1)
                    d = th.flatten(d, start_dim=0, end_dim=1)
                loss = fn.mse_loss(d_, d, reduction='sum').item()
                val_loss += loss

            for i,(b, d) in enumerate( tst_dl ):
                b, d = b.to(device), d.to(device=device, dtype=th.float)
                b.requires_grad_()
                if train_type == 0:
                    d_ = model.forward(b)
                elif train_type == 1:
                    d_ = model.forward(b)
                    d_ = th.flatten(d_, start_dim=0, end_dim=1)
                    d = th.flatten(d, start_dim=0, end_dim=1)
                loss = fn.mse_loss(d_, d, reduction='sum').item()
                tst_loss += loss

        trn_loss /= len(trn_dl.dataset)
        val_loss /= len(val_dl.dataset)
        tst_loss /= len(tst_dl.dataset)
        trn_loss /= 360
        val_loss /= 360
        tst_loss /= 360

    return trn_loss, val_loss, tst_loss


if __name__ == '__main__':


    parser = ap.ArgumentParser()
    parser.add_argument("--train_model", type=str)
    parser.add_argument("--train_type", type=str)
    args = parser.parse_args()


    #train_model, train_type = 1,1

    if args.train_model == "acoustic":
        train_model = 0 #am
    elif args.train_model == "duration":
        train_model = 1
    if args.train_type == "DNN":
        train_type = 0 #dnn
    elif args.train_type == "LSTM":
        train_type = 1

    ## make dataset

    if train_model == 0 and train_type == 0:
        PATH_to_log_dir = "run/am/"
        writer = SummaryWriter(PATH_to_log_dir)
        state_dict_path = os.path.join(model_dir_path, "am.state_dict")
        trn_set = tts.make_dataset(a_trn_path, a_order, 'float32', c_trn_path, c_order, 'float32')
        val_set = tts.make_dataset(a_val_path, a_order, 'float32', c_val_path, c_order, 'float32')
        tst_set = tts.make_dataset(a_tst_path, a_order, 'float32', c_tst_path, c_order, 'float32')
        trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
        trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
        val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
        tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
        print("making acoustic_DNN datast done")
    elif train_model == 1 and train_type == 0:
        PATH_to_log_dir = "run/dur/"
        writer = SummaryWriter(PATH_to_log_dir)
        state_dict_path = os.path.join(model_dir_path, "dur.state_dict")
        trn_set = tts.make_dataset(b_trn_path, b_order, 'float32', d_trn_path, d_order, 'float32')
        val_set = tts.make_dataset(b_val_path, b_order, 'float32', d_val_path, d_order, 'float32')
        tst_set = tts.make_dataset(b_tst_path, b_order, 'float32', d_tst_path, d_order, 'float32')
        trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
        trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
        val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
        tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
        print("making duration_DNN datast done")
    elif train_model == 0 and train_type == 1: ##lstm
        PATH_to_log_dir = "run/am_lstm/"
        writer = SummaryWriter(PATH_to_log_dir)
        state_dict_path = os.path.join(model_dir_path, "am_lstm.state_dict")
        trn_set = tts.make_dataset_lstm(a_trn_path_lstm, a_order, 'float32', c_trn_path_lstm, c_order, 'float32', train_model)
        val_set = tts.make_dataset_lstm(a_val_path_lstm, a_order, 'float32', c_val_path_lstm, c_order, 'float32', train_model)
        tst_set = tts.make_dataset_lstm(a_tst_path_lstm, a_order, 'float32', c_tst_path_lstm, c_order, 'float32', train_model)
        trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size_lstm, num_workers=1, shuffle=True, drop_last=True )
        trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size_lstm, num_workers=0, shuffle=False, drop_last=False )
        val_dl = data.DataLoader(val_set, batch_size=batch_size_lstm, num_workers=0, shuffle=False, drop_last=False)
        tst_dl = data.DataLoader(tst_set, batch_size=batch_size_lstm, num_workers=0, shuffle=False, drop_last=False)
        print("making acoustic_LSTM datast done")
    elif train_model == 1 and train_type == 1:
        PATH_to_log_dir = "run/dur_lstm/"
        writer = SummaryWriter(PATH_to_log_dir)
        state_dict_path = os.path.join(model_dir_path, "dur_lstm.state_dict")
        trn_set = tts.make_dataset_lstm(b_trn_path_lstm, b_order, 'float32', d_trn_path_lstm, d_order, 'float32', train_model)
        val_set = tts.make_dataset_lstm(b_val_path_lstm, b_order, 'float32', d_val_path_lstm, d_order, 'float32', train_model)
        tst_set = tts.make_dataset_lstm(b_tst_path_lstm, b_order, 'float32', d_tst_path_lstm, d_order, 'float32', train_model)
        trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size_lstm, num_workers=0, shuffle=True, drop_last=True )
        trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size_lstm, num_workers=8, shuffle=False, drop_last=False )
        val_dl = data.DataLoader(val_set, batch_size=batch_size_lstm, num_workers=8, shuffle=False, drop_last=False)
        tst_dl = data.DataLoader(tst_set, batch_size=batch_size_lstm, num_workers=8, shuffle=False, drop_last=False)
        print("making duration_LSTM datast done")


    device = th.device("cuda" if 1 else "cpu")
    variance = th.load(var_path)
    if train_model == 0 and train_type == 0:
        model = tts.AcousticModel( a_order, hidden_size, c_order, variance=variance).to(device)
    elif train_model == 1 and train_type == 0:
        model = tts.DurationModel( b_order, hidden_size ).to(device) # (1392 , 2048)
    elif train_model == 0 and train_type == 1:
        model = tts.AcousticModel_LSTM( a_order, hidden_size, c_order, variance=variance).to(device)
    elif train_model == 1 and train_type == 1:
        model = tts.DurationModel_LSTM( b_order, hidden_size ).to(device)

    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) ,lr=0.005)
    print(model)





    ## training
    print("+==================================================================+")
    print("|  Start Training Acoustic Model of the Source                     |")
    print("+=========+===============+===============+===============+========+")
    print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
    print("+=========+===============+===============+===============+========+")
    sys.stdout.flush()
    min_trn_loss, min_val_loss, min_tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device, train_model, train_type)
    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()

    for epoch in range(1, max_epoch+1):
        train(model, trn_dl_for_trainig, device, optimizer, train_model, train_type)
        trn_loss, val_loss, tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device, train_model, train_type)
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
