
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



a_order = 1386 # 1395
b_order = 1382 # 1391
c_order = 79 # 79
d_order = 1 # 1
max_epoch = configs["max_epoch"]
batch_size = configs["batch_size"]
use_cuda = configs["use_cuda"]
hidden_size = configs["hidden_size"]
model_dir_path = os.path.join("model")


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

a_trn_path_lstm = os.path.join(set_dir_path, "trn")
a_val_path_lstm = os.path.join(set_dir_path, "lstm_a" , "val")
a_tst_path_lstm = os.path.join(set_dir_path, "lstm_a" , "tst")
c_trn_path_lstm = os.path.join(set_dir_path, "trn")
c_val_path_lstm = os.path.join(set_dir_path, "lstm_c" , "val")
c_tst_path_lstm = os.path.join(set_dir_path, "lstm_c" , "tst")
b_trn_path_lstm = os.path.join(set_dir_path, "trn")
b_val_path_lstm = os.path.join(set_dir_path, "lstm_b" , "val")
b_tst_path_lstm = os.path.join(set_dir_path, "lstm_b" , "tst")
d_trn_path_lstm = os.path.join(set_dir_path, "trn")
d_val_path_lstm = os.path.join(set_dir_path, "lstm_d" , "val")
d_tst_path_lstm = os.path.join(set_dir_path, "lstm_d" , "tst")


def train_dur_lstm(model, trn_dl, device, optimizer ):
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

def train_am_lstm(model, trn_dl, device, optimizer ):
    model.train()
    variance = model.variance * 0.1
    for i, (a, c) in enumerate(trn_dl):
        a, c = a.to(device), c.to(device)
        a.requires_grad_()
        optimizer.zero_grad()
        m_ = th.squeeze(model.forward(a)) ## 4096 * 79
        loss = -th.mean( tts.log_gaussian_density(0.0, m_,variance , c) )
        loss.backward()
        optimizer.step()

def train_am(model, trn_dl, device, optimizer ):
    model.train()
    variance = model.variance * 0.1
    for i, (a, c) in enumerate(trn_dl):
        a, c = a.to(device), c.to(device)
        a.requires_grad_()
        optimizer.zero_grad()
        print("a :",a.shape)
        m_ = model.forward(a) ## 4096 * 79
        print("m_ :",m_.shape)
        loss = -th.mean( tts.log_gaussian_density(0.0, m_,variance , c) )
        loss.backward()
        optimizer.step()

def train_dur(model, trn_dl, device, optimizer ):
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

def eval_am(model, trn_dl, val_dl, tst_dl, device):
    model.eval()
    variance = model.variance * 0.1
    trn_loss = 0.0
    val_loss = 0.0
    tst_loss = 0.0
    with th.no_grad():
        #i = 0
        for i, (a, c) in enumerate( trn_dl ):
            a, c = a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
            trn_loss += loss

        #i = 0
        for i,(a,c) in enumerate( val_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
            val_loss += loss

        for i,(a,c) in enumerate( tst_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
            tst_loss += loss

    trn_loss /= len(trn_dl.dataset)
    val_loss /= len(val_dl.dataset)
    tst_loss /= len(tst_dl.dataset)
    return trn_loss, val_loss, tst_loss

def eval_am_lstm(model, trn_dl, val_dl, tst_dl, device):
    model.eval()
    variance = model.variance * 0.1
    trn_loss = 0.0
    val_loss = 0.0
    tst_loss = 0.0
    with th.no_grad():
        #i = 0
        for i, (a, c) in enumerate( trn_dl ):
            a, c = a.to(device), c.to(device)
            m_ = th.squeeze(model.forward(a))
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
            trn_loss += loss

        #i = 0
        for i,(a,c) in enumerate( val_dl ):
            a,c= a.to(device), c.to(device)
            m_ = th.squeeze(model.forward(a))
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
            val_loss += loss

        for i,(a,c) in enumerate( tst_dl ):
            a,c= a.to(device), c.to(device)
            m_ = th.squeeze(model.forward(a))
            loss = -th.sum( tts.log_gaussian_density(0.0,m_, variance, c) ).item()
            tst_loss += loss

    trn_loss /= len(trn_dl.dataset)
    val_loss /= len(val_dl.dataset)
    tst_loss /= len(tst_dl.dataset)
    return trn_loss, val_loss, tst_loss

def eval_dur(model, trn_dl, val_dl, tst_dl, device):
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

def eval_dur_lstm(model, trn_dl, val_dl, tst_dl, device):
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

def training_am_model():

    PATH_to_log_dir = "run/am/"
    writer = SummaryWriter(PATH_to_log_dir)

    state_dict_path = os.path.join(model_dir_path, "am.state_dict")
    # make the dataset
    trn_set = tts.make_dataset(a_trn_path, a_order, 'float32', c_trn_path, c_order, 'float32')
    val_set = tts.make_dataset(a_val_path, a_order, 'float32', c_val_path, c_order, 'float32')
    tst_set = tts.make_dataset(a_tst_path, a_order, 'float32', c_tst_path, c_order, 'float32')

    trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")
    variance = th.load(var_path)

    model = tts.AcousticModel( a_order, hidden_size, c_order, variance=variance).to(device)
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) ,lr=0.005)
    print(model)

    print("+==================================================================+")
    print("|  Start Training Acoustic Model of the Source                     |")
    print("+=========+===============+===============+===============+========+")
    print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
    print("+=========+===============+===============+===============+========+")
    sys.stdout.flush()
    min_trn_loss, min_val_loss, min_tst_loss = eval_am(model, trn_dl_for_eval, val_dl, tst_dl, device)
    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()

    for epoch in range(1, max_epoch+1):
        train_am(model, trn_dl_for_trainig, device, optimizer)
        trn_loss, val_loss, tst_loss = eval_am(model, trn_dl_for_eval, val_dl, tst_dl, device)
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



    return 0

def training_dur_model():

    PATH_to_log_dir = "run/dur/"
    writer = SummaryWriter(PATH_to_log_dir)

    state_dict_path = os.path.join(model_dir_path, "dur.state_dict")

    trn_set = tts.make_dataset(b_trn_path, b_order, 'float32', d_trn_path, d_order, 'int32')
    val_set = tts.make_dataset(b_val_path, b_order, 'float32', d_val_path, d_order, 'int32')
    tst_set = tts.make_dataset(b_tst_path, b_order, 'float32', d_tst_path, d_order, 'int32')

    trn_dl_for_trainig = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")
    model = tts.DurationModel( b_order, hidden_size ).to(device) # (1392 , 2048)
    optimizer = optim.Adam( model.parameters() , lr=0.005 )

    print(model)
    print("+==================================================================+")
    print("|  Start Training Duration Model of Source                         |")
    print("+=========+===============+===============+===============+========+")
    print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
    print("+=========+===============+===============+===============+========+")
    sys.stdout.flush()
    min_trn_loss, min_val_loss, min_tst_loss = eval_dur(model, trn_dl_for_eval, val_dl, tst_dl, device)
    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()

    for epoch in range(1, max_epoch+1):
        train_dur(model, trn_dl_for_trainig, device, optimizer)
        trn_loss, val_loss, tst_loss = eval_dur(model, trn_dl_for_eval, val_dl, tst_dl, device)
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

        writer.add_scalar('Loss/training', trn_loss, epoch)
        writer.add_scalar('Loss/testing', tst_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.flush()

        sys.stdout.flush()

    return 0

def training_am_model_lstm():

    print("training_am_model_lstm")
    PATH_to_log_dir = "run/am_lstm/"
    writer = SummaryWriter(PATH_to_log_dir)

    state_dict_path = os.path.join(model_dir_path, "am_lstm.state_dict")
    trn_set_lstm = tts.make_dataset_lstm(a_trn_path_lstm, a_order, 'float32', c_trn_path_lstm, c_order, 'float32')
    val_set_lstm = tts.make_dataset_lstm(a_val_path_lstm, a_order, 'float32', c_val_path_lstm, c_order, 'float32')
    tst_set_lstm = tts.make_dataset_lstm(a_tst_path_lstm, a_order, 'float32', c_tst_path_lstm, c_order, 'float32')
    #print(trn_set_lstm)
    trn_dl_for_trainig = data.DataLoader(trn_set_lstm, batch_size=1, num_workers=1, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set_lstm, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set_lstm, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set_lstm, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")

    variance = th.load(var_path)
    #model = tts.AcousticModel( a_order, hidden_size, c_order, variance=variance).to(device)
    model = tts.AcousticModel_LSTM( a_order, hidden_size, c_order, variance=variance).to(device)
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) ,lr=0.005)

    print(model)
    print("+==================================================================+")
    print("|  Start Training Acoustic Model of the Source                     |")
    print("+=========+===============+===============+===============+========+")
    print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
    print("+=========+===============+===============+===============+========+")

    sys.stdout.flush()
    min_trn_loss, min_val_loss, min_tst_loss = eval_am_lstm(model, trn_dl_for_eval, val_dl, tst_dl, device)
    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()
    print("start_training")

    for epoch in range(1, max_epoch+1):
        train_am_lstm(model, trn_dl_for_trainig, device, optimizer)
        trn_loss, val_loss, tst_loss = eval_am_lstm(model, trn_dl_for_eval, val_dl, tst_dl, device)
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

        ##eval
        #trn_loss, val_loss, tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device)
        sys.stdout.flush()

    return 0

def training_dur_model_lstm():

    print("training_dur_model_lstm")
    PATH_to_log_dir = "run/dur_lstm/"
    writer = SummaryWriter(PATH_to_log_dir)

    state_dict_path = os.path.join(model_dir_path, "dur_lstm.state_dict")

    trn_set_lstm = tts.make_dataset_lstm_dur(b_trn_path_lstm, b_order, 'float32', d_trn_path_lstm, d_order, 'int32')
    val_set_lstm = tts.make_dataset_lstm_dur(b_val_path_lstm, b_order, 'float32', d_val_path_lstm, d_order, 'int32')
    tst_set_lstm = tts.make_dataset_lstm_dur(b_tst_path_lstm, b_order, 'float32', d_tst_path_lstm, d_order, 'int32')

    trn_dl_for_trainig = data.DataLoader(trn_set_lstm, batch_size=1, num_workers=0, shuffle=True, drop_last=True )
    trn_dl_for_eval = data.DataLoader(trn_set_lstm, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
    val_dl = data.DataLoader(val_set_lstm, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
    tst_dl = data.DataLoader(tst_set_lstm, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    device = th.device("cuda" if use_cuda else "cpu")
    model = tts.DurationModel_LSTM( b_order, hidden_size ).to(device) # (1392 , 2048)
    optimizer = optim.Adam( model.parameters() , lr=0.005 )

    print(model)
    print("+==================================================================+")
    print("|  Start Training Duration Model of Source                         |")
    print("+=========+===============+===============+===============+========+")
    print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
    print("+=========+===============+===============+===============+========+")
    sys.stdout.flush()
    min_trn_loss, min_val_loss, min_tst_loss = eval_dur(model, trn_dl_for_eval, val_dl, tst_dl, device)
    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()

    for epoch in range(1, max_epoch+1):
        train_dur_lstm(model, trn_dl_for_trainig, device, optimizer)
        trn_loss, val_loss, tst_loss = eval_dur_lstm(model, trn_dl_for_eval, val_dl, tst_dl, device)
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

        writer.add_scalar('Loss/training', trn_loss, epoch)
        writer.add_scalar('Loss/testing', tst_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.flush()

        sys.stdout.flush()

    return 0


if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument("--am", action="store_true")
    parser.add_argument("--dur", action="store_true")
    parser.add_argument("--am_lstm", action="store_true")
    parser.add_argument("--dur_lstm", action="store_true")
    args = parser.parse_args()
    if args.am:
        training_am_model()
    if args.dur:
        training_dur_model()
    if args.am_lstm:
        training_am_model_lstm()
    if args.dur_lstm:
        training_dur_model_lstm()
