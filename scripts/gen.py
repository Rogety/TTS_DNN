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
import utils as ut

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
#use_cuda = configs["use_cuda"]
use_cuda = 0
sampling_rate = configs["sampling_rate"]
frame_shift = configs["frame_shift"]
number_of_states = configs["number_of_states"]

state_in_phone_min = configs["state_in_phone_min"]
state_in_phone_max = configs["state_in_phone_max"]
frame_in_state_min = configs["frame_in_state_min"]
frame_in_state_max = configs["frame_in_state_max"]
frame_in_phone_min = configs["frame_in_phone_min"]
frame_in_phone_max = configs["frame_in_phone_max"]

model_dir_path = path.join("model")
source_dir_path = configs["source_dir_path"]
gen_dir_path = configs["gen_dir_path"]

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


    if not path.isdir(gen_dir_path):
        os.mkdir(gen_dir_path)
        os.mkdir(path.join(gen_dir_path,"from_acoustic_DNN"))
        os.mkdir(path.join(gen_dir_path,"from_duration_DNN"))
        os.mkdir(path.join(gen_dir_path,"from_acoustic_LSTM"))
        os.mkdir(path.join(gen_dir_path,"from_duration_LSTM"))
        os.mkdir(path.join(gen_dir_path,"set"))
        os.mkdir(path.join(gen_dir_path,"set","a"))
        os.mkdir(path.join(gen_dir_path,"set","b"))


    if train_model == 0 and train_type == 0:
        print(gen_dir_path)
        #a_dir_path = path.join("gen","set","a")
        a_dir_path = path.join("data","set","a")
        stat_dir_path = path.join("data", "stat")
        gen_dir_path = path.join("gen", "from_acoustic_DNN")
        var_path = path.join(stat_dir_path, "c_trn_var.pt")
        state_dict_path = path.join(model_dir_path, "am.state_dict")
        device = th.device("cuda" if use_cuda else "cpu")

        variance = th.load(var_path)
        model = tts.AcousticModel( a_order, hidden_size, c_order, variance=variance).to(device)
        state_dict = th.load( state_dict_path )
        model.load_state_dict(state_dict)

        ## acoustic model DNN
        with th.no_grad():
            var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
            #var_lf0_mgc = variance[0:c_order].view(1,1, (lf0_order+mgc_order))
            for filename in sorted(os.listdir(a_dir_path)):
                if filename.endswith(".bin"):
                    base = path.splitext(filename)[0]
                    print(base)
                    a_path = path.join(a_dir_path, "{}.bin".format(base) )
                    a = ut.load_binfile(a_path)
                    print("a :" , type(a) , len(a))
                    a = np.asarray(a,dtype=np.float32)
                    a = np.reshape(a , (-1,a_order))
                    a = th.from_numpy(a)
                    a = a.to(device)
                    #a.view(-1,a_order).to(device)

                    #.view(-1,a_order).to(device)
                    # c_: lf0 mgc dlf0 dmgc ddlf0 ddmgc uv

                    c_ = model(a) ## mean


                    lf0 = c_[:,0]
                    uv = ( c_[:, c_order-1] > 0.4 ).cpu()
                    c =  c_[:, 0:c_order-1].view(-1, 3, lf0_order+mgc_order) ## (650,3,26)
                    lf0_m = c[:,:,0:lf0_order].view(-1, 3*lf0_order).cpu()
                    mgc_m = c[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).cpu()
                    lf0_v = var_lf0_mgc[:,:,0:lf0_order].view(-1, 3*lf0_order).expand_as(lf0_m)
                    mgc_v = var_lf0_mgc[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).expand_as( mgc_m )
                    '''
                    print("c_:",c_.shape)
                    lf0 = c_[:,0]
                    print("lf0:",lf0.shape)
                    c =  c_[:, 0:c_order].view(-1, 1, lf0_order+mgc_order) ## (650,3,26)
                    print("c:",c.shape)
                    lf0_m = c[:,:,0:lf0_order].view(-1, 1*lf0_order).cpu()
                    print("lf0_m:",lf0_m.shape)
                    mgc_m = c[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 1*mgc_order).cpu()
                    print("mgc_m:",mgc_m.shape)
                    lf0_v = var_lf0_mgc[:,:,0:lf0_order].view(-1, 1*lf0_order).expand_as(lf0_m)
                    print("lf0_v:",lf0_v.shape)
                    mgc_v = var_lf0_mgc[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 1*mgc_order).expand_as( mgc_m )
                    print("mgc_v:",mgc_v.shape)
                    '''
                    lf0_mlpg = th.cat( (lf0_m,lf0_v), dim=1 )
                    mgc_mlpg = th.cat( (mgc_m,mgc_v), dim=1 )

                    lf0_mlpg = lf0_mlpg.contiguous().numpy().flatten()
                    mgc_mlpg = mgc_mlpg.contiguous().numpy().flatten()

                    #import pdb; pdb.set_trace()

                    ut.save_binfile(os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)), lf0_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)), mgc_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.uv'.format(base)), uv.float())
                    '''
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)), lf0_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)), mgc_mlpg)
                    #ut.save_binfile(os.path.join(gen_dir_path, '{}.uv'.format(base)), uv.float())
                    '''


    elif train_model == 1 and train_type == 0:
        ## duration model DNN
        #b_dir_path = path.join("gen","set","b")
        b_dir_path = path.join("data","set","b")
        stat_dir_path = path.join("data", "stat")
        gen_dir_path = path.join("gen", "from_duration_DNN")
        var_path = path.join(stat_dir_path, "c_trn_var.pt")
        dur_state_dict_path = path.join(model_dir_path, "dur.state_dict")
        am_state_dict_path = path.join(model_dir_path, "am.state_dict")

        device = th.device("cuda" if use_cuda else "cpu")
        variance = th.load(var_path)

        dur_model = tts.DurationModel( b_order, hidden_size).to(device)
        am_model = tts.AcousticModel( a_order, hidden_size, c_order, variance=variance).to(device)

        dur_state_dict = th.load(dur_state_dict_path )
        am_state_dict = th.load( am_state_dict_path )

        dur_model.load_state_dict(dur_state_dict)
        am_model.load_state_dict(am_state_dict)


        with th.no_grad():
            var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
            for filename in os.listdir(b_dir_path):
                if filename.endswith(".bin"):
                    base = path.splitext(filename)[0]
                    print("gen lf0.mlpg, mgc.mlpg from b {}".format(base))
                    b_path = path.join(b_dir_path, "{}.bin".format(base))
                    b = ut.load_binfile(b_path)
                    #.view(-1,b_order).to(device)
                    b = np.asarray(b,dtype=np.float32)
                    b = np.reshape(b , (-1,b_order))
                    b = th.from_numpy(b)
                    b = b.to(device)

                    #d = dur_model(b).argmax(1, keepdim=False).view(-1, number_of_states)
                    d = dur_model(b).round().int().view(-1, number_of_states)
                    #print("d:",d.shape)
                    #import pdb; pdb.set_trace()
                    # 去掉 語速資訊
                    #b = b[:,0: b_order-1].view(-1, number_of_states, b_order-1)
                    b = b[:,0: b_order].view(-1, number_of_states, b_order)
                    a = []
                    #print("b :",b.shape)
                    #print("d :",d.shape)


                    for (phone_index, phone) in enumerate(d):
                        # shape = (number_of_states, b_order-1)
                        #print(phone_index, phone)
                        ans_of_cur_phone = b[phone_index]
                        total_frames_in_phone = phone.sum().item()
                        cur_frame_in_phone = 0

                        for (state_index, state) in enumerate(phone):
                            #(b_order-1)
                            ans_of_cur_state = ans_of_cur_phone[state_index]
                            total_frames_in_state = state.item()
                            #print(state_index, state)
                            #print("total_frames_in_state:",total_frames_in_state)
                            #import pdb; pdb.set_trace()
                            for frame_index in range(0,total_frames_in_state):
                                ans_of_cur_frame = ans_of_cur_state.clone()

                                frame_in_state_fwd_n = (frame_index) / (frame_in_state_max-frame_in_state_min)
                                frame_in_state_bwd_n = (total_frames_in_state - frame_index - frame_in_state_min) / (frame_in_phone_max-frame_in_phone_min)
                                frame_in_phone_fwd_n = (cur_frame_in_phone) / (frame_in_phone_max - frame_in_phone_min)
                                frame_in_phone_bwd_n = (total_frames_in_phone - cur_frame_in_phone - frame_in_phone_min ) / (frame_in_phone_max-frame_in_phone_min)

                                frame_info = ans_of_cur_frame.new([frame_in_state_fwd_n, frame_in_state_bwd_n, frame_in_phone_fwd_n, frame_in_phone_bwd_n])
                                #print(frame_index,frame_info)
                                a.append( th.cat( (ans_of_cur_state, frame_info) ) )
                                cur_frame_in_phone += 1

                    a = th.stack(a)
                    c_ = am_model(a)

                    lf0 = c_[:,0]
                    uv = ( c_[:, c_order-1] > 0.4 ).cpu()

                    c =  c_[:, 0:c_order-1].view(-1, 3, lf0_order+mgc_order)

                    lf0_m = c[:,:,0:lf0_order].view(-1, 3*lf0_order).cpu()
                    mgc_m = c[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).cpu()
                    lf0_v = var_lf0_mgc[:,:,0:lf0_order].view(-1, 3*lf0_order).expand_as(lf0_m)
                    mgc_v = var_lf0_mgc[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).expand_as( mgc_m )

                    lf0_mlpg = th.cat( (lf0_m,lf0_v), dim=1 )
                    mgc_mlpg = th.cat( (mgc_m,mgc_v), dim=1 )

                    lf0_mlpg = lf0_mlpg.contiguous().numpy().flatten()
                    mgc_mlpg = mgc_mlpg.contiguous().numpy().flatten()

                    ut.save_binfile(os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)), lf0_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)), mgc_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.uv'.format(base)), uv.float())

    elif train_model == 0 and train_type == 1:
        print(gen_dir_path)
        #a_dir_path = path.join("gen","set","a")
        a_dir_path = path.join("data","set","lstm_a")
        stat_dir_path = path.join("data", "stat")
        gen_dir_path = path.join("gen", "from_acoustic_LSTM")
        var_path = path.join(stat_dir_path, "c_trn_var.pt")
        state_dict_path = path.join(model_dir_path, "am_lstm.state_dict")
        device = th.device("cuda" if use_cuda else "cpu")

        variance = th.load(var_path)
        model = tts.AcousticModel_LSTM( a_order, hidden_size, c_order, variance=variance).to(device)
        state_dict = th.load( state_dict_path )
        model.load_state_dict(state_dict)

        ## acoustic model DNN
        with th.no_grad():
            var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
            for filename in os.listdir(a_dir_path):
                if filename.endswith(".bin"):
                    base = path.splitext(filename)[0]
                    print(base)
                    a_path = path.join(a_dir_path, "{}.bin".format(base) )
                    a = ut.load_binfile(a_path)
                    #print(a,type(a) , len(a))
                    a = np.asarray(a,dtype=np.float32)
                    a = np.reshape(a , (-1,a_order))
                    a = th.from_numpy(a)
                    a = th.unsqueeze(a , dim=0)
                    a = a.to(device)
                    #a.view(-1,a_order).to(device)

                    #.view(-1,a_order).to(device)
                    # c_: lf0 mgc dlf0 dmgc ddlf0 ddmgc uv
                    c_ = model(a) ## mean
                    c_ = th.squeeze(c_)
                    lf0 = c_[:,0]
                    uv = ( c_[:, c_order-1] > 0.4 ).cpu()
                    c =  c_[:, 0:c_order-1].view(-1, 3, lf0_order+mgc_order) ## (650,3,26)

                    lf0_m = c[:,:,0:lf0_order].view(-1, 3*lf0_order).cpu()
                    mgc_m = c[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).cpu()

                    lf0_v = var_lf0_mgc[:,:,0:lf0_order].view(-1, 3*lf0_order).expand_as(lf0_m)
                    mgc_v = var_lf0_mgc[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).expand_as( mgc_m )

                    lf0_mlpg = th.cat( (lf0_m,lf0_v), dim=1 )
                    mgc_mlpg = th.cat( (mgc_m,mgc_v), dim=1 )

                    lf0_mlpg = lf0_mlpg.contiguous().numpy().flatten()
                    mgc_mlpg = mgc_mlpg.contiguous().numpy().flatten()

                    #import pdb; pdb.set_trace()
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)), lf0_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)), mgc_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.uv'.format(base)), uv.float())


    elif train_model == 1 and train_type == 1:
        ## duration model DNN
        #b_dir_path = path.join("gen","set","b")
        b_dir_path = path.join("data","set","lstm_b")
        stat_dir_path = path.join("data", "stat")
        gen_dir_path = path.join("gen", "from_duration_LSTM")
        var_path = path.join(stat_dir_path, "c_trn_var.pt")
        dur_state_dict_path = path.join(model_dir_path, "dur_lstm.state_dict")
        am_state_dict_path = path.join(model_dir_path, "am_lstm.state_dict")

        device = th.device("cuda" if use_cuda else "cpu")
        variance = th.load(var_path)

        dur_model = tts.DurationModel_LSTM( b_order, hidden_size).to(device)
        am_model = tts.AcousticModel_LSTM( a_order, hidden_size, c_order, variance=variance).to(device)

        dur_state_dict = th.load(dur_state_dict_path )
        am_state_dict = th.load( am_state_dict_path )

        dur_model.load_state_dict(dur_state_dict)
        am_model.load_state_dict(am_state_dict)


        with th.no_grad():
            var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
            for filename in os.listdir(b_dir_path):
                if filename.endswith(".bin"):
                    base = path.splitext(filename)[0]
                    print("gen lf0.mlpg, mgc.mlpg from b {}".format(base))
                    b_path = path.join(b_dir_path, "{}.bin".format(base))
                    b = ut.load_binfile(b_path)
                    #.view(-1,b_order).to(device)
                    b = np.asarray(b,dtype=np.float32)
                    b = np.reshape(b , (-1,b_order))
                    b = th.from_numpy(b)
                    b = th.unsqueeze(b , dim=0)
                    b = b.to(device)


                    #d = dur_model(b).argmax(1, keepdim=False).view(-1, number_of_states)
                    d = dur_model(b).round().int().view(-1, number_of_states)


                    #print("d:",d.shape)
                    #import pdb; pdb.set_trace()
                    # 去掉 語速資訊
                    #b = b[:,0: b_order-1].view(-1, number_of_states, b_order-1)
                    b = b[:,0: b_order].view(-1, number_of_states, b_order)
                    a = []
                    #print("b :",b.shape)
                    #print("d :",d.shape)


                    for (phone_index, phone) in enumerate(d):
                        # shape = (number_of_states, b_order-1)
                        ans_of_cur_phone = b[phone_index]
                        total_frames_in_phone = phone.sum().item()
                        cur_frame_in_phone = 0

                        for (state_index, state) in enumerate(phone):
                            #(b_order-1)
                            ans_of_cur_state = ans_of_cur_phone[state_index]
                            total_frames_in_state = state.item()
                            #print(state_index, state)
                            #print("total_frames_in_state:",total_frames_in_state)
                            #import pdb; pdb.set_trace()
                            for frame_index in range(0,total_frames_in_state):
                                ans_of_cur_frame = ans_of_cur_state.clone()

                                frame_in_state_fwd_n = (frame_index) / (frame_in_state_max-frame_in_state_min)
                                frame_in_state_bwd_n = (total_frames_in_state - frame_index - frame_in_state_min) / (frame_in_phone_max-frame_in_phone_min)
                                frame_in_phone_fwd_n = (cur_frame_in_phone) / (frame_in_phone_max - frame_in_phone_min)
                                frame_in_phone_bwd_n = (total_frames_in_phone - cur_frame_in_phone - frame_in_phone_min ) / (frame_in_phone_max-frame_in_phone_min)

                                frame_info = ans_of_cur_frame.new([frame_in_state_fwd_n, frame_in_state_bwd_n, frame_in_phone_fwd_n, frame_in_phone_bwd_n])
                                #print(frame_index,frame_info)
                                a.append( th.cat( (ans_of_cur_state, frame_info) ) )
                                cur_frame_in_phone += 1

                    a = th.stack(a)
                    a = th.unsqueeze(a , dim=0)
                    c_ = am_model(a)
                    c_ = th.squeeze(c_)

                    lf0 = c_[:,0]
                    uv = ( c_[:, c_order-1] > 0.4 ).cpu()

                    c =  c_[:, 0:c_order-1].view(-1, 3, lf0_order+mgc_order)

                    lf0_m = c[:,:,0:lf0_order].view(-1, 3*lf0_order).cpu()
                    mgc_m = c[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).cpu()
                    lf0_v = var_lf0_mgc[:,:,0:lf0_order].view(-1, 3*lf0_order).expand_as(lf0_m)
                    mgc_v = var_lf0_mgc[:,:,lf0_order:(lf0_order+mgc_order)].contiguous().view(-1, 3*mgc_order).expand_as( mgc_m )

                    lf0_mlpg = th.cat( (lf0_m,lf0_v), dim=1 )
                    mgc_mlpg = th.cat( (mgc_m,mgc_v), dim=1 )

                    lf0_mlpg = lf0_mlpg.contiguous().numpy().flatten()
                    mgc_mlpg = mgc_mlpg.contiguous().numpy().flatten()

                    ut.save_binfile(os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)), lf0_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)), mgc_mlpg)
                    ut.save_binfile(os.path.join(gen_dir_path, '{}.uv'.format(base)), uv.float())
