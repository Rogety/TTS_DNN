import os
import sys
import json
import numpy as np
import torch as th
import torch.utils.data as data
import torch.optim as optim
import argparse as ap
import h5py
import shutil
import re

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

mgc_order = configs["mgc_order"]
lf0_order = configs["lf0_order"]
hidden_size = configs["hidden_size"]
use_cuda = configs["use_cuda"]
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


if args.source:
    b_dir_path = path.join("data","gen","set","b")
    stat_dir_path = path.join("data", "source", "stat")
    gen_dir_path = path.join("gen","source", "from_b")
    var_path = path.join(stat_dir_path, "c_trn_var.pt")
    dur_state_dict_path = path.join(model_dir_path, "source", "dur.state_dict")
    am_state_dict_path = path.join(model_dir_path, "source", "am.state_dict")

    device = th.device("cuda" if use_cuda else "cpu")
    variance = th.load(var_path)

    dur_model = tts.DurationModelV2( b_order, hidden_size).to(device)
    am_model = tts.AcousticModelV3( a_order, hidden_size, c_order, variance=variance).to(device)

    dur_state_dict = th.load(dur_state_dict_path )
    am_state_dict = th.load( am_state_dict_path )

    dur_model.load_state_dict(dur_state_dict)
    am_model.load_state_dict(am_state_dict)

    ## for hdf5 dir
    hdf5_dir_path = path.join("gen", "source","output_hdf5")
    trn_hdf5_path = path.join(hdf5_dir_path,"tr_slt")
    ev_hdf5_path = path.join(hdf5_dir_path,"ev_slt")
    hdf5_path = path.join(hdf5_dir_path,"hdf5")
    if os.path.isdir(trn_hdf5_path) != True:
        os.makedirs(trn_hdf5_path)
    if os.path.isdir(ev_hdf5_path)  != True:
        os.makedirs(ev_hdf5_path)
    if os.path.isdir(hdf5_path)  != True:
        os.makedirs(hdf5_path)
    hdf5_filename_list = []

    with th.no_grad():
        var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
        for filename in os.listdir(b_dir_path):
            if filename.endswith(".bin"):
                base = path.splitext(filename)[0]
                print("gen lf0.mlpg, mgc.mlpg from b {}".format(base))
                b_path = path.join(b_dir_path, "{}.bin".format(base))
                b = tts.load_bin( b_path, 'float32' ).view(-1,b_order).to(device)
                #d = dur_model(b).argmax(1, keepdim=False).view(-1, number_of_states)
                d = dur_model(b).round().int().view(-1, number_of_states)
                # 去掉 語速資訊
                b = b[:,0: b_order-1].view(-1, number_of_states, b_order-1)
                a = []

                for (phone_index, phone) in enumerate(d):
                    # shape = (number_of_states, b_order-1)
                    ans_of_cur_phone = b[phone_index]
                    total_frames_in_phone = phone.sum().item()
                    cur_frame_in_phone = 0

                    for (state_index, state) in enumerate(phone):
                        #(b_order-1)
                        ans_of_cur_state = ans_of_cur_phone[state_index]
                        total_frames_in_state = state.item()

                        for frame_index in range(0,total_frames_in_state):
                            ans_of_cur_frame = ans_of_cur_state.clone()

                            frame_in_state_fwd_n = (frame_index) / (frame_in_state_max-frame_in_state_min)
                            frame_in_state_bwd_n = (total_frames_in_state - frame_index - frame_in_state_min) / (frame_in_phone_max-frame_in_phone_min)
                            frame_in_phone_fwd_n = (cur_frame_in_phone) / (frame_in_phone_max - frame_in_phone_min)
                            frame_in_phone_bwd_n = (total_frames_in_phone - cur_frame_in_phone - frame_in_phone_min ) / (frame_in_phone_max-frame_in_phone_min)

                            frame_info = ans_of_cur_frame.new([frame_in_state_fwd_n, frame_in_state_bwd_n, frame_in_phone_fwd_n, frame_in_phone_bwd_n])

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

                # print("lf0_mlpg :",lf0_mlpg.shape)
                # print("mgc_mlpg :",mgc_mlpg.shape)
                # print("uv :",uv.shape)

                uv_reshape = th.reshape(uv, (-1,1)).float()

                lf0_without_dynamic = lf0_m[:,0:lf0_order]
                f0_without_dynamic = th.exp(lf0_without_dynamic)
                mgc_without_dynamic = mgc_m[:,0:mgc_order]
                feat = th.cat( (uv_reshape,f0_without_dynamic,mgc_without_dynamic)  , dim=1)
                #feat = th.cat( (lf0_m,mgc_m,uv_reshape)  , dim=1)

                #print(f0_without_dynamic)
                #print(lf0_without_dynamic)
                # print(lf0_m.shape,lf0_without_dynamic.shape)
                # print(lf0_without_dynamic[0][0],lf0_m[0,0] )
                # print(lf0_without_dynamic[1][0],lf0_m[1,0] )
                # print(lf0_without_dynamic[2][0],lf0_m[2,0] )
                #
                # print(mgc_m.shape,mgc_without_dynamic.shape)
                # print(mgc_without_dynamic[0][0],mgc_m[0,0] )
                # print(mgc_without_dynamic[1][0],mgc_m[1,0] )
                # print(mgc_without_dynamic[2][0],mgc_m[2,0] )
                # print(mgc_without_dynamic[0][0],mgc_m[0,0] )
                # print(mgc_without_dynamic[0][1],mgc_m[0,1] )
                # print(mgc_without_dynamic[0][2],mgc_m[0,2] )
                #


                # print(type(lf0_without_dynamic))
                # print(lf0_without_dynamic.shape)
                # print(type(f0_without_dynamic))
                # print(f0_without_dynamic.shape)
                # print("feat :",feat.shape)
                #questionimport pdb;pdb.set_trace()

                tts.savebin(lf0_mlpg, os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)))
                tts.savebin(mgc_mlpg, os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)))
                tts.savebin(uv.float(), os.path.join(gen_dir_path, '{}.uv'.format(base)))
                tts.savebin(feat, os.path.join(gen_dir_path, '{}.feat'.format(base)))
                np.save(os.path.join(gen_dir_path, '{}'.format(base)) , feat.numpy())


                hdf5_filename = path.join(hdf5_path,base)
                #print("hdf5_filename",hdf5_filename)
                #pattern = r'[ab]\d{4}' # arctic
                pattern = r'\d+-\d+'
                result = re.search(pattern, hdf5_filename).group(0)
                #print("result",result)
                #hdf5_filename = "arctic_" + result  + ".h5"
                hdf5_filename = "lj_speech_" + result  + ".h5"
                hdf5_filename = os.path.join(hdf5_path,hdf5_filename)
                hdf5_filename_list.append(hdf5_filename)
                with h5py.File(hdf5_filename , 'w') as f :
                    f['world'] = feat
                    print("write {} finished".format(hdf5_filename))

    '''
    # dataset_num = 1132 ## 1132 # 1419
    # trn_num = 1028 # 1028 # 300
    dataset_num = 1419 ## 1132 # 1419
    trn_num = 300 # 1028 # 300
    ev_num = dataset_num - trn_num

    hdf5_filename_list.sort()
    #print(hdf5_filename_list)
    print("dataset_num :",dataset_num)
    print("trn_num :",trn_num)
    print("ev_num :",ev_num)
    print("hdf5_filename_list_num :", len(hdf5_filename_list))
    for i in range(0,dataset_num):
        shutil.copy(hdf5_filename_list[i] , trn_hdf5_path)
    for i in range(trn_num,dataset_num):
        shutil.copy(hdf5_filename_list[i] , ev_hdf5_path)
    '''

if args.target:
    b_dir_path = path.join("data","gen","set","b")
    stat_dir_path = path.join("data", "target", "stat")
    gen_dir_path = path.join("gen","target", "from_b")
    var_path = path.join(stat_dir_path, "c_trn_var.pt")
    dur_state_dict_path = path.join(model_dir_path, "target", "dur.state_dict")
    am_state_dict_path = path.join(model_dir_path, "target", "am.state_dict")

    device = th.device("cuda" if use_cuda else "cpu")
    variance = th.load(var_path)

    dur_model = tts.DurationModelV2( b_order, hidden_size).to(device)
    am_model = tts.AcousticModelV3( a_order, hidden_size, c_order, variance=variance).to(device)

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
                b = tts.load_bin( b_path, 'float32' ).view(-1,b_order).to(device)
                #d = dur_model(b).argmax(1, keepdim=False).view(-1, number_of_states)
                d = dur_model(b).round().int().view(-1, number_of_states)
                # 去掉 語速資訊
                b = b[:,0: b_order-1].view(-1, number_of_states, b_order-1)
                a = []

                for (phone_index, phone) in enumerate(d):
                    # shape = (number_of_states, b_order-1)
                    ans_of_cur_phone = b[phone_index]
                    total_frames_in_phone = phone.sum().item()
                    cur_frame_in_phone = 0

                    for (state_index, state) in enumerate(phone):
                        #(b_order-1)
                        ans_of_cur_state = ans_of_cur_phone[state_index]
                        total_frames_in_state = state.item()

                        for frame_index in range(0,total_frames_in_state):
                            ans_of_cur_frame = ans_of_cur_state.clone()

                            frame_in_state_fwd_n = (frame_index) / (frame_in_state_max-frame_in_state_min)
                            frame_in_state_bwd_n = (total_frames_in_state - frame_index - frame_in_state_min) / (frame_in_phone_max-frame_in_phone_min)
                            frame_in_phone_fwd_n = (cur_frame_in_phone) / (frame_in_phone_max - frame_in_phone_min)
                            frame_in_phone_bwd_n = (total_frames_in_phone - cur_frame_in_phone - frame_in_phone_min ) / (frame_in_phone_max-frame_in_phone_min)

                            frame_info = ans_of_cur_frame.new([frame_in_state_fwd_n, frame_in_state_bwd_n, frame_in_phone_fwd_n, frame_in_phone_bwd_n])

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

                tts.savebin(lf0_mlpg, os.path.join(gen_dir_path, '{}.lf0.mlpg'.format(base)))
                tts.savebin(mgc_mlpg, os.path.join(gen_dir_path, '{}.mgc.mlpg'.format(base)))
                tts.savebin(uv.float(), os.path.join(gen_dir_path, '{}.uv'.format(base)))
