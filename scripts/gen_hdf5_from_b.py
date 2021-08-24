import os
import sys
import h5py
import numpy as np
from os import path
import tts
import json
import argparse as ap
import torch as th
import torch.utils.data as data
import torch.optim as optim
import pdb
import shutil
import utils as ut
'''
SaveDirectory = os.getcwd()
print(SaveDirectory)
#h = ut.read_hdf5( "../tr_slt/arctic_a0001.h5", "/world")
#print(h)
#print(h.shape)


f1 = h5py.File( "../tr_slt/arctic_a0001.h5", "r")

f2 = h5py.File( "../tr_slt/stats.h5", "r")


for key in f1.keys():
	print(f1[key].name)
	print(f1[key].shape)
	#print(f[key].value)

for key in f2.keys():
	print(f2[key].name)
	#print(f2[key].shape)
	#print(f2[key].value)

print(f2)

group2 = f2.get('world/scale')

print(group2)

uv_fn = File.join(gen_dir_path, "#{base}.uv")
f0_mlpg_fn = File.join(gen_dir_path, "#{base}.lf0.mlpg")
mgc_mlpg_fn = File.join(gen_dir_path, "#{base}.mgc.mlpg")
'''
'''
with open(path.join("configs","IOConfigs.json")) as io_configs_file:
    io_configs = json.load(io_configs_file)

a_order = io_configs["a_order"]
b_order = io_configs["b_order"]
c_order = io_configs["c_order"]
d_order = io_configs["d_order"]
d_class = io_configs["d_class"]

SaveDirectory = os.getcwd()
print("curren directory : " , SaveDirectory)

source_dir_path = path.join("gen","source","from_a")
gen_dir_path = path.join("gen","source","hdf5")
print("source_dir_path :",source_dir_path)
print("gen_dir_path :",gen_dir_path)

if not path.isdir(source_dir_path):
	os.mkdir(source_dir_path)
if not path.isdir(gen_dir_path):
	os.mkdir(gen_dir_path)

lf0_mlpg = []
mgc_mlpg = []
uv = []

for root, dirs, files in os.walk(source_dir_path, topdown=False):
	for name in files:
		path = os.path.join(root, name)
		if name.endswith(".lf0.mlpg"):
			lf0_mlpg.append(path)
		if name.endswith(".mgc.mlpg"):
			mgc_mlpg.append(path)
		if name.endswith(".uv"):
			uv.append(path)

a = tts.load_bin( lf0_mlpg[0], 'float32').view(-1,a_order)
print(a.shape)
'''
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
number_of_states = configs["number_of_states"]

model_dir_path = path.join("model")

state_in_phone_min = configs["state_in_phone_min"]
state_in_phone_max = configs["state_in_phone_max"]
frame_in_state_min = configs["frame_in_state_min"]
frame_in_state_max = configs["frame_in_state_max"]
frame_in_phone_min = configs["frame_in_phone_min"]
frame_in_phone_max = configs["frame_in_phone_max"]

if args.source:
	a_dir_path = path.join("data","set","a")
	b_dir_path = path.join("data","set","b")
	stat_dir_path = path.join("data", "stat")
	gen_dir_path = path.join("gen", "hdf5_DNN")
	trn_hdf5_path = path.join(gen_dir_path,"tr_slt")
	ev_hdf5_path = path.join(gen_dir_path,"ev_slt")
	hdf5_path = path.join(gen_dir_path,"hdf5")
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


	if os.path.isdir(trn_hdf5_path) != True:
		os.makedirs(trn_hdf5_path)
	if os.path.isdir(ev_hdf5_path)  != True:
		os.makedirs(ev_hdf5_path)
	if os.path.isdir(hdf5_path)  != True:
		os.makedirs(hdf5_path)

	hdf5_filename_list = []
	with th.no_grad():
		var_lf0_mgc = variance[0:c_order-1].view(1,3, (lf0_order+mgc_order))
		for filename in sorted(os.listdir(a_dir_path)):
			if filename.endswith(".bin"):
				base = path.splitext(filename)[0]
				b_path = path.join(b_dir_path, "{}.bin".format(base))
				b = ut.load_binfile(b_path)
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
				feat = c_.cpu()

				print("feat : ",feat.shape)

				#import pdb; pdb.set_trace()

				hdf5_filename = path.join(hdf5_path,base) + ".h5"
				hdf5_filename_list.append(hdf5_filename)

				if os.path.isdir(gen_dir_path) != True:
					os.makedirs(gen_dir_path)
				#print(hdf5_filename)
				with h5py.File(hdf5_filename , 'w') as hdf :
					#G1 = hdf.create_group('world/lf0')
					#G1.create_dataset('mean' , data = lf0_m)
					#G1.create_dataset('variance' , data = lf0_v)
					#G2 = hdf.create_group('world/mgc')
					#G2.create_dataset('mean' , data = mgc_m)
					#G2.create_dataset('variance' , data = mgc_v)
					#G3 = hdf.create_dataset('world/uv' , data = uv)
					G4 = hdf.create_dataset('world' , data = feat)
					print("write {}.h5 finished".format(base))

				'''
				with h5py.File(hdf5_filename , 'r') as hdf :
					base_items= list(hdf.items())
					print("items in the base directory: ", base_items)
					lf0 = hdf.get('lf0')
					G2_items = list(lf0.items())
					print('Items in mean: ', G2_items)
					dataset3 = np.array(lf0.get('mean'))
					dataset4 = np.array(lf0.get('variance'))
					print(dataset3.shape)
					print(dataset4.shape)
				'''

	if os.path.isdir(trn_hdf5_path) != True:
		os.makedirs(trn_hdf5_path)
	if os.path.isdir(ev_hdf5_path)  != True:
		os.makedirs(ev_hdf5_path)

	dataset_num = 1132 ## 1132
	trn_num = 1028 # 1028
	ev_num = dataset_num - trn_num

	hdf5_filename_list.sort()
	#print(hdf5_filename_list)
	print("dataset_num :",dataset_num)
	print("trn_num :",trn_num)
	print("ev_num :",ev_num)

	for i in range(0,dataset_num):
		shutil.copy(hdf5_filename_list[i] , trn_hdf5_path)
	for i in range(trn_num,dataset_num):
		shutil.copy(hdf5_filename_list[i] , ev_hdf5_path)
