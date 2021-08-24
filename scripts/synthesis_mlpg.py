import json
import os
import sys
import subprocess
from subprocess import Popen, PIPE, run
import argparse as ap

with open(os.path.join("configs","Configs.json")) as configs_file:
    configs = json.load(configs_file)


sr = configs["sampling_rate"]
fs = int(configs["frame_shift"]*sr)
gm = configs["gamma"]
fw = configs["frequency_warping"]
fft = configs["fft_len"]
mgc_order = configs["mgc_order"]
source_dir_path = configs["gen_dir_path"]
gen_dir_path = configs["gen_dir_path"]
hts_dir_path = "/home/adamliu/Desktop/HTS-demo_CMU-ARCTIC-SLT_0803/gen/qst001/ver1/2mix/0/"

print("sr", sr ,type(sr))
print("fs", fs ,type(fs))
print("gm", gm ,type(gm))
print("fw", fw ,type(fw))
print("fft", fft ,type(fft))
print("mgc_order", mgc_order ,type(mgc_order))
print("gen_dir_path", gen_dir_path ,type(gen_dir_path))


if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument("--train_model", type=str)
    parser.add_argument("--train_type", type=str)
    args = parser.parse_args()

    if args.train_model == "acoustic":
        train_model = 0 #am
    elif args.train_model == "duration":
        train_model = 1
    if args.train_type == "DNN":
        train_type = 0 #dnn
    elif args.train_type == "LSTM":
        train_type = 1

    if not os.path.isdir(os.path.join(source_dir_path,"output")):
        os.mkdir(os.path.join(source_dir_path,"output"))
        os.mkdir(os.path.join(source_dir_path,"output","from_acoustic_DNN"))
        os.mkdir(os.path.join(source_dir_path,"output","from_duration_DNN"))
        os.mkdir(os.path.join(source_dir_path,"output","from_acoustic_LSTM"))
        os.mkdir(os.path.join(source_dir_path,"output","from_duration_LSTM"))
        os.mkdir(os.path.join(source_dir_path,"output","hts"))

    if train_model == 0 and train_type == 0:
        gen_dir_path = os.path.join(gen_dir_path,"from_acoustic_DNN")
        outpath = os.path.join(source_dir_path,"output","from_acoustic_DNN")
    elif train_model == 1 and train_type == 0:
        gen_dir_path = os.path.join(gen_dir_path,"from_duration_DNN")
        outpath = os.path.join(source_dir_path,"output","from_duration_DNN")
    elif train_model == 0 and train_type == 1:
        gen_dir_path = os.path.join(gen_dir_path,"from_acoustic_LSTM")
        outpath = os.path.join(source_dir_path,"output","from_acoustic_LSTM")
    elif train_model == 1 and train_type == 1:
        gen_dir_path = os.path.join(gen_dir_path,"from_duration_LSTM")
        outpath = os.path.join(source_dir_path,"output","from_duration_LSTM")


    for _,_ ,file in os.walk(os.path.join(gen_dir_path)):
        f0_filename = sorted([os.path.join(gen_dir_path,x) for x in file if x.endswith(".lf0.mlpg")])
        mgc_filename= sorted([os.path.join(gen_dir_path,x) for x in file if x.endswith(".mgc.mlpg")])
        uv_filename = sorted([os.path.join(gen_dir_path,x) for x in file if x.endswith(".uv")])

    for i in range(len(uv_filename)):
        base = os.path.basename(uv_filename[i]).strip(".uv")
        print(base)

        f0_mlpg_fn = f0_filename[i]
        mgc_mlpg_fn = mgc_filename[i]
        uv_fn = uv_filename[i]

        lf0_fn = os.path.join(gen_dir_path,'{}.lf0'.format(base))
        f0_fn = os.path.join(gen_dir_path,'{}.f0'.format(base))
        mgc_fn = os.path.join(gen_dir_path,'{}.mgc'.format(base))

        print("f0_mlpg_fn :",f0_mlpg_fn)
        print("mgc_mlpg_fn :",mgc_mlpg_fn)
        print("uv_fn :",uv_fn)
        print("lf0_fn :",lf0_fn)
        print("f0_fn :",f0_fn)
        print("mgc_fn :",mgc_fn)


        # apply mlpg to f0
        line = "mlpg -m 0 -d -0.5 0 0.5 -d 1 -2 1 0 0 %s | sopr -EXP -INV -m %d | vopr -m %s > %s" \
                % (f0_mlpg_fn, sr, uv_fn, f0_fn)
        subprocess.check_output(line, shell=True)


        line = "mlpg -m 0 -d -0.5 0 0.5 -d 1 -2 1 0 0 %s | sopr -EXP -INV -m %d | vopr -m %s > %s " \
                %(f0_mlpg_fn, sr, uv_fn, lf0_fn)
        subprocess.check_output(line, shell=True)


        line = "mlpg -m %d -d -0.5 0 0.5 -d 1 -2 1 0 0 %s > %s" \
                % (mgc_order-1, mgc_mlpg_fn, mgc_fn)
        subprocess.check_output(line, shell=True)


        rawpath = os.path.join(gen_dir_path,'{}.raw'.format(base))
        line = 'excite -n -p %d %s | mglsadf -P 5 -m %d -p %d -a %f -c %d %s | x2x +fs -o > %s' \
                %(fs, f0_fn, mgc_order-1, fs, fw, gm, mgc_fn, rawpath)
        subprocess.check_output(line, shell=True)


        sr_t = int(sr/1000)
        line = 'raw2wav -s %d -d %s %s' \
                % (sr_t, outpath, rawpath)

        subprocess.check_output(line, shell=True)

    '''
    ## HTS
    for _,_ ,file in os.walk(os.path.join(hts_dir_path)):
        #lf0_filename = sorted([os.path.join(hts_dir_path,x) for x in file if x.endswith(".lf0")])
        mgc_filename= sorted([os.path.join(hts_dir_path,x) for x in file if x.endswith(".mgc")])
        f0_filename = sorted([os.path.join(hts_dir_path,x) for x in file if x.endswith(".pit")])

    for i in range(len(f0_filename)):
        base = os.path.basename(f0_filename[i]).strip(".pit")
        print(base)

        f0_fn = f0_filename[i]
        mgc_fn = mgc_filename[i]



        rawpath = os.path.join(hts_dir_path,'{}.raw'.format(base))
        line = 'excite -n -p %d %s | mglsadf -P 5 -m %d -p %d -a %f -c %d %s | x2x +fs -o > %s' \
                %(fs, f0_fn, mgc_order-1, fs, fw, gm, mgc_fn, rawpath)
        subprocess.check_output(line, shell=True)

        outpath = "/home/adamliu/Desktop/HTS-demo_CMU-ARCTIC-SLT_0803/gen/qst001/ver1/2mix/out"
        sr_t = int(sr/1000)
        line = 'raw2wav -s %d -d %s %s' \
                % (sr_t, outpath, rawpath)

        subprocess.check_output(line, shell=True)
    '''
