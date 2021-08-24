
import os
import json
import utils as ut
from os import path
import subprocess


with open(path.join("configs","Configs.json")) as configs_file:
    configs = json.load(configs_file)
source_dir_path = configs["source_dir_path"]


def split_dataset( filepaths ):

    trn_list = []
    tst_list = []
    val_list = []

    for path in filepaths :
        if path.endswith("1.bin") :
            val_list.append(path)
        elif path.endswith("2.bin") :
            tst_list.append(path)
        else :
            trn_list.append(path)

    return trn_list , tst_list , val_list


if __name__ == '__main__':

    set_dir_path = os.path.join("data","set")

    a_trn_list , a_tst_list , a_val_list = [],[],[]
    b_trn_list , b_tst_list , b_val_list = [],[],[]
    c_trn_list , c_tst_list , c_val_list = [],[],[]
    d_trn_list , d_tst_list , d_val_list = [],[],[]

    if not path.isdir(os.path.join(source_dir_path,"set")):
        os.mkdir(os.path.join(source_dir_path,"set"))
        os.mkdir(os.path.join(source_dir_path,"set","a"))
        os.mkdir(os.path.join(source_dir_path,"set","b"))
        os.mkdir(os.path.join(source_dir_path,"set","c"))
        os.mkdir(os.path.join(source_dir_path,"set","d"))
        os.mkdir(os.path.join(source_dir_path,"set","p"))

    for dir , dirname , filename in os.walk(set_dir_path):
        if dir == "data/set/a" :
            a_filepaths = sorted([os.path.join(dir,x) for x in filename])
        if dir == "data/set/b" :
            b_filepaths = sorted([os.path.join(dir,x) for x in filename])
        if dir == "data/set/c" :
            c_filepaths = sorted([os.path.join(dir,x) for x in filename])
        if dir == "data/set/d" :
            d_filepaths = sorted([os.path.join(dir,x) for x in filename])

    a_trn_list , a_tst_list , a_val_list = split_dataset(a_filepaths)
    b_trn_list , b_tst_list , b_val_list = split_dataset(b_filepaths)
    c_trn_list , c_tst_list , c_val_list = split_dataset(c_filepaths)
    d_trn_list , d_tst_list , d_val_list = split_dataset(d_filepaths)
    print("a :" , len(a_trn_list),len(a_tst_list),len(a_val_list))
    print("b :" , len(b_trn_list),len(b_tst_list),len(b_val_list))
    print("c :" , len(c_trn_list),len(c_tst_list),len(c_val_list))
    print("d :" , len(d_trn_list),len(d_tst_list),len(d_val_list))



    ut.combine_binfile( a_trn_list , "a_trn.bin" )
    ut.combine_binfile( a_tst_list , "a_tst.bin" )
    ut.combine_binfile( a_val_list , "a_val.bin" )
    ut.combine_binfile( b_trn_list , "b_trn.bin" )
    ut.combine_binfile( b_tst_list , "b_tst.bin" )
    ut.combine_binfile( b_val_list , "b_val.bin" )
    ut.combine_binfile( c_trn_list , "c_trn.bin" )
    ut.combine_binfile( c_tst_list , "c_tst.bin" )
    ut.combine_binfile( c_val_list , "c_val.bin" )
    ut.combine_binfile( d_trn_list , "d_trn.bin" )
    ut.combine_binfile( d_tst_list , "d_tst.bin" )
    ut.combine_binfile( d_val_list , "d_val.bin" )
    
    if not path.isdir(os.path.join(source_dir_path,"set","lstm_a")):
        os.mkdir(os.path.join(source_dir_path,"set","lstm_a"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_a","trn"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_a","tst"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_a","val"))
    if not path.isdir(os.path.join(source_dir_path,"set","lstm_b")):
        os.mkdir(os.path.join(source_dir_path,"set","lstm_b"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_b","trn"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_b","tst"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_b","val"))
    if not path.isdir(os.path.join(source_dir_path,"set","lstm_c")):
        os.mkdir(os.path.join(source_dir_path,"set","lstm_c"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_c","trn"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_c","tst"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_c","val"))
    if not path.isdir(os.path.join(source_dir_path,"set","lstm_d")):
        os.mkdir(os.path.join(source_dir_path,"set","lstm_d"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_d","trn"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_d","tst"))
        os.mkdir(os.path.join(source_dir_path,"set","lstm_d","val"))


    for i in range(len(a_trn_list)):
        inputpath = a_trn_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_a","trn")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_a/trn success")
    for i in range(len(a_tst_list)):
        inputpath = a_tst_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_a","tst")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_a/tst success")
    for i in range(len(a_val_list)):
        inputpath = a_val_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_a","val")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_a/val success")

    for i in range(len(b_trn_list)):
        inputpath = b_trn_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_b","trn")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_b/trn success")
    for i in range(len(b_tst_list)):
        inputpath = b_tst_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_b","tst")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_b/tst success")
    for i in range(len(b_val_list)):
        inputpath = b_val_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_b","val")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_b/val success")

    for i in range(len(c_trn_list)):
        inputpath = c_trn_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_c","trn")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_c/trn success")
    for i in range(len(c_tst_list)):
        inputpath = c_tst_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_c","tst")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_c/tst success")
    for i in range(len(c_val_list)):
        inputpath = c_val_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_c","val")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_c/val success")

    for i in range(len(d_trn_list)):
        inputpath = d_trn_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_d","trn")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_d/trn success")
    for i in range(len(d_tst_list)):
        inputpath = d_tst_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_d","tst")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_d/tst success")
    for i in range(len(d_val_list)):
        inputpath = d_val_list[i]
        outputpath = os.path.join(source_dir_path,"set","lstm_d","val")
        line = "cp %s %s " \
            % (inputpath, outputpath)
        subprocess.check_output(line, shell=True)
    print("cp lstm_d/val success")
