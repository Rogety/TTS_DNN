
import os
from os import path
import json
import utils as ut
import numpy as np


with open(path.join("configs","Configs.json")) as configs_file:
    configs = json.load(configs_file)

source_dir_path = configs["source_dir_path"]


def interpolation(lf0):
    lf0 = list(lf0).copy()
    output = []
    index = []
    edge_index = []
    for idx , val in enumerate(lf0) :
        if val != -(10**10):
            output.append(1.0)
        else:
            output.append(0.0)
            index.append(idx)
    #print("uv :",output,len(output))
    #print(index)
    for i in range(len(index)-1):
        if (index[i+1] - index[i]) != 1:
            edge_index.append(index[i])
            edge_index.append(index[i+1])

    #print("lf0 :",lf0,len(lf0))
    ## 內插 lf0
    for i in range(len(edge_index)):
        if i == 0 : ## first
            for j in range(edge_index[i]+1):
                lf0[j] = lf0[edge_index[0]+1]
        elif i == len(edge_index)-1 : ## last
            for j in range(edge_index[i] , len(lf0)):
                lf0[j] = lf0[edge_index[-1]-1]
        elif (i % 2) == 1: ## middle
            #print(i , edge_index[i] , edge_index[i+1])
            for j in range(edge_index[i] , edge_index[i+1]+1):
                ratio = (lf0[edge_index[i+1]+1]-lf0[edge_index[i]-1]) \
                / (edge_index[i+1]-edge_index[i]+1)
                lf0[j] = lf0[edge_index[i]-1] +(j-edge_index[i]+1)*ratio
        else:
            pass
    #print("lf0 :",lf0,len(lf0))

    lf0_delta = []
    for i in range(len(lf0)):
        if i == 0:
            lf0_delta.append(0.5 * lf0[i+1])
        elif i == (len(lf0)-1):
            lf0_delta.append(-0.5 * lf0[i-1])
        else:
            lf0_delta.append(0.5*lf0[i+1] - 0.5*lf0[i-1])

    #print("lf0_delta :",lf0_delta,len(lf0_delta))

    lf0_delta2 = []
    for i in range(len(lf0)):
        if i == 0 :
            lf0_delta2.append(-2.0*lf0[i] + lf0[i+1])
        elif i == (len(lf0)-1) :
            lf0_delta2.append(-2.0*lf0[i] + lf0[i-1])
        else :
            lf0_delta2.append(lf0[i-1] -2.0*lf0[i] + lf0[i+1])

    #print("lf0_delta2 :",lf0_delta2,len(lf0_delta2))
    #print(lf0)
    return output , lf0 , lf0_delta , lf0_delta2

def get_mgc_delta(mgc,lf0):

    length = len(lf0)
    mgc = list(mgc).copy()
    mgc = np.reshape(mgc,(-1,25)) ## (671 , 25)

    #print("mgc :",mgc.shape)
    mgc_delta = []
    for i in range(len(mgc)):
        mgc_delta.append([])
        if i == 0:
            for j in range(len(mgc[0])):
                mgc_delta[i].append(0.5 * mgc[i+1][j])
        elif i == (len(lf0)-1):
            for j in range(len(mgc[0])):
                mgc_delta[i].append(-0.5 * mgc[i-1][j])
        else:
            for j in range(len(mgc[0])):
                mgc_delta[i].append(0.5*mgc[i+1][j] - 0.5*mgc[i-1][j])
    #print("mgc_delta :",len(mgc_delta),len(mgc_delta[0]))

    mgc_delta2 = []
    for i in range(len(mgc)):
        mgc_delta2.append([])
        if i == 0:
            for j in range(len(mgc[0])):
                mgc_delta2[i].append(-2.0*mgc[i][j] + mgc[i+1][j] )
        elif i == (len(lf0)-1):
            for j in range(len(mgc[0])):
                mgc_delta2[i].append(-2.0*mgc[i][j] + mgc[i-1][j])
        else:
            for j in range(len(mgc[0])):
                mgc_delta2[i].append(mgc[i-1][j] - 2.0*mgc[i][j] + mgc[i+1][j])
    #print("mgc_delta2 :",len(mgc_delta2),len(mgc_delta2[0]))

    return mgc , mgc_delta , mgc_delta2


if __name__ == '__main__':
    lf0_dir_path = path.join(source_dir_path, "lf0")
    mgc_dir_path = path.join(source_dir_path, "mgc")
    c_dir_path = path.join(source_dir_path, "set", "c")

    io_config_path = os.path.join("configs","IOConfigs.json")
    with open(io_config_path,'r') as fin:
        load_dict = json.load(fin)
    load_dict["c_order"]=79
    with open(io_config_path,'w') as fin:
        json.dump(load_dict, fin, indent ="")

    for _ , _ ,filename in os.walk(lf0_dir_path):
        lf0paths = sorted([os.path.join(lf0_dir_path,x) for x in filename])
    for _ , _ ,filename in os.walk(mgc_dir_path):
        mgcpaths = sorted([os.path.join(mgc_dir_path,x) for x in filename])

    #lf0paths = [os.path.join(lf0_dir_path,"cmu_us_arctic_slt_a0001.lf0") ]
    #mgcpaths = [os.path.join(mgc_dir_path,"cmu_us_arctic_slt_a0001.mgc") ]

    for i in range(len(lf0paths)):
        print(lf0paths[i])
        print(mgcpaths[i])
        base = os.path.basename(lf0paths[i]).replace(".lf0",".bin")
        c_bin_path = path.join(c_dir_path , base)

        lf0 = ut.load_binfile(lf0paths[i])

        mgc = ut.load_binfile(mgcpaths[i])


        if len(lf0) != len(mgc)/25:
            print("lf0 :" , len(lf0))
            print("mgc :" , len(mgc))
            print("error : f0 and mgc not match ,",lf0paths[i])

        voiced,lf0,lf0_delta,lf0_delta2 = interpolation(lf0)
        mgc , mgc_delta , mgc_delta2 = get_mgc_delta(mgc,lf0) ## 內插且get uv

        cmp = []
        for i in range(len(lf0)):
            cmp.append([])

            cmp[i].append(lf0[i])
            for j in range(len(mgc[0])):
                cmp[i].append(mgc[i][j])

            cmp[i].append(lf0_delta[i])
            for j in range(len(mgc_delta[0])):
                cmp[i].append(mgc_delta[i][j])
            cmp[i].append(lf0_delta2[i])
            for j in range(len(mgc_delta2[0])):
                cmp[i].append(mgc_delta2[i][j])
            cmp[i].append(voiced[i])

        cmp_flattern = [y for x in cmp for y in x]

        #print("cmp_flattern :",len(cmp_flattern))


        ut.save_binfile( c_bin_path , cmp_flattern)
