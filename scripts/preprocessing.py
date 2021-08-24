import json
import os
from os import path
import re
import copy
import struct
import numpy as np
import torch as th
import argparse as ap


with open(path.join("configs","Configs.json")) as configs_file:
    configs = json.load(configs_file)

questions_set = []
with open(path.join("data","questions","questions_qst001.conf")) as questions:
    for line in questions :
        questions_set.append(line)

source_dir_path = configs["source_dir_path"]
frame_shift = configs["frame_shift"]*(10**7)

def slice_label(lab_phone):

    lab_p = []
    cnt = -1
    for i in range(len(lab_phone)):
        if i % 5 ==0 :
            lab_p.append([])
            cnt += 1
        lab_p[cnt].append(lab_phone[i])

    return lab_p

def get_lab_info(lab,flag):

    pattern = r'[0-9]+'
    if flag == "phone":
        phone_start = re.findall(pattern , lab[0])[0]
        phone_end = re.findall(pattern , lab[4] )[1]
        phone_str = lab[0].split(" ")[3].rstrip("\n")
        frames_in_phone = (float(phone_end) - float(phone_start))/frame_shift
        return frames_in_phone , phone_str
    elif flag == "state":
        frames_in_state = []
        for i in range(5):
            state_start = re.findall(pattern , lab[i])[0]
            state_end = re.findall(pattern , lab[i])[1]
            frames_in_state.append((float(state_end) - float(state_start))/frame_shift)
        phone_str = lab[0].split(" ")[3].rstrip("\n")

        return frames_in_state , phone_str
    else :
        print("error_get_lab_info")
        return 0


def qask(pstr , questions_set , flag , frames_in_state=0 , frames_in_phone=0):

    ## p
    pattern_dict = {"LLphone": r'\w+\^',"Lphone":r'\^\w+\-',"Cphone":r'\-\w+\+',"Rphone":r'\+\w+\=',"RRphone":r'\=\w+\@',
    "ph_in_syl_fw":r'\@\w+\_',"ph_in_syl_bw":r'\_\w+\/A\:', ## p
    "Lstress":r'\@\w+\_',"phnum_in_presyl":r'\_\w+\/A\:', ## a
    "Cstress":r'\/B\:\w+\-',"phnum_in_cursyl":r'\-\w+\-',"syl_in_word_fw":r'\-\w+\@',"syl_in_word_bw":r'\@\w+\-',"syl_in_phrase_fw":r'\-\w+\&',
    "syl_in_phrase_bw":r'\&\w+\-',"b7":r'\-\w+\#',"b8":r'\#\w+\-',"b9":r'\-\w+\$',"b10":r'\$\w+\-',"vowel_in_syl":r'\-\w+\/C\:', ## b
    "Rstress":r'\/C\:\w+\+',"phnum_in_nextsyl":r'\+\w+\/D\:', ## c
    "gpos_in_preword":r'\/D\:\w+\_',"sylnum_in_preword":r'\_\w+\/E\:', ## d
    "gpos_in_curword":r'\/E\:\w+\+',"sylnum_in_curword":r'\+\w+\@',"word_in_phrase_fw":r'\@\w+\+',"word_in_phrase_bw":r'\+\w+\&',
    "e5":r'\&\w+\+',"e6":r'\+\w+\#',"e7":r'\#\w+\+',"e8":r'\+\w+\/F\:', ## e
    "gpos_in_next_word":r'\/F\:\w+\_',"wordnum_in_prephrase":r'\_\w+\/G\:', ## f
    "sylnum_in_prephrase":r'\/G\:\w+\_',"wordnum_in_curphrase":r'\_\w+\/H\:', ## g
    "sylnum_in_curphrase":r'\/H\:\w+\=',"wordnum_in_prephrase":r'\=\w+\^',"phrase_in_sentense_fw":r'\^\w+\=',"phrase_in_sentense_bw":r'\=\w+\/I\:', ## h
    "sylnum_in_nextphrase":r'\/I\:\w+\_',"wordnum_in_nextphrase":r'\_\w+\/J\:', ## i
    "sylnum_in_sentense":r'\/J\:\w+\+',"wordnum_in_sentense":r'\+\w+\-',"phrasenum_in_sentense":r'\-\w+', ## j
    }
    #print("pattern_dict",pattern_dict)

    lab_dict = {}
    #lab_list = [p1,p2,p3,p4,p5,p6,p7]
    linquistic_list = ["LLphone","Lphone","Cphone","Rphone","RRphone","ph_in_syl_fw","ph_in_syl_bw", ## p
    "Lstress","phnum_in_presyl", ## a
    "Cstress","phnum_in_cursyl","syl_in_word_fw","syl_in_word_bw","syl_in_phrase_fw","syl_in_phrase_bw","b7","b8","b9","b10","vowel_in_syl", ## b
    "Rstress","phnum_in_nextsyl", ## c
    "gpos_in_preword","sylnum_in_preword", ## d
    "gpos_in_curword","sylnum_in_curword","word_in_phrase_fw","word_in_phrase_bw","e5","e6","e7","e8", ## e
    "gpos_in_next_word","wordnum_in_prephrase", ## f
    "sylnum_in_prephrase","wordnum_in_curphrase", ## g
    "sylnum_in_curphrase","wordnum_in_prephrase","phrase_in_sentense_fw","phrase_in_sentense_bw", ## h
    "sylnum_in_nextphrase","wordnum_in_nextphrase", ## i
    "sylnum_in_sentense","wordnum_in_sentense","phrasenum_in_sentense", ## j
    ]
    for i in range(len(linquistic_list)):
        #print(pattern_dict[linquistic_list[i]] , pstr)
        if linquistic_list[i] == "LLphone":
            lab_dict[linquistic_list[i]] = re.search( pattern_dict[linquistic_list[i]], pstr).group(0) + "*"
        elif linquistic_list[i] == "phrasenum_in_sentense":
            lab_dict[linquistic_list[i]] = "*" + re.search( pattern_dict[linquistic_list[i]], pstr).group(0)
        else:
            try :
                lab_dict[linquistic_list[i]] = "*" + re.search( pattern_dict[linquistic_list[i]], pstr).group(0) + "*"
            except :
                lab_dict[linquistic_list[i]] = "none"
                #print("error")
    #print(lab_dict)

    #print("lab_dict : ", lab_dict)

    ## 計算 questionset match 的 index
    qes_list = []
    for i in range(len(lab_dict)) :
        for index , qes in enumerate(questions_set) :
            if lab_dict[linquistic_list[i]] in qes :
                qes_list.append(index)

    ## generate phone answer
    ## phone
    ans = []
    for i in range(len(questions_set)):
        if i in qes_list:
            ans.append(1.0)
        else:
            ans.append(0.0)


    if flag == "state":
        ans_state = []
        fw = [0.0,0.25,0.5,0.75,1.0]
        for i in range(5):
            tmp = ans.copy()
            tmp.extend([fw[i],fw[4-i]])
            ans_state.append(tmp)
        #print("state" , len(ans_state) , len(ans_state[0]))
        return ans_state

    elif flag == "frame":
        #print("frames_in_state : ", frames_in_state)
        #print("frames_in_phone : ", frames_in_phone)
        ans_frame = []
        fw = [0.0,0.25,0.5,0.75,1.0]
        bw = [1.0,0.75,0.5,0.25,0.0]
        state_in_phone_fw , state_in_phone_bw= [],[]
        frame_in_state_fw_2d , frame_in_state_bw_2d= [],[]
        frame_in_state_fw , frame_in_state_bw= [],[]
        frame_in_phone_fw , frame_in_phone_bw= [],[]

        for index , item in enumerate(frames_in_state):
            for i in range(int(item)):
                state_in_phone_fw.append(fw[index])
                state_in_phone_bw.append(bw[index])
        #print(state_in_phone_fw , len(state_in_phone_fw))
        #print(state_in_phone_bw , len(state_in_phone_bw))
        ## frame_in_state_fw_2d
        for i in range(5):
            frame_in_state_fw_2d.append([])
            frame_in_state_bw_2d.append([])
            if frames_in_state[i] == 1:
                frame_in_state_fw_2d[i].append(0.0)
                frame_in_state_bw_2d[i].append(1.0)
            else:
                ratio = 1 / (frames_in_state[i] - 1)
                for j in range(int(frames_in_state[i])):
                    frame_in_state_fw_2d[i].append(float(j*ratio))
                    frame_in_state_bw_2d[i].append(1.0-float(j*ratio))
        #print(frame_in_state_fw_2d , len(frame_in_state_fw_2d) )
        ## flattern
        for items in frame_in_state_fw_2d:
            for item in items :
                frame_in_state_fw.append(item)
        for items in frame_in_state_bw_2d:
            for item in items :
                frame_in_state_bw.append(item)
        #print(frame_in_state_fw , len(frame_in_state_fw))
        #print(frame_in_state_bw , len(frame_in_state_bw))
        ## frame_in_phone_fw
        if frame_in_phone_fw == 1:
            frame_in_phone_fw.append(0.0)
        else:
            ratio = 1 / (frames_in_phone - 1)
            for j in range(int(frames_in_phone)):
                frame_in_phone_fw.append(float(j*ratio))
        #print(frame_in_phone_fw , len(frame_in_phone_fw))
        for i in range(int(frames_in_phone)):
            tmp = ans.copy()
            tmp.extend([state_in_phone_fw[i],state_in_phone_bw[i]])
            tmp.extend([frame_in_state_fw[i],frame_in_state_bw[i]])
            tmp.extend([frame_in_phone_fw[i],frame_in_phone_fw[int(frames_in_phone)-1-i]])
            ans_frame.append(tmp)
        #print(ans_frame)
        #print(len(ans_frame) , len(ans_frame[0]))
        return ans_frame
        #print("frame" , state_index , frame_in_state)
    else :
        return ans


def mkdir():
    if not path.isdir(path.join(source_dir_path,"stat")):
        os.mkdir(path.join(source_dir_path,"stat"))
    if not path.isdir(path.join(source_dir_path,"set")):
        os.mkdir(path.join(source_dir_path,"set"))
        os.mkdir(path.join(source_dir_path,"set","a"))
        os.mkdir(path.join(source_dir_path,"set","b"))
        os.mkdir(path.join(source_dir_path,"set","c"))
        os.mkdir(path.join(source_dir_path,"set","d"))
    if not path.isdir("model"):
        os.mkdir("model")

    return 0

def save_binfile(path , data):

    buffer = struct.pack( "f" * len(data) , *data )
    #print(struct.calcsize("f"))
    with open(path , mode="wb" ) as f :
        f.write(buffer)

    return 0

def load_binfile(path):

    with open(path , mode="rb" ) as f :
        tmp = f.read()
    buffer = struct.unpack( "f" * (len(tmp) // 4) , tmp )

    return buffer



def mkdab():
    lab_state_dir = path.join(source_dir_path, "lab", "state")
    a_dir_path = path.join(source_dir_path, "set", "a")
    b_dir_path = path.join(source_dir_path, "set", "b")
    d_dir_path = path.join(source_dir_path, "set", "d")

    for _ , _ ,filename in os.walk(lab_state_dir):
        labpaths = sorted([os.path.join(lab_state_dir,x) for x in filename])

    labpaths = [os.path.join(lab_state_dir,"cmu_us_arctic_slt_a0004.lab")]
    ## debug
    '''
    labpaths = [os.path.join(lab_state_dir,"cmu_us_arctic_slt_a0101.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0014.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0180.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0398.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0405.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0420.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0460.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0521.lab"), \
    os.path.join(lab_state_dir,"cmu_us_arctic_slt_b0523.lab"), \
    ]
    '''

    for lab_path in labpaths:
        print(lab_path)
        lab_state = []
        with open(lab_path) as lab_file:
            for line in lab_file :
                #line = line.strip("")
                lab_state.append(line)
        ## phone ans
        total_frames_in_phone = []
        total_frames_in_state = []
        ans_of_phone = []
        ans_of_state = []
        ans_of_frame = []
        lab_phone = slice_label(lab_state)
        #print(lab_phone[0])
        for index , lab in enumerate(lab_phone):
            #print(index , lab , len(lab) , len(lab_phone))
            frames_in_phone,pstr = get_lab_info(lab,"phone")
            print(frames_in_phone , pstr)
            frames_in_state,pstr = get_lab_info(lab,"state")
            print(frames_in_state , pstr)
            total_frames_in_phone.append(frames_in_phone)
            total_frames_in_state.append(frames_in_state)
            print("phone_string :",pstr)
            ## in:lab , out:ans
            ans = qask(pstr , questions_set,"phone")
            ans_of_phone.append(ans)
            #print(ans_of_phone)
            ans = qask(pstr , questions_set,"state")
            for item in ans:
                ans_of_state.append(item)
            #print(ans_of_state)
            ans = qask(pstr , questions_set,"frame" , frames_in_state , frames_in_phone)
            for item in ans:
                ans_of_frame.append(item)

        base = os.path.basename(lab_path).rstrip(".lab") + ".bin"
        a_bin_path = path.join(a_dir_path , base)
        b_bin_path = path.join(b_dir_path , base)
        d_bin_path = path.join(d_dir_path , base)
        #print(a_bin_path)

        ans_of_phone_flattern = [y for x in ans_of_phone for y in x]
        ans_of_state_flattern = [y for x in ans_of_state for y in x]
        ans_of_frame_flattern = [y for x in ans_of_frame for y in x]
        total_frames_in_state_flattern = [y for x in total_frames_in_state for y in x]
        #print("total_frames_in_state :", len(total_frames_in_state_flattern))
        #print("total_frames_in_state :", total_frames_in_state_flattern)
        #print(len(ans_of_phone_flattern))
        #print(len(ans_of_state_flattern))
        #print(len(ans_of_frame_flattern))
        #print(ans_of_phone_flattern)

        save_binfile(a_bin_path , ans_of_frame_flattern)
        save_binfile(b_bin_path , ans_of_state_flattern)
        save_binfile(d_bin_path , total_frames_in_state_flattern)


        #print(len(ans_of_phone) ,len(ans_of_phone[0]))
        #print(len(ans_of_state) ,len(ans_of_state[0]))
        #print(len(ans_of_frame) ,len(ans_of_frame[0]))

    io_config_path = os.path.join("configs","IOConfigs.json")
    if not os.path.isfile(io_config_path) :
        json_file = json.dumps({"a_order":1386,"b_order":1382,"d_order":1,"d_class":91,"c_order":79}, \
        sort_keys=True,indent=2,separators=(',',':'))
        with open(io_config_path, mode = "w+") as IOconfigs_file:
            json.dump(json_file, IOconfigs_file)
    else :
        pass

    with open(io_config_path,'r') as load_f:
        load_dict = json.load(load_f)
    print("load_dict : ",load_dict)


    return 0

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
    #print(index)
    for i in range(len(index)-1):
        if (index[i+1] - index[i]) != 1:
            edge_index.append(index[i])
            edge_index.append(index[i+1])
    #print(edge_index)
    ## 內插 lf0
    for i in range(len(edge_index)):
        if i == 0 : ## first
            for j in range(edge_index[i]):
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

    lf0_delta = []
    for i in range(len(lf0)):
        if i == 0:
            lf0_delta.append(0.5 * lf0[i+1])
        elif i == (len(lf0)-1):
            lf0_delta.append(-0.5 * lf0[i-1])
        else:
            lf0_delta.append(0.5*lf0[i+1] - 0.5*lf0[i-1])

    lf0_delta2 = []
    for i in range(len(lf0)):
        if i == 0 :
            lf0_delta2.append(-2.0*lf0[i] + lf0[i+1])
        elif i == (len(lf0)-1) :
            lf0_delta2.append(-2.0*lf0[i] + lf0[i-1])
        else :
            lf0_delta2.append(lf0[i-1] -2.0*lf0[i] + lf0[i+1])

    #print(lf0)
    return output , lf0 , lf0_delta , lf0_delta2

def get_mgc_delta(mgc,lf0):

    length = len(lf0)
    mgc = list(mgc).copy()
    mgc = np.reshape(mgc,(-1,25)) ## (671 , 25)

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


    return mgc , mgc_delta , mgc_delta2

def mkc():
    lf0_dir_path = path.join(source_dir_path, "lf0")
    mgc_dir_path = path.join(source_dir_path, "mgc")
    c_dir_path = path.join(source_dir_path, "set", "c")

    for _ , _ ,filename in os.walk(lf0_dir_path):
        lf0paths = sorted([os.path.join(lf0_dir_path,x) for x in filename])
    for _ , _ ,filename in os.walk(mgc_dir_path):
        mgcpaths = sorted([os.path.join(mgc_dir_path,x) for x in filename])

    for i in range(len(lf0paths)):
        print(lf0paths[i])
        base = os.path.basename(lf0paths[i]).replace(".lf0",".bin")
        c_bin_path = path.join(c_dir_path , base)

        lf0 = load_binfile(lf0paths[i])
        print("lf0 :" , len(lf0))
        mgc = load_binfile(mgcpaths[i])
        print("mgc :" , len(mgc))
        voiced,lf0,lf0_delta,lf0_delta2 = interpolation(lf0)
        mgc , mgc_delta , mgc_delta2 = get_mgc_delta(mgc,lf0) ## 內插且get uv

        cmp = []
        for i in range(len(lf0)):
            cmp.append([])
            cmp[i].append(voiced[i])
            cmp[i].append(lf0[i])
            cmp[i].append(lf0_delta[i])
            cmp[i].append(lf0_delta2[i])
            for j in range(len(mgc[0])):
                cmp[i].append(mgc[i][j])
            for j in range(len(mgc_delta[0])):
                cmp[i].append(mgc_delta[i][j])
            for j in range(len(mgc_delta2[0])):
                cmp[i].append(mgc_delta2[i][j])

        cmp_flattern = [y for x in cmp for y in x]

        #print(cmp[0])
        #print(len(cmp))
        #print(len(cmp[0]))
        ## 計算 lf0_dalta
        #print(lf0_delta)
        #print(lf0)
        #print(lf0_delta2)
        #print(voiced)
        #print("voiced : ",len(voiced))
        #print("lf0 : ",len(lf0))
        #print("mgc : ",len(mgc))
        #print(len(cmp_flattern))
        save_binfile( c_bin_path , cmp_flattern)
    return 0

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

def combine_binfile( path_list , filename ):

    bin_file = os.path.join("data","set",filename)
    if not os.path.isfile(bin_file):
        for path in path_list:
            print(path)
            data = load_binfile(path)
            buffer = struct.pack( "f" * len(data) , *data )
            with open(bin_file , "ab") as f :
                f.write(buffer)
    else :
        print("data need to be clean")

    print("save : " , filename , "success")

def mkset():

    set_dir_path = os.path.join("data","set")

    a_trn_list , a_tst_list , a_val_list = [],[],[]
    b_trn_list , b_tst_list , b_val_list = [],[],[]
    c_trn_list , c_tst_list , c_val_list = [],[],[]
    d_trn_list , d_tst_list , d_val_list = [],[],[]


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

    combine_binfile( a_trn_list , "a_trn.bin" )
    combine_binfile( a_tst_list , "a_tst.bin" )
    combine_binfile( a_val_list , "a_val.bin" )
    combine_binfile( b_trn_list , "b_trn.bin" )
    combine_binfile( b_tst_list , "b_tst.bin" )
    combine_binfile( b_val_list , "b_val.bin" )
    combine_binfile( c_trn_list , "c_trn.bin" )
    combine_binfile( c_tst_list , "c_tst.bin" )
    combine_binfile( c_val_list , "c_val.bin" )
    combine_binfile( d_trn_list , "d_trn.bin" )
    combine_binfile( d_tst_list , "d_tst.bin" )
    combine_binfile( d_val_list , "d_val.bin" )

    return 0

def mkstat():

    a_order = 1386
    b_order = 1382
    c_order = 79
    d_order = 1
    d_class = 91

    set_dir_path = path.join("data","set")
    stat_dir_path = path.join("data","stat")

    a_trn_path = path.join(set_dir_path,"a_trn.bin")
    a_val_path = path.join(set_dir_path,"a_val.bin")
    a_tst_path = path.join(set_dir_path,"a_tst.bin")

    b_trn_path = path.join(set_dir_path,"b_trn.bin")
    b_val_path = path.join(set_dir_path,"b_val.bin")
    b_tst_path = path.join(set_dir_path,"b_tst.bin")

    c_trn_path = path.join(set_dir_path,"c_trn.bin")
    c_val_path = path.join(set_dir_path,"c_val.bin")
    c_tst_path = path.join(set_dir_path,"c_tst.bin")

    d_trn_path = path.join(set_dir_path,"d_trn.bin")
    d_val_path = path.join(set_dir_path,"d_val.bin")
    d_tst_path = path.join(set_dir_path,"d_tst.bin")

    a_trn_len = int(os.stat( a_trn_path ).st_size / 4 / a_order)
    a_val_len = int(os.stat( a_val_path ).st_size / 4 / a_order)
    a_tst_len = int(os.stat( a_tst_path ).st_size / 4 / a_order)

    b_trn_len = int(os.stat( b_trn_path ).st_size / 4 / b_order)
    b_val_len = int(os.stat( b_val_path ).st_size / 4 / b_order)
    b_tst_len = int(os.stat( b_tst_path ).st_size / 4 / b_order)

    c_trn_len = int(os.stat( c_trn_path ).st_size  / 4 / c_order) ## /4 float32
    c_val_len = int(os.stat( c_val_path ).st_size / 4 / c_order)
    c_tst_len = int(os.stat( c_tst_path ).st_size / 4 / c_order)

    d_trn_len = int(os.stat( d_trn_path ).st_size / 4 / d_order)
    d_val_len = int(os.stat( d_val_path ).st_size / 4 / d_order)
    d_tst_len = int(os.stat( d_tst_path ).st_size / 4 / d_order)

    print("a_trn_len : ",a_trn_len)
    print("a_val_len : ",a_val_len)
    print("a_tst_len : ",a_tst_len)
    print("b_trn_len : ",b_trn_len)
    print("b_val_len : ",b_val_len)
    print("b_tst_len : ",b_tst_len)
    print("c_trn_len : ",c_trn_len)
    print("c_val_len : ",c_val_len)
    print("c_tst_len : ",c_tst_len)
    print("d_trn_len : ",d_trn_len)
    print("d_val_len : ",d_val_len)
    print("d_tst_len : ",d_tst_len)

    c_trn = th.from_numpy( np.memmap( c_trn_path , dtype="float32", mode="r", shape=(c_trn_len, c_order) ) )
    print("c_trn :",c_trn.shape)
    c_trn_mean = th.mean(c_trn, 0)
    c_trn_var = th.var(c_trn, 0)
    c_trn_std = th.std(c_trn, 0)
    print(c_trn_mean.shape)
    print(c_trn_var.shape)
    print(c_trn_std.shape)

    th.save(c_trn_mean, path.join(stat_dir_path, "c_trn_mean.pt"))
    th.save(c_trn_var, path.join(stat_dir_path, "c_trn_var.pt"))
    th.save(c_trn_std, path.join(stat_dir_path, "c_trn_std.pt"))

def check_data():
    print(os.getcwd())
    lab_state_dir_path = os.path.join("data","lab","state")
    lab_full_dir_path = os.path.join("data","lab","full")
    lf0_dir_path = os.path.join("data","lf0")
    mgc_dir_path = os.path.join("data","mgc")
    questions_dir_path = os.path.join("data","questions")

    lab_state_num , lab_full_num = [],[]
    lf0_num , mgc_num = [],[]

    for _ , _ ,filename in os.walk(lab_state_dir_path):
        lab_state_num = len(filename)
    for _ , _ ,filename in os.walk(lab_state_dir_path):
        lab_full_num = len(filename)
    for _ , _ ,filename in os.walk(lf0_dir_path):
        lf0_num = len(filename)
    for _ , _ ,filename in os.walk(mgc_dir_path):
        mgc_num = len(filename)

    #with open(os.path.join(questions_dir_path,questions_qst001.conf),ode="r") as f

    qst_path = os.path.join(questions_dir_path,"questions_qst001.conf")
    qst_num = len(open(qst_path,'r').readlines())

    print("lab_state :",lab_state_num)
    print("lab_full :",lab_full_num)
    print("lf0_num :",lf0_num)
    print("mgc_num :",mgc_num)
    print("qst_num :",qst_num)

def test():

    a_trn_list , a_tst_list , a_val_list = [],[],[]
    b_trn_list , b_tst_list , b_val_list = [],[],[]
    c_trn_list , c_tst_list , c_val_list = [],[],[]
    d_trn_list , d_tst_list , d_val_list = [],[],[]
    set_dir_path = os.path.join("data","set")

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

    a_trn_path = os.path.join("data","set","a_trn.bin")

    if not os.path.isfile(a_trn_path):
        for path in a_trn_list:
            print(path)
            data = load_binfile(path)
            buffer = struct.pack( "f" * len(data) , *data )
            with open(a_trn_path , "ab") as f :
                f.write(buffer)

def test2():

    list1 = [1.0 , 2.0 , 3.0 , 4.0]
    list2 = [5.0 , 6.0 , 7.0 , 8.0]

    buffer1 = struct.pack("f" * len(list1) , *list1)
    buffer2 = struct.pack("f" * len(list2) , *list2)

    print("buffer1 :",buffer1 , len(buffer1))
    print("buffer2 :",buffer2 , len(buffer2))

    set_dir_path = os.path.join(os.getcwd() , "data" , "set")

    with open(os.path.join(set_dir_path,"1.bin") , mode="wb") as f:
        f.write(buffer1)
    with open(os.path.join(set_dir_path,"2.bin") , mode="wb") as f:
        f.write(buffer2)

    with open(os.path.join(set_dir_path,"1.bin") , mode="rb") as f:
        buffer1 = f.read()
    with open(os.path.join(set_dir_path,"2.bin") , mode="rb") as f:
        buffer2 = f.read()

    print("buffer1 :",buffer1 , len(buffer1))
    print("buffer2 :",buffer2 , len(buffer2))

    with open(os.path.join(set_dir_path,"all.bin") , mode="ab") as f:
        f.write(buffer1)
    with open(os.path.join(set_dir_path,"all.bin") , mode="ab") as f:
        f.write(buffer2)

    with open(os.path.join(set_dir_path,"all.bin") , mode="rb") as f:
        buffer_all = f.read()

    print("buffer_all :",buffer_all , len(buffer_all))



if __name__ == '__main__':
    #check_data()
    #mkdir()
    mkdab()
    #mkc()
    #mkset()
    #mkstat()
