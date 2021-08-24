import os
import json
from os import path
import re
import struct
import utils as ut



with open(path.join("configs","Configs.json")) as configs_file:
    configs = json.load(configs_file)

#print(configs)
questions_set = []
with open(path.join("data","questions","questions_qst001.conf")) as questions:
    for line in questions :
        line = line.strip("QS ")
        line = line.replace("\t"," ")
        line = line.rstrip("\n")
        if line != "":
            questions_set.append(line)

with open(path.join("configs","qstConfigs.json"), mode = "r") as qst_configs_file:
    qst_configs = json.load(qst_configs_file)

pattern_dict = qst_configs.copy()
pattern_dict.pop("length")
linquistic_list = list(pattern_dict.keys())
'''
print(pattern_dict)
print(linquistic_list)

string = " 2995000  3932500 /A:y^uw-w+er+m/B:"
for i in range(len(linquistic_list)):
    pattern = pattern_dict[linquistic_list[i]]
    result = re.search(pattern, string).group(0)
    print(result)
'''


source_dir_path = configs["source_dir_path"]
frame_shift = configs["frame_shift"]*(10**7)
frame_shift = configs["frame_shift"]*(10**7)

def count_frames(Start , End):
    frames_in_phone = []
    frames_in_state = []

    for i in range(len(Start)):
        if i % 5 == 0:
            frameses = []
        frames = (float(End[i])-float(Start[i])) / frame_shift
        frameses.append(frames)
        if i % 5 == 4 :
            frames_in_state.append(frameses)

    for i in range(len(Start)):
        if i % 5 == 0:
            start = float(Start[i])
        if i % 5 == 4:
            end = float(End[i])
            frames = (end-start) / frame_shift
            frames_in_phone.append(frames)

    # check
    fail = False
    for i in range(len(frames_in_phone)):
        total_frames_in_state = 0
        for frame in frames_in_state[i]:
            total_frames_in_state += frame
        if total_frames_in_state != frames_in_phone[i]:
            fail = True

    if fail==False :
        print("phone and state frames check ok")
    else :
        print("failed")

    #print("frames_in_state",frames_in_state)
    return frames_in_phone, frames_in_state

def qask(pstr , questions_set , flag , frames_in_state=0 , frames_in_phone=0):

    '''
    pattern_dict = {"LLphone": r'\w+\^',"Lphone":r'\^\w+\-',"Cphone":r'\-\w+\+',"Rphone":r'\+\w+\=',"RRphone":r'\=\w+\@',
    "ph_in_syl_fw":r'\@\w+\_',"ph_in_syl_bw":r'\_\w+\/A\:', ## p
    "Lstress":r'\/A\:\w+\_',"phnum_in_presyl":r'\_\w+\/B\:', ## a
    "Cstress":r'\/B\:\w+\-',"phnum_in_cursyl":r'\-\w+\-',"syl_in_word_fw":r'\-\w+\@',"syl_in_word_bw":r'\@\w+\-',"syl_in_phrase_fw":r'\-\w+\&',
    "syl_in_phrase_bw":r'\&\w+\-',"b7":r'\-\w+\#',"b8":r'\#\w+\-',"b9":r'\-\w+\$',"b10":r'\$\w+\-',"vowel_in_syl":r'\-\w+\/C\:', ## b
    "Rstress":r'\/C\:\w+\+',"phnum_in_nextsyl":r'\+\w+\/D\:', ## c
    "gpos_in_preword":r'\/D\:\w+\_',"sylnum_in_preword":r'\_\w+\/E\:', ## d
    "gpos_in_curword":r'\/E\:\w+\+',"sylnum_in_curword":r'\+\w+\@',"word_in_phrase_fw":r'\@\w+\+',"word_in_phrase_bw":r'\+\w+\&',
    "e5":r'\&\w+\+',"e6":r'\+\w+\#',"e7":r'\#\w+\+',"e8":r'\+\w+\/F\:', ## e
    "gpos_in_next_word":r'\/F\:\w+\_',"sylnum_in_nextword":r'\_\w+\/G\:', ## f
    "sylnum_in_prephrase":r'\/G\:\w+\_',"wordnum_in_curphrase":r'\_\w+\/H\:', ## g
    "sylnum_in_curphrase":r'\/H\:\w+\=',"wordnum_in_prephrase":r'\=\w+\^',"phrase_in_sentense_fw":r'\^\w+\=',"phrase_in_sentense_bw":r'\=\w+\/I\:', ## h
    "sylnum_in_nextphrase":r'\/I\:\w+\_',"wordnum_in_nextphrase":r'\_\w+\/J\:', ## i
    "sylnum_in_sentense":r'\/J\:\w+\+',"wordnum_in_sentense":r'\+\w+\-',"phrasenum_in_sentense":r'\-\w', ## j
    }
    linquistic_list = ["LLphone","Lphone","Cphone","Rphone","RRphone","ph_in_syl_fw","ph_in_syl_bw", ## p
    "Lstress","phnum_in_presyl", ## a
    "Cstress","phnum_in_cursyl","syl_in_word_fw","syl_in_word_bw","syl_in_phrase_fw","syl_in_phrase_bw","b7","b8","b9","b10","vowel_in_syl", ## b
    "Rstress","phnum_in_nextsyl", ## c
    "gpos_in_preword","sylnum_in_preword", ## d
    "gpos_in_curword","sylnum_in_curword","word_in_phrase_fw","word_in_phrase_bw","e5","e6","e7","e8", ## e
    "gpos_in_next_word","sylnum_in_nextword", ## f
    "sylnum_in_prephrase","wordnum_in_curphrase", ## g
    "sylnum_in_curphrase","wordnum_in_prephrase","phrase_in_sentense_fw","phrase_in_sentense_bw", ## h
    "sylnum_in_nextphrase","wordnum_in_nextphrase", ## i
    "sylnum_in_sentense","wordnum_in_sentense","phrasenum_in_sentense", ## j
    ]
    '''
    #pattern_dict = {"LLphone": r'\/A\:\w+\^',"Lphone":r'\^\w+\-',"Cphone":r'\-\w+\+',"Rphone":r'\+\w+\+',"RRphone":r'\+\w+\/B\:'}
    #linquistic_list = ["LLphone","Lphone","Cphone","Rphone","RRphone"]

    '''
    lab_dict = {}
    for i in range(len(linquistic_list)):
        if linquistic_list[i] == "LLphone":
            lab_dict[linquistic_list[i]] = re.search( pattern_dict[linquistic_list[i]], pstr).group(0) + "*"
        elif linquistic_list[i] == "phrasenum_in_sentense":
            lab_dict[linquistic_list[i]] = "*" + re.findall( pattern_dict[linquistic_list[i]], pstr)[-1]
            #print("phrasenum_in_sentense :",lab_dict[linquistic_list[i]])
        else:
            try :
                lab_dict[linquistic_list[i]] = "*" + re.search( pattern_dict[linquistic_list[i]], pstr).group(0) + "*"
            except :
                lab_dict[linquistic_list[i]] = "none"

                #print("error")
    '''
    lab_dict = {}
    for i in range(len(linquistic_list)):
        lab_dict[linquistic_list[i]] = "*" + re.search( pattern_dict[linquistic_list[i]], pstr).group(0) + "*"

    #print(lab_dict)


    ## 計算 questionset match 的 index

    qes_list = []
    for i in range(len(lab_dict)) :
        for index , qes in enumerate(questions_set) :
            #print(qes)
            qes = qes.split(" ")[1]
            qes = qes.strip("\{\}")
            qes = qes.split(",")
            #print(qes)

            if lab_dict[linquistic_list[i]] in qes :
                #print(index , lab_dict[linquistic_list[i]] )
                qes_list.append(index)

    ## generate phone answer
    ## phone
    ans = []
    for i in range(len(questions_set)):
        if i in qes_list:
            ans.append(1.0)
        else:
            ans.append(0.0)

    #print("phone" , len(ans) )

    if flag == "state":
        ans_state = []
        fw = [0.0,0.25,0.5,0.75,1.0]
        for i in range(5):
            tmp = ans.copy()
            tmp.extend([fw[i],fw[4-i]])
            ans_state.append(tmp)
        #print("state" , len(ans_state) , len(ans_state[0]))
        '''
        for item in ans_state:
            print("ans_state", item[-2] ,item[-1])
        '''
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
        #print("frames_in_phone", frames_in_phone)
        #print("state_in_phone_fw", state_in_phone_fw, len(state_in_phone_fw))
        #print("state_in_phone_bw", state_in_phone_bw, len(state_in_phone_bw))
        for i in range(int(frames_in_phone)):
            tmp = ans.copy()
            tmp.extend([state_in_phone_fw[i],state_in_phone_bw[i]])
            tmp.extend([frame_in_state_fw[i],frame_in_state_bw[i]])
            tmp.extend([frame_in_phone_fw[i],frame_in_phone_fw[int(frames_in_phone)-1-i]])
            ans_frame.append(tmp)
        #print(ans_frame)
        #print(len(ans_frame) , len(ans_frame[0]))
        '''
        print("frame" , len(ans_frame) , len(ans_frame[0]))
        for item in ans_frame:
            print("ans_frame", item[-6] ,item[-5],item[-4] ,item[-3],item[-2] ,item[-1])
        '''
        return ans_frame
        #print("frame" , state_index , frame_in_state)
    else :
        return ans



if __name__ == '__main__':

    if not path.isdir(path.join(source_dir_path,"stat")):
        os.mkdir(path.join(source_dir_path,"stat"))
    if not path.isdir(path.join(source_dir_path,"set")):
        os.mkdir(path.join(source_dir_path,"set"))
        os.mkdir(path.join(source_dir_path,"set","a"))
        os.mkdir(path.join(source_dir_path,"set","b"))
        os.mkdir(path.join(source_dir_path,"set","c"))
        os.mkdir(path.join(source_dir_path,"set","d"))
        os.mkdir(path.join(source_dir_path,"set","p"))
    if not path.isdir("model"):
        os.mkdir("model")

    lab_state_dir = path.join(source_dir_path, "lab", "state")
    a_dir_path = path.join(source_dir_path, "set", "a")
    b_dir_path = path.join(source_dir_path, "set", "b")
    d_dir_path = path.join(source_dir_path, "set", "d")
    p_dir_path = path.join(source_dir_path, "set", "p")

    qst_length = qst_configs["length"]
    io_config_path = os.path.join("configs","IOConfigs.json")
    io_config = {"a_order":qst_length+6,"b_order":qst_length+2,"c_order":79,
                    "d_order":1,"d_class":91}
    with open(io_config_path, mode = "w") as IOconfigs_file:
            json.dump(io_config, IOconfigs_file, indent ="",sort_keys=True)


    for _ , _ ,filename in os.walk(lab_state_dir):
        labpaths = sorted([os.path.join(lab_state_dir,x) for x in filename])

    # test
    #labpaths = [os.path.join(lab_state_dir,"cmu_us_arctic_slt_a0001.lab")]
    #print(labpaths)



    for lab_path in labpaths:
        print(lab_path)
        lab_phone, Start, End = [],[],[]

        with open(lab_path) as lab_file:
            for line in lab_file :
                line = line.split(" ")
                Start.append(line[0])
                End.append(line[1])
                if len(line) > 3:
                    lab_phone.append(line[3])

        ## check
        if len(Start) == len(lab_phone)*5 :
            print("phone and state label check ok")
        else :
            print("failed")

        ans_of_phone,ans_of_state,ans_of_frame = [],[],[]
        frames_in_phone, frames_in_state = count_frames(Start , End)
        #print("frames_in_phone :",frames_in_phone,len(frames_in_phone))
        #print("frames_in_state :",frames_in_state,len(frames_in_state))
        #print(questions_set)
        for index , lab in enumerate(lab_phone):
            ans = qask(lab , questions_set,"phone")
            #print(index , ans)
            ans_of_phone.append(ans)
            ans = qask(lab , questions_set,"state")
            for item in ans:
                ans_of_state.append(item)
            ans = qask(lab , questions_set,"frame" , frames_in_state[index] , frames_in_phone[index])
            for item in ans:
                ans_of_frame.append(item)


        print(len(ans_of_phone))
        ## check
        if len(ans_of_phone[0])+2 == len(ans_of_state[0]) and len(ans_of_phone[0])+6 == len(ans_of_frame[0]):
            print(len(ans_of_phone) , len(ans_of_phone[0]))
            print(len(ans_of_state) , len(ans_of_state[0]))
            print(len(ans_of_frame) , len(ans_of_frame[0]))
            print("question set num check ok")
        else:
            print("failed")

        ans_of_phone_flattern = [y for x in ans_of_phone for y in x]
        ans_of_state_flattern = [y for x in ans_of_state for y in x]
        ans_of_frame_flattern = [y for x in ans_of_frame for y in x]
        frames_in_state_flattern = [y for x in frames_in_state for y in x]

        print("ans_of_frame_flattern : %d , frames : %d , type : %s" % (len(ans_of_frame_flattern) , len(ans_of_frame_flattern)/io_config["a_order"] , type(ans_of_frame_flattern[0])))
        print("ans_of_state_flattern : %d , states : %d , type : %s" % (len(ans_of_state_flattern) , len(ans_of_state_flattern)/io_config["b_order"], type(ans_of_state_flattern[0])))
        print("frames_in_state_flattern : ",len(frames_in_state_flattern))


        base = os.path.basename(lab_path).rstrip(".lab") + ".bin"
        a_bin_path = path.join(a_dir_path , base)
        b_bin_path = path.join(b_dir_path , base)
        d_bin_path = path.join(d_dir_path , base)
        p_bin_path = path.join(p_dir_path , base)
        #print(frames_in_state_flattern)
        #print("ans_of_frame_flattern : ",len(ans_of_frame_flattern))
        ut.save_binfile(p_bin_path , ans_of_phone_flattern)
        ut.save_binfile(a_bin_path , ans_of_frame_flattern)
        ut.save_binfile(b_bin_path , ans_of_state_flattern)
        ut.save_binfile(d_bin_path , frames_in_state_flattern)
