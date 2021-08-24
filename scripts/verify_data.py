import struct
import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                     filename='log/mylog.txt',
                     filemode='w')


path_a = '/home/adamliu/Desktop/project/dnn_tts/data/set/a/cmu_us_arctic_slt_a0001.bin'
path_b = '/home/adamliu/Desktop/project/dnn_tts/data/set/b/cmu_us_arctic_slt_a0001.bin'
path_c = '/home/adamliu/Desktop/project/dnn_tts/data/set/c/cmu_us_arctic_slt_a0001.bin'
path_d = '/home/adamliu/Desktop/project/dnn_tts/data/set/d/cmu_us_arctic_slt_a0001.bin'
path_p = '/home/adamliu/Desktop/project/dnn_tts/data/set/p/cmu_us_arctic_slt_a0001.bin'
path_phone = '/home/adamliu/Desktop/project/HTS_2.2/data/labels/mono/cmu_us_arctic_slt_a0001.lab'

path_a_r = '/home/adamliu/Desktop/DNN-TTS-V5_MINE/data/source/set/a/cmu_us_arctic_slt_a0001.bin'
path_b_r = '/home/adamliu/Desktop/DNN-TTS-V5_MINE/data/source/set/b/cmu_us_arctic_slt_a0001.bin'
path_c_r = '/home/adamliu/Desktop/DNN-TTS-V5_MINE/data/source/set/c/cmu_us_arctic_slt_a0001.bin'
path_d_r = '/home/adamliu/Desktop/DNN-TTS-V5_MINE/data/source/set/d/cmu_us_arctic_slt_a0001.bin'

phone = []
with open(path_phone , mode="r" ) as f :
    for line in f :
        line = line.strip("\n")
        line = line.split(" ")
        if line != "":
            phone.append(line[-1])

logging.debug("\np")
with open(path_p , mode="rb" ) as f :
    tmp = f.read()
buffer = struct.unpack( "f" * (len(tmp) // 4) , tmp )

num = int (len(buffer) / 1380)
buffer = np.array(buffer)
buffer = np.reshape(buffer , (-1,1380))

for idx , item in enumerate(buffer):
    item = list(item)
    item.insert(0,phone[idx])
    logging.debug(item[0:1380])

## excel phone check
q_list = []
with open("/home/adamliu/Desktop/project/dnn_tts/data/questions/questions_qst001.conf","r") as qst :
    for line in qst:
        line = line.split(" ")
        q_list.append(line[0])

df = pd.DataFrame(buffer.transpose(), index=q_list, columns=phone )
with open('log/myphone_log.txt','w') as outfile:
    df.to_string(outfile)

'''
## check a
logging.debug("\na")
with open(path_a , mode="rb" ) as f :
    tmp = f.read()
buffer = struct.unpack( "f" * (len(tmp) // 4) , tmp )

num = int (len(buffer) / 1386)
buffer = np.array(buffer)
buffer = np.reshape(buffer , (-1,1386))
for item in buffer:
    logging.debug(list(item[0:1380]))

logging.debug("\nb")
with open(path_b , mode="rb" ) as f :
    tmp = f.read()
buffer = struct.unpack( "f" * (len(tmp) // 4) , tmp )

num = int (len(buffer) / 1382)
buffer = np.array(buffer)
buffer = np.reshape(buffer , (-1,1382))
for item in buffer:
    logging.debug(list(item[0:1380]))

logging.debug("\nc")
with open(path_c , mode="rb" ) as f :
    tmp = f.read()
buffer = struct.unpack( "f" * (len(tmp) // 4) , tmp )

num = int (len(buffer) / 79)
buffer = np.array(buffer)
buffer = np.reshape(buffer , (-1,79))
for item in buffer:
    logging.debug(list(item[0:79]))

logging.debug("\nd")
with open(path_d , mode="rb" ) as f :
    tmp = f.read()
buffer = struct.unpack( "f" * (len(tmp) // 4) , tmp )

num = int (len(buffer) / 1)
buffer = np.array(buffer)
buffer = np.reshape(buffer , (-1,1))
for item in buffer:
    logging.debug(list(item[0:1]))
'''
