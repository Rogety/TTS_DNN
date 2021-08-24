import os
import struct

hts_dir_path = "/home/adamliu/Desktop/HTS-demo_CMU-ARCTIC-SLT_0803/gen/qst001/ver1/2mix/0/"

for _,_ ,file in os.walk(os.path.join(hts_dir_path)):
    lf0_filename = sorted([os.path.join(hts_dir_path,x) for x in file if x.endswith(".lf0")])
    mgc_filename= sorted([os.path.join(hts_dir_path,x) for x in file if x.endswith(".mgc")])
    uv_filename = sorted([os.path.join(hts_dir_path,x) for x in file if x.endswith(".pit")])

print(lf0_filename)

for filepath in uv_filename:
    with open(filepath, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        tmp = struct.unpack("f" * ((len(fileContent) -24) // 4), fileContent[20:-4])
        print(len(tmp))
        print(tmp)


    import pdb; pdb.set_trace()
