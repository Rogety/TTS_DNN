import struct
import os


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
