import struct
import binascii
import pickle
import os
'''
no = 365.9731750488281
print("[05]: Type is %s" % type(no))

packNo = struct.pack('f', no)
print("[08]: Type is %s, packNo is %s" % (type(packNo), packNo))

reverseBytes = packNo[::-1]
print("[11]: %s" % reverseBytes)

bytesToHex = binascii.b2a_hex(reverseBytes)
print("[14]: %s" % bytesToHex)

bytesHexToASCII = bytesToHex.decode('ascii')
print("[17]: %s" % bytesHexToASCII)

unpackFromhex = struct.unpack('!f', bytes.fromhex(bytesHexToASCII))[0]
print("[20]: %s" % unpackFromhex)

unpackBytes = struct.unpack('f', b'\x91\xfc\xb6C')[0]
print("[23]: %s" % unpackBytes)
'''
'''
buffer = struct.pack("f", 120.0)
print("floating : ", repr(buffer))
with open("test_float.bin", "wb") as f :
    f.write(buffer)
with open("test_float.bin", "rb") as f :
    buffer = f.read(4)
    result = struct.unpack("f", buffer)
size = os.path.getsize("test.bin")
print(result, size)

buffer = struct.pack("i", 120)
print("integer :", repr(buffer))
with open("test_int.bin", "wb") as f :
    f.write(buffer)
with open("test_int.bin", "rb") as f :
    buffer = f.read(4)
    result = struct.unpack("i", buffer)
size = os.path.getsize("test.bin")
print(result, size)

'''



'''
#mybytes = [120.0, 3.1, 255.3, 0.6, 100.9]
mybytes = 120.0
with open("bytesfile.bin", "wb") as mypicklefile:
    pickle.dump(mybytes, mypicklefile)
with open("bytesfile.bin", "rb") as mypicklefile:
    result = pickle.load(mypicklefile)
size = os.path.getsize("bytesfile.bin")
print(result, size)
'''
