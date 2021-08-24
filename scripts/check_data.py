import os


print(os.getcwd())
lf0path = os.path.join("data","lf0")
mgcpath = os.path.join("data","mgc")

for _ , _ ,lf0_filename in os.walk(lf0path) :
    lf0paths = sorted([os.path.join("data","lf0",x) for x in lf0_filename])

for _,_,mgc_filename in os.walk(mgcpath) :
    mgcpaths = sorted([os.path.join("data","mgc",x) for x in mgc_filename])



#print(lf0paths)
#print(mgcpaths)

#print(os.stat(lf0paths[0]).st_size % 4)

lf0_dict = {}
lf0_frame = []
mgc_frame = []

for lf0 in lf0paths :
    lf0_dict[lf0] = os.stat(lf0).st_size // 4
    lf0_frame.append(os.stat(lf0).st_size // 4 )
    #print(lf0 , os.stat(lf0).st_size // 4)
    #print(lf0)
#print(lf0_dict)


for mgc in mgcpaths :
    #print(os.stat(mgc).st_size // 4 // 25)
    mgc_frame.append(os.stat(mgc).st_size // 4 // 25)

for i in range(len(lf0_frame)):
    if lf0_frame[i] != mgc_frame[i]:
        print(i , lf0_frame[i] , mgc_frame[i] , lf0paths[i] ,mgcpaths[i])
