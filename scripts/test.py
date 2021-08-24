import demjson
import os
import json
import torch as th
import tts
import torch
import re
import nltk
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg


'''
blackjack_hand = {"a":1,"b":2}

with open("test.json", mode = "w") as IOconfigs_file:
    json.dump(blackjack_hand, IOconfigs_file)

with open("test.json", mode = "r") as IOconfigs_file:
    load_hand = json.load(IOconfigs_file)

print(load_hand, type(load_hand))
load_hand['c'] = 3

with open("test.json", mode = "w") as IOconfigs_file:
    json.dump(load_hand, IOconfigs_file ,indent ="")

with open("test.json", mode = "r") as IOconfigs_file:
    load_hand = json.load(IOconfigs_file)

print(load_hand, type(load_hand))


print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)


stat_dir_path = os.path.join("data", "stat")
var_path = os.path.join(stat_dir_path, "c_trn_var.pt")
device = th.device("cuda" if 1 else "cpu")
variance = th.load(var_path)
model = tts.AcousticModel( 1386, 1024, 79, variance=variance).to(device)


import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('integer' , type=int , help='display an integer' , ) ## 定位參數
parser.add_argument('--string' , type=bool , help='display an string')
parser.add_argument("--square",help="display a square of a given number" , type=int ,dest='test') ## 可選參數
parser.add_argument("--cubic",help="display a cubic of a given number" , type=int) ## 可選參數
args = parser.parse_args()

#print(args.integer)
print(args.string)

if args.test:
    print(args.test**2)

if args.cubic:
    print(args.cubic**3)


string = " 2995000  3932500 /A:y^uw-w+er+m/B:"
pattern = '\/A\:\w+\^'
print(string)

#pattern = re.compile(string)
result = re.search(pattern, string).group(0)
print(result)

nltk.download('gutenberg')
nltk.download('punkt')

# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(100):
    print(tok[x])

nltk.download('averaged_perceptron_tagger')
text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
'''

from g2p_en import G2p

texts = ["I have $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist."] # newly coined word
g2p = G2p()
for text in texts:
    out = g2p(text)
    print(out)
