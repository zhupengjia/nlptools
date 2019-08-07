#!/usr/bin/env python
import sys, os, argparse
from nlptools.audio.tts import MozillaTTS
from nlptools.audio.utils import *
import spacy

tts_model = "/home/pzhu/data/tts/checkpoint_261000.pth.tar"
tts_config = "/home/pzhu/data/tts/config.json"
wavernn_model = "/home/pzhu/data/wavernn/checkpoint_433000.pth.tar"
wavernn_config = "/home/pzhu/data/wavernn/config.json"

parser = argparse.ArgumentParser(description='txt file to wav file')
parser.add_argument('-i', '--in', dest='txtfile', help='input txt file')
parser.add_argument('-o', '--out', dest='wavfile', help='output wav file')
parser.add_argument('-tm', '--tts_model', dest='tts_model', default='/home/pzhu/data/tts/checkpoint_261000.pth.tar', help='tts model path')
parser.add_argument('-tc', '--tts_config', dest='tts_config', default='/home/pzhu/data/tts/config.json', help='tts model config path')
parser.add_argument('-wm', '--wavernn_model', dest='wavernn_model', default='/home/pzhu/data/wavernn/checkpoint_433000.pth.tar', help='wavernn model path')
parser.add_argument('-wc', '--wavernn_config', dest='wavernn_config', default='/home/pzhu/data/wavernn/config.json', help='wavernn model config path')

args = parser.parse_args()

if args.txtfile is None:
    parser.print_help()
    sys.exit()

if args.wavfile is None:
    args.wavfile = args.txtfile + ".wav"

if not os.path.exists(args.wavernn_model) or not os.path.exists(args.wavernn_config):
    args.wavernn_model = None
    args.wavernn_config = None

nlp = spacy.load("en")
tts_model = MozillaTTS(args.tts_model, args.tts_config, args.wavernn_model, args.wavernn_config, device="cuda:0")

base_name = "/tmp/gen_{}.wav"

wavefiles = []

with open(args.txtfile) as f:
    doc = nlp(f.read())
    for i, sent in enumerate(doc.sents):
        text = sent.text.strip()
        print("----------------")
        print(text)
        wavefile = base_name.format(i)
        tts_model(text, wavefile)
        wavefiles.append(wavefile)

concat_wavs(wavefiles, args.wavfile, 0.6)

for w in wavefiles:
    os.remove(w)
