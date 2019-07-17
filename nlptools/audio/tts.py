#!/usr/bin/env python
import os
import sys
import io
import torch 
import time
import numpy as np
from collections import OrderedDict

import librosa
import librosa.display

from TTS.models.tacotron import Tacotron 
from TTS.layers import *
from TTS.utils.data import *
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text import text_to_sequence
from TTS.utils.synthesis import synthesis


class MozillaTTS:
    """
        Wrapper for Mozilla TTS
        
        Related repositories:
            - Mozilla TTS:
                - https://github.com/mozilla/TTS
                - commit 824c091
                - data: https://drive.google.com/drive/folders/1FJRjGDAqWIyZRX4CsppaIPEW8UWXCWzF?usp=drive_open
            - WaveRNN(optional):
                - https://github.com/erogol/WaveRNN
                - commit 8a1c152
                - data: https://drive.google.com/drive/folders/1wpPn3a0KQc6EYtKL0qOi4NqEmhML71Ve

    """
    def __init__(self, tts_model, tts_config, wavernn_model=None, wavernn_config=None, device="cpu"):
        self.tts_config = load_config(tts_config)
        self.tts_config.windowing = True
        if not torch.cuda.is_available():
            device = "cpu"
        self.use_cuda = device != "cpu"
        self.device = torch.device(device)
        self.tts_model_path = tts_model

        self._load_tts()

        if wavernn_model and wavernn_config:
            self.use_gl = False
            self.batched_wavernn = True
            self.wavernn_model_path = wavernn_model
            self.wavernn_config = load_config(wavernn_config)
            self._load_wavernn()
        else:
            self.use_gl = True

    def _load_tts(self):
        # LOAD TTS MODEL
        from TTS.utils.text.symbols import symbols, phonemes

        # load the model
        num_chars = len(phonemes) if self.tts_config.use_phonemes else len(symbols)
        self.tts_model = setup_model(num_chars, self.tts_config)

        # load the audio processor
        self._ap = AudioProcessor(**self.tts_config.audio)         

        # load model state
        cp = torch.load(self.tts_model_path, map_location=lambda storage, loc: storage)

        # load the model
        self.tts_model.load_state_dict(cp['model'])
        self.tts_model.to(self.device)
        self.tts_model.eval()
        self.tts_model.decoder.max_decoder_steps = 2000

    def _load_wavernn(self):
        from WaveRNN.models.wavernn import Model
        bits = 10

        self.wavernn = Model(
                rnn_dims=512,
                fc_dims=512,
                mode="mold",
                pad=2,
                upsample_factors=self.wavernn_config.upsample_factors,  # set this depending on dataset
                feat_dims=self.wavernn_config.audio["num_mels"],
                compute_dims=128,
                res_out_dims=128,
                res_blocks=10,
                hop_length=self._ap.hop_length,
                sample_rate=self._ap.sample_rate,
            ).to(self.device)

        check = torch.load(self.wavernn_model_path)
        self.wavernn.load_state_dict(check['model'])
        self.wavernn.eval()

    def __call__(self, text, out_path):
        waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens = synthesis(self.tts_model, text, self.tts_config, self.use_cuda, self._ap, False, self.tts_config.enable_eos_bos_chars)
        if not self.use_gl:
            waveform = self.wavernn.generate(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0).to(self.device), batched=self.batched_wavernn, target=11000, overlap=550)

        self._ap.save_wav(waveform, out_path)

if __name__ == "__main__":
    sentence = "Contents have various formats and specific rules need to be designed for each case"
    tts_model = "/home/pzhu/data/tts/checkpoint_261000.pth.tar"
    tts_config = "/home/pzhu/data/tts/config.json"
    wavernn_model = "/home/pzhu/data/wavernn/checkpoint_433000.pth.tar"
    wavernn_config = "/home/pzhu/data/wavernn/config.json"

    tts = MozillaTTS(tts_model, tts_config, wavernn_model=None, wavernn_config=None, device="cuda:0")
    tts(sentence, "/tmp/tts.wav")

