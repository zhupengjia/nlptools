#!/usr/bin/env python
import wave, numpy, subprocess, requests
"""
    Wrapper for speech recognition

    Author: Pengjia Zhu(zhupengjia@gmail.com)
"""

class Speech_Deepspeech:
    """
    Wrapper for deepspeech

    Input:
        - model: model path
        - alphabet: alphabet file
        - lm: lm file
        - trie: trie file
    """
    def __init__(self, model, alphabet, lm, trie):
        from deepspeech import Model as DSModel
        self.ds_model = DSModel(model, 26, 9, alphabet, 500)
        self.ds_model.enableDecoderWithLM(alphabet, lm, trie, 0.75, 1.85)

    @staticmethod
    def ogg2wav(oggfile, wavfile):
        """
            Convert ogg file to wav file

            Input:
                - oggfile: ogg file path
                - wavfile: wav file path
        """
        process = subprocess.run(['ffmpeg', '-i', oggfile, "-ar", "16000",  wavfile])
        if process.returncode != 0:
            raise Exception("something went wrong when converting voice data")

    def __call__(self, wavfile):
        with wave.open(wavfile, "rb") as fin:
            fs = fin.getframerate()
            audio = numpy.frombuffer(fin.readframes(fin.getnframes()), numpy.int16)
        return self.ds_model.stt(audio, fs)

