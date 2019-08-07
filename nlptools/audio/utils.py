#!/usr/bin/env python
import subprocess, random, wave, os
from ..utils import flat_list
"""
    Some audio utils
"""


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

def wav2ogg(wavfile, oggfile):
    """
        Convert wav file to ogg file

        Input:
            - wavfile: wav file path
            - oggfile: ogg file path
    """
    process = subprocess.run(['ffmpeg', '-i', wavfile, oggfile])
    if process.returncode != 0:
        raise Exception("something went wrong when converting voice data")

def make_silence(wavfile, length=0.6, channel=1, rate=22050):
    """
        make a silence wav

        Input:
            - wavfile: wav file path
            - length: silence length, default is 0.6
            - channel: default is 1
            - rate: sample rate, default is 22050
    """
    process = subprocess.run(["sox", "-n", "-r", str(rate), "-c", str(channel), wavfile, "trim", "0.0", str(length)])
    if process.returncode != 0:
        raise Exception("something went wrong when making silence wav, please check if you have sox installed")

def concat_wavs(wavfiles, outfile, silence=0):
    """
        concatenate wave files. make sure all wave files have same rate and channel

        Input:
            - wavfiles: list of wave filepaths
            - outfile: output wave file
    """
    if silence > 0:
        silence_file = "/tmp/silence_{}.wav".format(random.randint(0,100000))
        with wave.open(wavfiles[0], "rb") as wf:
            rate = wf.getframerate()
            channel = wf.getnchannels()
        make_silence(silence_file, silence, channel=channel, rate=rate)
        process = subprocess.run(["sox"]
                                 + flat_list(zip(wavfiles, [silence_file]*len(wavfiles)))
                                 + [outfile])
        os.remove(silence_file)
    else:
        process = subprocess.run(["sox"] + wavfiles + [outfile])
    if process.returncode != 0:
        print(process.returncode)
        raise Exception("something went wrong when making silence wav, please check if you have sox installed")

