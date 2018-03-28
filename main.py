'''
    Main process of entire project.
'''

import matplotlib.pyplot as plt
import pdb
import scipy.io.wavfile
import wave
import xmltodict

from json import loads, dumps
from os.path import join
from PreProcessingFunctions import *
from PublicFunctions import get_files


# MetaData get
xml_path = r'D:\DataSet\homburg_audio\MetaData'
Metas = []
Classes = {}
XMLs = get_files(xml_path)
for XML in XMLs:
    # Transfer the doc to lowercase due to some of the xml tags are UPPERCASE, WTF!!!!
    doc = xmltodict.parse(open(join(xml_path, XML)).read().lower())

    try:
        for band in doc['bands']['band']:
            # Some xml tags with nothing ...
            if band['songs']:
                # It WTF again, the type of band['songs']['song'] can be either list or OrderedDict...
                # Due to use the xmltodict.
                if isinstance(band['songs']['song'], list):
                    for song in band['songs']['song']:
                        Metas.append(dict(loads(dumps(song)), **{'band': band['@name']}))
                        Classes.append({song['@path']: {'class': song['@genre']}})
                else:
                    Metas.append(dict(loads(dumps(band['songs']['song'])), **{'band': band['@name']}))
                    Classes.update({band['songs']['song']['@path']: {'class': band['songs']['song']['@genre']}})
    except:
        pdb.set_trace()

# Audios get
adu_path = r'D:\DataSet\homburg_audio\waves'
audios = get_files(adu_path)

# set up
for audio in audios:
    sample_rate, signal = scipy.io.wavfile.read(join(adu_path, audio))  # read file
    # Keep all time range :10 seconds
    # signal = signal[0:int(10 * sample_rate)]

    signal = pre_emphasis(signal)
    frame, f_len = framing(signal, sample_rate)
    win = window(frame, f_len)
    f_bank = fft_filterbank(win, sample_rate)
    coef = mfccs(f_bank)

    pdb.set_trace()
    # Need save the coef.s for each audio.

    # Now we have the name, the genre of audios and their mfcc coef.
    Classes[audio[:-4]]['mfcc'] = coef


