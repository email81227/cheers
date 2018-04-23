'''
Before started, install python package "pydub" ("pip install pydub") and ffmpeg
(https://ffmpeg.zeranoe.com/builds/) first for converting.

The whole process can refer here: How to Install FFmpeg on Windows;
http://adaptivesamples.com/how-to-install-ffmpeg-on-windows/
'''
import pdb

from pydub import AudioSegment
from os.path import join
from PublicFunctions import get_folders, get_files

import_path = r'D:\DataSet\homburg_audio\audios'
export_path = r'D:\DataSet\homburg_audio\waves'
ffmpeg_path = r'D:\Develop\Python\ffmpeg\bin'

# demos = [r'.\SampleAudio_0.4mb', r'.\SampleAudio_0.7mb']
# Point the converter's location
AudioSegment.converter = join(ffmpeg_path, 'ffmpeg.exe')

for folder in get_folders(import_path):
    for file in get_files(join(import_path, folder)):
        sound = AudioSegment.from_mp3(join(join(import_path, folder), file))
        sound.export(join(export_path, file[:-4] + '.wav'), format="wav")

