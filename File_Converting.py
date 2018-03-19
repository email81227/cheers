# import subprocess
import pydub
from pydub import AudioSegment

demos = [r'.\SampleAudio_0.4mb', r'.\SampleAudio_0.7mb']

AudioSegment.converter = r'H:\Python\ffmpeg-3.4.2-win64-static\ffmpeg-3.4.2-win64-static\bin\ffmpeg.exe'

for demo in demos:
    sound = AudioSegment.from_mp3(demo + '.mp3')
    sound.export(demo + '.wav', format="wav")

