'''
    Created by Tao

    Reference:
        http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Notes:
        frames_ham and frames_kai is the final value after three steps,type is <class 'numpy.ndarray'>
'''

import wave as wave
import numpy as numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt

# set up
sample_rate, signal = scipy.io.wavfile.read('1nothing-walk_away.wav')  # read file
# sample_rate, signal = scipy.io.wavfile.read('22-Lonely.wav')
# sample_rate, signal = scipy.io.wavfile.read('3_Blind_Mice-Emily_Has_Compassion_Fatigue.wav')
# File assumed to be in the same directory
# *** Keep sll time range :10 seconds
signal =signal[0:int(10* sample_rate)]

# ***other method for getting more information from this wave file
# f = wave.open("1nothing-walk_away.wav", "rb")
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]

# step 1:pre_emphasis
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# Step 2:Framing
frame_size = 0.025 # 25 ms for the frame size
frame_stride = 0.01 # a 10 ms stride (15 ms overlap)
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
# Make sure that we have at least 1 frame
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
# Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
pad_signal = numpy.append(emphasized_signal, z)

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]

# Step3 window function
# kaiser window
frames_kai = numpy.kaiser(frame_length, 5) * frames
# hamming window
frames_ham = numpy.hamming(frame_length) * frames
# ***frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

# choose each frame for ploting
kai = frames_kai[0]
ham = frames_ham[0]

# time
lenthk=len(kai)
ti = 0.025   # 25ms
tk = numpy.arange(0, ti, ti/lenthk)

lenthh=len(ham)
ti = 0.025   # 25ms
th = numpy.arange(0, ti, ti/lenthh)


# plot window with data both hum and kai in one graph and they are very similar
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(th,ham)
plt.title("humming window",)
plt.subplot(3,1,2)
plt.plot(tk,kai)
plt.title("kaiser window")
plt.subplot(3,1,3)
plt.plot(th,ham,linewidth=1)
plt.plot(tk,kai,linewidth=1)
plt.show()

# ***other method for comparing with them(313)
# b0=numpy.hstack((frames_ham[0],frames_kai[0]))
# b1=numpy.vstack((ham,kai))
# b=b1.T
# plt.subplot(3,1,3)
# plt.plot(tk,b)
# plt.title("kaiser window")

# plot only humming window
plt.figure(2)
plt.plot(th,ham)
plt.title("humming window",)
plt.show()

# print the value for FFT
# *****frames_ham and frames_kai is the final value after three steps,type is <class 'numpy.ndarray'>*****
print(frames_ham[0])  #one example
print(type(frames_ham[0])) #for one example <class 'numpy.ndarray'>
print(type(frames_ham))  #<class 'numpy.ndarray'>

print(frames_kai[0])
print(type(frames_kai))


# ***plot only window function
# plt.figure(3)
# plt.subplot(2,1,1)
# #hamming window
# from scipy.signal import get_window
# m=513
# t=numpy.arange(m)
# w=get_window('hamming',m)
# plt.plot(t, w)
# # plt.xlabel("(time) sample #")
# plt.ylabel("amplitude")
# plt.title("hamming window")
# plt.xlim(0, m-1)
# plt.ylim(-0.025, 1.025)
#
# plt.subplot(2,1,2)
# #kaiser window
# from scipy.signal import get_window
# m=500
# t=numpy.arange(m)
# w=get_window(('kaiser',4.0),m)
# plt.plot(t, w)
# plt.xlabel("time(seconds)")
# plt.ylabel("amplitude")
# plt.title("kaiser window")
# plt.xlim(0, m-1)
# plt.ylim(-0.025, 1.025)
# plt.show()