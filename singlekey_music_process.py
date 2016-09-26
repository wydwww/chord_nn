import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from os import listdir
from os.path import isfile, join
import numpy as np
count = 1
mypath = '/Users/intel/music_real/'
names = [f for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith('wav')]


outputpath = '/Users/intel/real_output/'
for file_name in names:

# Apply FFT to the signal.
# fs, data = wavfile.read('A4-1.wav')
# a = data.T[0]  # channel soundtrack 1
# b = [(ele/2**16.)*2-1 for ele in a]  # this is 16-bit track, b is now normalized on [-1,1)
# c = fft(a)  # calculate fourier transform (complex numbers list)
# d = len(c)/2  # you only need half of the fft list (real signal symmetry)
# plt.plot(abs(c[:(d-1)]),'r')

# Set xlabel in second.
# k = np.arange(len(data))
# frqLabel = k/fs

# Plot the signal in time domain.
# plt.plot(frqLabel,abs(a),'b')
# plt.show()
    
    print 'No. %d Now processing ' % count + file_name
    count += 1
    music = AudioSegment.from_wav(mypath+file_name)
    chunks = split_on_silence(music,100,-20)

    # Split the sample to 100 segments.
    for i, chunk in enumerate(chunks):
        chunk.export(outputpath + file_name[:-4]+'-{0}.wav'.format('%.2d' % i), format="wav")
