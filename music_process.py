import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import numpy as np
count = 1
names = {'A4-1.wav', 'A4-2.wav', 'A4-3.wav', 'A4-4.wav', 'B3-1.wav', 'B3-2.wav', 'B3-3.wav',
         'B3-4.wav', 'B4-1.wav', 'B4-2.wav', 'B4-3.wav', 'B4-4.wav', 'C3-1.wav', 'C3-2.wav', 'C3-3.wav', 'C3-4.wav',
         'C3A4-1.wav', 'C3A4-2.wav', 'C3A4-3.wav', 'C3A4-4.wav', 'C3B4-1.wav', 'C3B4-2.wav', 'C3B4-3.wav', 'C3B4-4.wav',
         'C3C4-1.wav', 'C3C4-2.wav', 'C3C4-3.wav', 'C3C4-4.wav', 'C3D4-1.wav', 'C3D4-2.wav', 'C3D4-3.wav', 'C3D4-4.wav',
         'C3E4-1.wav', 'C3E4-2.wav', 'C3E4-3.wav', 'C3E4-4.wav', 'C3F4-1.wav', 'C3F4-2.wav', 'C3F4-3.wav', 'C3F4-4.wav',
         'C3G4-1.wav', 'C3G4-2.wav', 'C3G4-3.wav', 'C3G4-4.wav', 'C4-1.wav', 'C4-2.wav', 'C4-3.wav', 'C4-4.wav',
         'C4A#4-1.wav', 'C4A#4-2.wav', 'C4A#4-3.wav', 'C4A#4-4.wav', 'C4A4-1.wav', 'C4A4-2.wav', 'C4A4-3.wav',
         'C4A4-4.wav', 'C4B4-1.wav', 'C4B4-2.wav', 'C4B4-3.wav', 'C4B4-4.wav', 'C4C#4-1.wav', 'C4C#4-2.wav',
         'C4C#4-3.wav', 'C4C#4-4.wav', 'C4D#4-1.wav', 'C4D#4-2.wav', 'C4D#4-3.wav', 'C4D#4-4.wav', 'C4D4-1.wav',
         'C4D4-2.wav', 'C4D4-3.wav', 'C4D4-4.wav', 'C4E4-1.wav', 'C4E4-2.wav', 'C4E4-3.wav', 'C4E4-4.wav',
         'C4E4G4-1.wav', 'C4E4G4-2.wav', 'C4E4G4-3.wav', 'C4E4G4-4.wav', 'C4F#4-1.wav', 'C4F#4-2.wav', 'C4F#4-3.wav',
         'C4F#4-4.wav', 'C4F4-1.wav', 'C4F4-2.wav', 'C4F4-3.wav', 'C4F4-4.wav', 'C4G#4-1.wav', 'C4G#4-2.wav',
         'C4G#4-3.wav', 'C4G#4-4.wav', 'C4G4-1.wav', 'C4G4-2.wav', 'C4G4-3.wav', 'C4G4-4.wav', 'C5-1.wav', 'C5-2.wav',
         'C5-3.wav', 'C5-4.wav', 'D4-1.wav', 'D4-2.wav', 'D4-3.wav', 'D4-4.wav', 'D4E4F4-1.wav', 'D4E4F4-2.wav',
         'D4E4F4-3.wav', 'D4E4F4-4.wav', 'E4-1.wav', 'E4-2.wav', 'E4-3.wav', 'E4-4.wav', 'F4-1.wav', 'F4-2.wav',
         'F4-3.wav', 'F4-4.wav', 'G4-1.wav', 'G4-2.wav', 'G4-3.wav', 'G4-4.wav'}

for file_name in names:

# TODO(Yiding) Apply FFT to the signal.
# fs, data = wavfile.read('A4-1.wav')
# a = data.T[0]  # channel soundtrack 1
# b = [(ele/2**16.)*2-1 for ele in a]  # this is 16-bit track, b is now normalized on [-1,1)
# c = fft(a)  # calculate fourier transform (complex numbers list)
# d = len(c)/2  # you only need half of the fft list (real signal symmetry)
# plt.plot(abs(c[:(d-1)]),'r')

# TODO(Yiding) Set xlabel in second.
# k = np.arange(len(data))
# frqLabel = k/fs

# TODO(Yiding) Plot the signal in time domain.
# plt.plot(frqLabel,abs(a),'b')
# plt.show()
    
    print 'No. %d Now processing ' % count + file_name
    count += 1
    music = AudioSegment.from_wav(file_name)
    chunks = split_on_silence(music,1000,-70)

    # TODO(Yiding) Split the sample to 100 segments.
    for i, chunk in enumerate(chunks):
        chunk.export('segments/'+file_name[:-4]+'-{0}.wav'.format('%.2d' % i), format="wav")
