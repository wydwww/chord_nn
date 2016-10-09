import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random

# count = 1
mypath = '/Users/intel/music_real/'
outputpath = '/Users/intel/real_output/'
ascoutputpath = '/Users/intel/real_asc/'

names = [f for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith('wav')]

for file_name in names:    
    # print 'No. %d Now processing ' % count + file_name
    # count += 1
    music = AudioSegment.from_wav(mypath+file_name)
    chunks = split_on_silence(music,100,-20)

    for i, chunk in enumerate(chunks):
        chunk.export(outputpath + file_name[:-4]+'-{0}.wav'.format('%.2d' % i), format="wav")

allfile = [ f for f in listdir(outputpath) if isfile(join(outputpath,f)) and f.endswith('wav') ]
# allfiles = []
# for ff in allfile:
#     a = re.match('[A-Z]?[0-9]?-.+',ff)
#     if a:
#         allfiles.append(a.group())

# count = 1

for file_name in allfile:
    fs, data = wavfile.read(outputpath+file_name)
    for i in [0, 1]:
        a = data.T[i]
        p = np.fft.fft(a, fs)
        n = len(p)
        p = p[:n/2]
        p = abs(p)
        p = p/max(p)
        p = p[:4000]
        x = torch.fromNumpyArray(p)
        np.savetxt(ascoutputpath + file_name[:-4] + '_%d' % i + '.asc', p)
    # print '%d' % count
    # count=count+1

# intergrate

ascfiles = [f for f in listdir(ascoutputpath) if isfile(join(ascoutputpath,f)) and f.endswith('asc')]
random.shuffle(ascfiles)
random.shuffle(ascfiles)

f = open('./data.asc', 'w')

for file_name in ascfiles:
    # print file_name.split('-')[0]
    label = file_name.split('-')[0]
    f.write(label + ' ')
    ff = open(mypath + file_name, 'r')
    for line in ff.read().split('\n'):
        f.write(line + ' ')
    # f.write(ff.read().split(''))
    ff.close()
    f.write('\n')

f.close()

