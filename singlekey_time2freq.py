from scipy.io import wavfile
import lutorpy
import numpy as np
from os import listdir
from os.path import isfile, join

mypath = '/home/arda/yiding/singleKeys'
allfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

count = 1

for file_name in allfiles:
    fs, data = wavfile.read(file_name)
    for i in [0, 1]:
        a = data.T[i]
        p = np.fft.fft(a, fs)
        n = len(p)
        p = p[:n/2]
        p = abs(p)
        p = p/max(p)
        p = p[:4000]
        
        x = torch.fromNumpyArray(p)
        np.savetxt('singleData/' + file_name[:-4] + '_%d' % i + '.asc', x)
    print '%d/4000' % count
    count=count+1
