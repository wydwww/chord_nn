from scipy.io import wavfile
# import lutorpy
import numpy as np
from os import listdir
from os.path import isfile, join

mypath = '/Users/intel/real_output/'
allfiles = [f for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith('wav')]
outputpath = '/Users/intel/real_asc/'
count = 1

for file_name in allfiles:
    fs, data = wavfile.read(mypath + file_name)
    a = data.T[0]
    p = np.fft.fft(a, fs)
    n = len(p)
    p = p[:n/2]
    p = abs(p)
    p = p/max(p)
    p = p[:4000]
    
    # x = torch.fromNumpyArray(p)
    np.savetxt(outputpath + file_name[:-4] + '.asc', p)
    print '%d/12000' % count
    count=count+1
