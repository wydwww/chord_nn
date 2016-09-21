from os import listdir
from os.path import isfile, join
import random

mypath = './singledata/'
allfiles = [f for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith('asc')]
random.shuffle(allfiles)
random.shuffle(allfiles)

f = open('./data.asc', 'w')

for file_name in allfiles:
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
