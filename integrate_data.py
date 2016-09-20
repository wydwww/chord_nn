from os import listdir
from os.path import isfile, join
import re

mypath = './data/'
allfiles = [f for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith('asc')]

count = 1
f = open('./alldata.asc', 'w')

for file_name in allfiles:
    # print file_name.split('-')[0]
    label = file_name.split('-')[0]
    f.write(label + ' ')
    ff = open('./data/' + file_name, 'r')
    for line in ff.read().split('\n'):
        f.write(line + ' ')
    # f.write(ff.read().split(''))
    ff.close()
    f.write('\n')

f.close()
