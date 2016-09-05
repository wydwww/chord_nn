import os

f_list = os.listdir('../singleData')
for i in f_list:
    if os.path.splitext(i)[1] == '.asc':
        f = open('../singleData/'+i, 'r')
        if f.read(3) == 'nan':
            print i
