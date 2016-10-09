import os

f_list = os.listdir('../c3c4d4asc')
print "nan files:"
for i in f_list:
    if os.path.splitext(i)[1] == '.asc':
        f = open('../c3c4d4asc/'+i, 'r')
        if f.read(3) == 'nan':
            print i
