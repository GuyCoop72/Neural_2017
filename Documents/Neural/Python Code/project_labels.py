import os
from Data_Handling import data_handler as dh

#########Generate txt files###########
with open ('train.txt', 'w') as f:
    for file in os.listdir("./train"):
        if file.endswith(".jpg"):
            file_path = os.path.abspath(file)
            f.write(file_path + '\n')

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

NUM_OF_LINES=file_len('train.txt') / 2
#print NUM_OF_LINES
filename = 'train.txt'
with open(filename) as fin:
    fout = open("val.txt","wb")
    for i,line in enumerate(fin):
      fout.write(line)
      
      if (i+1)%NUM_OF_LINES == 0:
        fout.close()
        fout = open("output%d.txt"%(i/NUM_OF_LINES+1),"wb")

    fout.close()

with open(filename, 'r') as fin:
    data = fin.read().splitlines(True)
with open(filename, 'w') as fout:
    fout.writelines(data[NUM_OF_LINES:]);

with open ('test.txt', 'w') as f_test:
    for file in os.listdir("./test"):
        if file.endswith(".jpg"):
            file_path_test = os.path.abspath(file)
            f_test.write(file_path_test + '\n')

##########Generate Labels##########
for file in os.listdir("./train"):
    if file.endswith(".txt"):
        file_path = os.path.abspath(file)
        dh.get_yolo_text_files(file_path + file, file_path + file)





