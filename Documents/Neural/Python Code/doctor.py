import os
from shutil import copyfile
import re

os.chdir('/Users/brendan/Downloads')
with open('submission_failed.txt') as f:
    new_f = open('submission_copy3.txt', 'w+')
    new_f.write(f.readline())
    pix_seen = []
    line_ctr = 1
    for line in f.readlines():
        sp = line.split(',')
        new_f.write(sp[0] + ',')
        if line_ctr == 21:
            line_ctr = 1
            pix_seen = []
        eval_list = [eval(x) for x in sp[1].strip().split(' ')]
        for i in range(0, len(eval_list), 2):
            pix = eval_list[i]
            if pix == 0:
                new_f.write(str(pix) + ' ')
                new_f.write(str(eval_list[i + 1]) + ' ')
            elif pix not in pix_seen and pix > 0:
                pix_seen.append(pix)
                new_f.write(str(pix) + ' ')
                new_f.write(str(eval_list[i+1]) + ' ')
        line_ctr += 1
        new_f.write('\n')
    new_f.close()


