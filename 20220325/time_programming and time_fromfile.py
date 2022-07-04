# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:05:16 2022

@author: weinien
"""

import numpy as np
import matplotlib.pyplot as plt
import os     
import time
from datetime import datetime
plt.close('all')
res=1024
dim=2901


data=np.zeros([6,dim,res])
file_name=np.chararray(6,dim,res)

file_name[0]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/water_pH9NaCl/test_V78500241_"
file_name[1]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/water_pH6.5NaCl/test_V78500241_"
file_name[2]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/pH9NaCl_pH6.5NaCl/test_V78500241_"
file_name[3]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/dark/test_V78500241_"
file_name[4]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/laser/test_V78500241_"
file_name[5]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/glass/glass_V78500241_"



file_time0 = []
for i_file in range(3721):
    file_name_0=file_name[0]+"%d.txt" % i_file
    if os.path.exists(file_name_0):
        time_data = os.path.getmtime(file_name_0)
        file_time0.append(time_data)


file_time0_fix = []
for i in range(3635):
    fix_time = datetime.fromtimestamp(file_time0[i]).strftime('%Y-%m-%d %H:%M:%S')
    file_time0_fix.append(fix_time)
    
    

file_time2 = []
for i_file in range(3760):
    file_name_2=file_name[2]+"%d.txt" % i_file
    if os.path.exists(file_name_2):
        time_data = os.path.getmtime(file_name_2)
        file_time2.append(time_data)


file_time2_fix = []
for i in range(3752):
    fix_time = datetime.fromtimestamp(file_time2[i]).strftime('%Y-%m-%d %H:%M:%S')
    file_time2_fix.append(fix_time)
    
print(file_time2_fix[0])

file_time2_min_sec =[]
for i in range(3752):
    sec = file_time2_fix[i][-2:]
    minute = file_time2_fix[i][-5:-3]
    hour = file_time2_fix[i][-8:-6]
    file_time2_min_sec.append([hour,minute,sec])

file_time2_programming =[]
for i_file in range(3760):
    file_name_2=file_name[2]+"%d.txt" % i_file
    if os.path.exists(file_name_2):
        with open(file_name_2) as f:
            temp = f.readlines()[2]
            # temp=[i.strip('\n') for i in temp]
            hour = int(temp[17:19])
            minute = int(temp[20:22])
            sec = int(temp[23:25])
            file_time2_programming.append([hour , minute , sec])
            

    
