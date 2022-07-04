# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:27:48 2022

@author: weinien
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os     
plt.close('all')
res=1024
dim=2901

writer1 = pd.ExcelWriter('water_pH9.xlsx')
writer2 = pd.ExcelWriter('water_pH6.5.xlsx')
writer3 = pd.ExcelWriter('pH9_pH6.5.xlsx')
writer4 = pd.ExcelWriter('dark.xlsx')
writer5 = pd.ExcelWriter('laser.xlsx')
writer6 = pd.ExcelWriter('glass.xlsx')

data=np.zeros([6,dim,res])
file_name=np.chararray(6,dim,res)

file_name[0]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/water_pH9NaCl/test_V78500241_"
file_name[1]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/water_pH6.5NaCl/test_V78500241_"
file_name[2]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/pH9NaCl_pH6.5NaCl/test_V78500241_"
file_name[3]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/dark/test_V78500241_"
file_name[4]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/laser/test_V78500241_"
file_name[5]="C:/Users/weinien/Desktop/NYCU/2022 spring/Data/20220325/glass/glass_V78500241_"

data0 = np.zeros([3646,1024])
for i_file in range(75,3721):
    file_name_0=file_name[0]+"%d.txt" % i_file
    if os.path.exists(file_name_0):    
        with open(file_name_0) as f:
            temp = f.readlines()[-1024:]
            temp=[i.strip('\n') for i in temp]
            # temp = np.array(temp)
            row = [float(row.split('\t')[0])for row in temp]
            data0[i_file-75]=[float(row.split('\t')[1]) for row in temp]

data1 = np.zeros([3619,1024])
for i_file in range(1,3620):
    file_name_1=file_name[1]+"%d.txt" % i_file
    if os.path.exists(file_name_1):    
        with open(file_name_1) as f:
            temp = f.readlines()[-1024:]
            temp=[i.strip('\n') for i in temp]
            # temp = np.array(temp)
            row = [float(row.split('\t')[0])for row in temp]
            data1[i_file-1]=[float(row.split('\t')[1]) for row in temp]
            
data2 = np.zeros([3759,1024])
for i_file in range(1,3760):
    file_name_2=file_name[2]+"%d.txt" % i_file
    if os.path.exists(file_name_2):    
        with open(file_name_2) as f:
            temp = f.readlines()[-1024:]
            temp=[i.strip('\n') for i in temp]
            # temp = np.array(temp)
            row = [float(row.split('\t')[0])for row in temp]
            data2[i_file-1]=[float(row.split('\t')[1]) for row in temp]
            
data3 = np.zeros([2060,1024])
for i_file in range(1,2061):
    file_name_3=file_name[3]+"%d.txt" % i_file
    if os.path.exists(file_name_3):    
        with open(file_name_3) as f:
            temp = f.readlines()[-1024:]
            temp=[i.strip('\n') for i in temp]
            # temp = np.array(temp)
            row = [float(row.split('\t')[0])for row in temp]
            data3[i_file-1]=[float(row.split('\t')[1]) for row in temp]
            
data4 = np.zeros([2854,1024])
for i_file in range(1,2855):
    file_name_4=file_name[4]+"%d.txt" % i_file
    if os.path.exists(file_name_4):    
        with open(file_name_4) as f:
            temp = f.readlines()[-1024:]
            temp=[i.strip('\n') for i in temp]
            # temp = np.array(temp)
            row = [float(row.split('\t')[0])for row in temp]
            data4[i_file-1]=[float(row.split('\t')[1]) for row in temp]
            
data5 = np.zeros([2904,1024])
for i_file in range(1,2905):
    file_name_5=file_name[5]+"%d.txt" % i_file
    if os.path.exists(file_name_5):    
        with open(file_name_5) as f:
            temp = f.readlines()[-1024:]
            temp=[i.strip('\n') for i in temp]
            # temp = np.array(temp)
            row = [float(row.split('\t')[0])for row in temp]
            data5[i_file-1]=[float(row.split('\t')[1]) for row in temp]



data_tmp = pd.DataFrame(np.array(data0[0,:]),index=row)
data_tmp1 = pd.DataFrame(np.array(data1[0,:]),index=row)
data_tmp2 = pd.DataFrame(np.array(data2[0,:]),index=row)
data_tmp3 = pd.DataFrame(np.array(data3[0,:]),index=row)
data_tmp4 = pd.DataFrame(np.array(data4[0,:]),index=row)
data_tmp5 = pd.DataFrame(np.array(data5[0,:]),index=row)


for i in range(1,3646):
    df = pd.DataFrame(np.array(data0[i,:]))
    data_tmp.insert(i,i,np.array(df))

for i in range(1,3619):
    df1 = pd.DataFrame(np.array(data1[i,:]))
    data_tmp1.insert(i,i,np.array(df1))

for i in range(1,3759):
    df2 = pd.DataFrame(np.array(data2[i,:]))
    data_tmp2.insert(i,i,np.array(df2))

for i in range(1,2060):
    df3 = pd.DataFrame(np.array(data3[i,:]))
    data_tmp3.insert(i,i,np.array(df3))
    
for i in range(1,2854):
    df4 = pd.DataFrame(np.array(data4[i,:]))
    data_tmp4.insert(i,i,np.array(df4))
    
for i in range(1,2904):
    df5 = pd.DataFrame(np.array(data5[i,:]))
    data_tmp5.insert(i,i,np.array(df5))
# data_tmp = pd.DataFrame.append(data_tmp,df)
data_tmp.to_excel(writer1)
data_tmp1.to_excel(writer2)
data_tmp2.to_excel(writer3)
data_tmp3.to_excel(writer4)
data_tmp4.to_excel(writer5)
data_tmp5.to_excel(writer6)


writer1.close() 
writer2.close()    
writer3.close()    
writer4.close()   
writer5.close()    
writer6.close()

