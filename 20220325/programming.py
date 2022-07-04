# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:23:22 2022

@author: weinien
"""


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab as p
import scipy.signal as sci    
from math import pi
import requests
from scipy.signal import savgol_filter
import shelve
import math
from sklearn.decomposition import NMF
from scipy.optimize import minimize
# from keras import backend as K
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# %matplotlib qt
colour = ['blue', 'orange', 'darkgreen', 'purple', 'lightgreen', 'lime']

order=[
    'water_pH9',
    'water_pH6.5',
    'pH9_pH6.5',
    'dark',
    'laser',
    'glass'
    ]

def remove_cosmic_rays(data, window):
       # '''
       #  Parameters
       #  ----------
       #  data
       #  window
       #  Returns
       #  -------
       #  '''
    data_out = np.copy(data)  # Copy the data to not modify original
    delta_data = np.abs(np.diff(data_out))  # Find the difference between consecutive points
    if data.ndim > 1:
        # If you feed the array of spectra, then the function will run recursively on each individual spectrum
        return np.apply_along_axis(func1d=remove_cosmic_rays, axis=-1, arr=data_out, window=window)
    else:
        # Find the outliers (outlier > 3 standard deviations)
        # + 1 for the correct index
        cosmic_ray_indices = np.where(delta_data > 3 * np.std(delta_data))[0] + 1
        for i in cosmic_ray_indices:
            w = np.arange(i - window, i + 1 + window)  # select 2*window + 1 points around spikes
            w2 = w[np.in1d(w, cosmic_ray_indices) == False]  # Select points apart from spikes
            arr = np.take(data, w2, mode='clip')  # Check if selected points raise out of bound, if yes-clip them
            data_out[i] = np.mean(arr)  # Substitute spike with the average of the selected points
        return data_out



# read files

database = np.array(pd.read_excel('water_pH9.xlsx'))
dataacid = np.array(pd.read_excel('water_pH6.5.xlsx'))
datamix = np.array(pd.read_excel('pH9_pH6.5.xlsx'))
datadark = np.array(pd.read_excel('dark.xlsx'))
datalaser = np.array(pd.read_excel('laser.xlsx'))
dataglass = np.array(pd.read_excel('glass.xlsx'))

#minimize function define
def f3(initial_guess,t_data,time_data_input):
    result = np.zeros(3119)
    for i in range(len(initial_guess)):
        result += np.dot(initial_guess[i],t_data[:,i]) 
    return np.linalg.norm(result-time_data_input)

# generate the zero matrix 
data = np.zeros((1024,2901))
mean_glass = np.zeros((1024,2905))
mean_laser = np.zeros((1024,2855))
mean_acid = np.zeros((1024,3620))
mean_base = np.zeros((1024,3647))
mean_mix = np.zeros((1024,3760))
datafix = np.zeros((1024,3619))
datafix1 = np.zeros((1024,2900))
dataNaclfix = np.zeros((1024,2900))
dataacidfix = np.zeros((1024,3620))
normalize_matrix = np.zeros((3619,1024))
coe = np.zeros(1024)

f_sup_px = database[0:1024,0]
f_sup=-13719+f_sup_px*21.42 -0.0048456*f_sup_px**2

plt.plot(dataacid[:,:100])
mean_glasses =np.sum(dataglass[:2905],axis = 1)/2903
mean_laser =np.sum(datalaser[:2855],axis = 1)/2897
mean_base =np.sum(database[:3647],axis = 1)/3637
mean_acid =np.sum(dataacid[:3620],axis = 1)/3613
mean_mix =np.sum(datamix[:3760],axis = 1)/3753
plt.figure("mean of glasses and laser")
plt.plot(mean_acid)
plt.plot(mean_glasses)
plt.plot(mean_laser)
max_place = np.argmax(mean_laser[0:100])
print(max_place)

min_place = 97

coe = (mean_glasses[max_place]- mean_glasses[min_place])/(mean_laser[max_place]-mean_laser[min_place])
print(coe)

fix_glasses = mean_glasses-coe*mean_laser
#normalize the spectrum
normalize_fix_glasses = (fix_glasses-min(fix_glasses))/(max(fix_glasses)-min(fix_glasses))
#normalize_mean_Nacl = (mean_Nacl-min(mean_Nacl))/(max(mean_Nacl)-min(mean_Nacl))
normalize_glasses = (mean_glasses-min(mean_glasses))/(max(mean_glasses)-min(mean_glasses))
plt.plot(normalize_fix_glasses)

n = 0
zero_place = []
for i in range(len(dataacid[0])):
    if dataacid[0,i] == 0 :
        n+=1
        zero_place.append(i)
        
dataacidfix = dataacid
for i in range(len(zero_place)):
    dataacidfix[:,zero_place[i]] = (dataacidfix[:,zero_place[i]-1]+dataacidfix[:,zero_place[i]+1])/2

dataacidfix = dataacidfix[:,1:]

result_innerprod = np.dot(dataacidfix.T,normalize_glasses)
result_innerprod = result_innerprod.reshape(3619,1)
normalize_glasses = normalize_glasses.reshape(1024,1)
normalize_fix_glasses = normalize_fix_glasses.reshape(1024,1)
matrix = np.dot(result_innerprod,normalize_glasses.T)

for i in range(3619):
    normalize_matrix[i,:] = (matrix[i,:] - min(matrix[i,:]))/(max(matrix[i,:]) - min(matrix[i,:]))
    #plt.plot(normalize_matrix[i,:])
normalize_matrix = normalize_matrix.T
plt.plot(normalize_matrix)


peak_place = np.argmax(mean_glasses[50:100])
print(peak_place)
min_place = np.argmin(mean_glasses[50:100])
print(min_place)

coe = (mean_acid[peak_place+50]-mean_acid[min_place+50])/(mean_glasses[peak_place+50]-mean_glasses[min_place+50])
print(coe)

laser_fix = (mean_laser - min(mean_laser))


dataacidfix_normailze = np.zeros((1024,3619))
for i in range(3619):
    datafix[:,i] = dataacidfix[:,i] - coe*mean_glasses
    
for i in range(3619):
    dataacidfix_normailze[:,i] = (dataacidfix[:,i]-min(dataacidfix[:,i]))/(max(dataacidfix[:,i]) - min(dataacidfix[:,i]))
    
for i in range(3619):
    datafix[:,i] = (datafix[:,i]-min(datafix[:,i]))/(max(datafix[:,i]) - min(datafix[:,i]))
plt.figure("data")
plt.plot(datafix)

plt.plot(np.mean(dataacidfix,axis = 1))
plt.plot(np.mean(normalize_matrix,axis = 1))

temp = datafix - (np.dot(datafix.T,normalize_fix_glasses)/np.linalg.norm(normalize_fix_glasses,1)*normalize_fix_glasses.T).T
plt.plot(temp)


concentration = np.ones(3619)
# plt.plot(concentration)
# =========salinatiy concentration====
for i in range(1,360):
    concentration[360+i] = (1*(359-i)+200*i)/359
    concentration[719+i] = (1*i+200*(359-i))/359
    concentration[1078+i] = (1*(359-i)+200*i)/359
    concentration[1437+i] = (1*i+200*(359-i))/359
    concentration[1796+i] = (1*(359-i)+200*i)/359
    concentration[2155+i] = (1*i+200*(359-i))/359

for i in range(1,420):
    concentration[2514+i] = (1*(419-i)+200*i)/419
    concentration[2933+i] = (1*i+200*(419-i))/419    
plt.plot(concentration)


concentration_normalization = (concentration-min(concentration))/(max(concentration) - min(concentration))
plt.plot(concentration_normalization)

# ============PH concentration======
ph_concentration = np.ones(3619)
# ph_concentration = 6.5
for i in range(3619):
    ph_concentration[i] = 6.5
for i in range(1,370):
    ph_concentration[375+i] = -math.log((((374-i)/374)*10**(-6.5)+(i/374)*10**(-7)),10)
    ph_concentration[749+i] = -math.log((10**(-6.5)*(i/374)+10**(-7)*((374-i)/374)),10)
    ph_concentration[1123+i] = -math.log((((374-i)/374)*10**(-6.5)+(i/374)*10**(-7)),10)
    ph_concentration[1497+i] = -math.log((10**(-6.5)*(i/374)+10**(-7)*((374-i)/374)),10)
    ph_concentration[1871+i] = -math.log((((374-i)/374)*10**(-6.5)+(i/374)*10**(-7)),10)
    ph_concentration[2245+i] = -math.log((10**(-6.5)*(i/374)+10**(-7)*((374-i)/374)),10)
    ph_concentration[2619+i] = -math.log((((374-i)/374)*10**(-6.5)+(i/374)*10**(-7)),10)
    ph_concentration[2993+i] = -math.log((10**(-6.5)*(i/374)+10**(-7)*((374-i)/374)),10)
plt.plot(ph_concentration)
plt.plot(concentration_normalization)

concentration_normalization_fix = concentration_normalization[500:]

ph_concentration_normailze = (ph_concentration-min(ph_concentration))/(max(ph_concentration)-min(ph_concentration))
plt.figure()
plt.plot(ph_concentration_normailze)
plt.plot(concentration_normalization)

plt.plot(concentration_normalization+ph_concentration_normailze)

datafix_2 = datafix[:,0:243]

plt.plot(datafix_2)


plt.plot(dataacid[:,0:243])

data_obser = dataacid[540:560,243:613]

peakplace1 = []
# for i in range(243):
peakplace1.append(np.argmax(data_obser,axis = 0))


data_obser2 = datafix[710:730,243:613]

peakplace2 = []
peakplace2.append(np.argmax(data_obser2,axis = 0))


# ===========PCA=============
n_components= 5
model = PCA(n_components)
W_PCA = model.fit_transform(datafix_2)
H_PCA = model.components_
t_dataN_PCA = H_PCA
f_dataN_PCA = W_PCA
plt.figure("nmf up:time low:frequency ", figsize=(15, 10))
plt.subplot(121)#, figsize = (15, 10))
for i in range(n_components):
    plt.plot((f_dataN_PCA[:,i]-min(f_dataN_PCA[:,i]))/np.mean(max(f_dataN_PCA[:,i])-min(f_dataN_PCA[:,i]))-i)
# plt.show()
plt.subplot(122)#, figsize=(15, 10))
for i in range(n_components):
    plt.plot(((t_dataN_PCA[i]-min(t_dataN_PCA[i]))/(max(t_dataN_PCA[i])-min(t_dataN_PCA[i]))-i))  
plt.show()


constant_signal = np.reshape(t_dataN_PCA[0],(1,3119))
# ============remove constant from data==
remove_constant_signal = datafix_2 - (np.dot(datafix_2,constant_signal.T)*constant_signal)
remove_constant_signal = remove_constant_signal
plt.plot(remove_constant_signal)
normailze_remove_constant_signal = np.zeros((1024,3119))
for i in range(3119):
    normailze_remove_constant_signal[:,i] = (remove_constant_signal[:,i] - min(remove_constant_signal[:,i]))/(max(remove_constant_signal[:,i]) - min(remove_constant_signal[:,i]))
plt.plot(normailze_remove_constant_signal,color = 'blue')
plt.plot(datafix_2,color = 'red')

plt.plot(normailze_remove_constant_signal)
# ================PCA of remove constant data
n_components= 5
model = PCA(n_components)
W_PCA1 = model.fit_transform(remove_constant_signal)
H_PCA1 = model.components_
t_dataN_PCA1 = H_PCA1
f_dataN_PCA1 = W_PCA1
plt.figure("nmf up:time low:frequency ", figsize=(15, 10))
plt.subplot(121)#, figsize = (15, 10))
for i in range(n_components):
    plt.plot((f_dataN_PCA1[:,i]-min(f_dataN_PCA1[:,i]))/np.mean(max(f_dataN_PCA1[:,i])-min(f_dataN_PCA1[:,i]))-i)
# plt.show()
plt.subplot(122)#, figsize=(15, 10))
for i in range(n_components):
    plt.plot(((t_dataN_PCA1[i]-min(t_dataN_PCA1[i]))/(max(t_dataN_PCA1[i])-min(t_dataN_PCA1[i]))-i))  
plt.show()



# ==========copare with setting time axis

concentration_axis = (t_dataN_PCA1[1]-min(t_dataN_PCA1[1]))/np.mean(max(t_dataN_PCA1[1])-min(t_dataN_PCA1[1]))
plt.figure()
plt.plot(concentration_axis)
plt.plot(concentration_normalization_fix)


# =====================smoooth PCA result

time_axis0 = savgol_filter(t_dataN_PCA1[0], 201, 3)

time_axis1 = savgol_filter(t_dataN_PCA1[1], 201, 3)
#plt.plot(time_axis1)
time_axis2 = savgol_filter(t_dataN_PCA1[2], 201, 3)
#plt.plot(time_axis2)
time_axis3 = savgol_filter(t_dataN_PCA1[3], 201, 3)
#plt.plot(time_axis3)
time_axis4 = savgol_filter(t_dataN_PCA1[4], 201, 3)
#plt.plot(time_axis4)
time_axis0_fix = (time_axis0-min(time_axis0))/(max(time_axis0)-min(time_axis0))
time_axis1_fix = (time_axis1-min(time_axis1))/(max(time_axis1)-min(time_axis1))
time_axis2_fix = (time_axis2-min(time_axis2))/(max(time_axis2)-min(time_axis2))
time_axis3_fix = (time_axis3-min(time_axis3))/(max(time_axis3)-min(time_axis3))
time_axis4_fix = (time_axis4-min(time_axis4))/(max(time_axis4)-min(time_axis4))

plt.plot(time_axis0_fix)
plt.plot(time_axis1_fix)
plt.plot(time_axis2_fix)
plt.plot(time_axis3_fix)
plt.plot(time_axis4_fix)

test_PCA_result = np.vstack((time_axis0, time_axis1,time_axis2,time_axis3,time_axis4))


# ================reconstruct by removed data

ans_salt = np.zeros(3119)
initial_guess_input = np.ones(2)
result_PCA = minimize(f3, initial_guess_input,args=(test_PCA_result[:2].T,concentration_normalization_fix),tol = 10**-12)

for i in range(len(initial_guess_input)):
    ans_salt+= result_PCA.x[i]*(test_PCA_result[i])
ans_salt = result_PCA.x.dot(test_PCA_result[:2])
labelsalt = (200*(ans_salt-min(ans_salt))/(max(ans_salt)-min(ans_salt)))

plt.plot(labelsalt)
plt.plot(200*concentration_normalization_fix)
plt.plot(-test_PCA_result[1]*20000/3)

# ==========
C_matrix = np.zeros((3,3))
for i in range(3):
    C_matrix[i][i] = result_PCA.x[i]
    
plt.plot(np.sum(C_matrix.dot(t_dataN_PCA1[:3]).T,axis=1))
x_pred = np.sum(C_matrix.dot(t_dataN_PCA1[:3]).T,axis=1)
x_pred = x_pred.reshape(1,3119)


Cinv = np.linalg.inv(C_matrix)
print(Cinv.dot(C_matrix))
plt.plot(f_dataN_PCA1[:,:3].dot(Cinv))
x_input = f_dataN_PCA1[:,:3].dot(Cinv)
print(x_input.shape)

testP = np.zeros(1024)
for i in range(3):
    testP +=x_input[:,i]
plt.plot(testP)    


# ========= test ========
train_label = labelsalt[:1000]
test_label  = labelsalt[1000:2900]

train_data = remove_constant_signal[:,:1000]
test_data = remove_constant_signal[:,1000:2900]

# ========== regression=====
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


modelLR=LinearRegression()
modelLR.fit(train_data.T,train_label)


Y_pred = modelLR.predict(test_data.T)
Y_pred.shape



Y_true = test_label#[:2000]
Y_true.shape
plt.plot(Y_pred)
plt.plot(Y_true)

print(mean_squared_error(Y_pred,Y_true))


# ========= testDNN ========
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=1024, kernel_initializer='glorot_uniform', 
activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(train_data.T,train_label,epochs = 700)


train_mse = model.evaluate(test_data.T, test_label)
print(train_mse)

DNN_predict = model.predict(test_data.T)
plt.plot(DNN_predict)
plt.plot(test_label)

# ======== randomize data=========
randomize_data = np.c_[remove_constant_signal.T,labelsalt]
np.random.shuffle(randomize_data)  

train_label_random = randomize_data[:2000,-1]
test_label_random  = randomize_data[2000:2900,-1]

train_data_random = randomize_data[:2000,:-1]
test_data_random = randomize_data[2000:2900,:-1]
# ========= testDNN with randomize data========
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=1024, kernel_initializer='glorot_uniform', 
activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

# model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(train_data_random,train_label_random,epochs = 1000)


train_mse = model.evaluate(test_data_random, test_label_random)
print(train_mse)

DNN_predict = model.predict(test_data.T)
plt.plot(DNN_predict)
plt.plot(test_label)


# =============================

randomize_dataN = np.c_[datafix_2.T,labelsalt]
np.random.shuffle(randomize_dataN)  

train_label_randomN = randomize_dataN[:1000,-1]
test_label_randomN  = randomize_dataN[1000:2900,-1]

train_data_randomN = randomize_dataN[:1000,:-1]
test_data_randomN = randomize_dataN[1000:2900,:-1]

# ========= testDNN with randomize dataN========
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(256, input_dim=1024, kernel_initializer='glorot_uniform', 
activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

# model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(train_data_randomN,train_label_randomN,epochs = 500)


train_mse = model.evaluate(test_data_randomN, test_label_randomN)
print(train_mse)

DNN_predict = model.predict(test_data.T)
plt.plot(DNN_predict)
plt.plot(test_label)


# ==============




# =========data process=====
smoo_tdata = np.zeros((5,3619))
for i in range(5):
    smoo_tdata[i] = savgol_filter(t_dataN_PCA[i],201,3)
    smoo_tdata[i] = (smoo_tdata[i]-min(smoo_tdata[i]))/(max(smoo_tdata[i])-min(smoo_tdata[i]))
plt.plot(smoo_tdata)

smoo_fdata = np.zeros((1024,5))
for i in range(5):
    smoo_fdata[:,i] = savgol_filter(f_dataN_PCA[:,i],201,3)
    smoo_fdata[:,i] = (smoo_fdata[:,i]-min(smoo_fdata[:,i]))/(max(smoo_fdata[:,i])-min(smoo_fdata[:,i]))
plt.plot(smoo_fdata)




ans_salt = np.zeros(3619)
initial_guess_input = np.ones(5)
result_PCA = minimize(f3, initial_guess_input,args=(t_dataN_PCA.T,concentration_normalization),tol = 10**-12)

for i in range(len(initial_guess_input)):
    ans_salt+= result_PCA.x[i]*(t_dataN_PCA[i,:])
ans_salt = result_PCA.x.dot(t_dataN_PCA)
labelsalt = (200*(ans_salt-min(ans_salt))/(max(ans_salt)-min(ans_salt)))
plt.plot(labelsalt)
plt.plot(concentration)

ans_ph = np.zeros(3619)
initial_guess_input = np.ones(5)
result_PCA = minimize(f3, initial_guess_input,args=(t_dataN_PCA.T,ph_concentration_normailze),tol = 10**-12)

for i in range(len(initial_guess_input)):
    ans_ph+= result_PCA.x[i]*(t_dataN_PCA[i,:])
ans_ph = result_PCA.x.dot(t_dataN_PCA)
labelph = ((ans_ph-min(ans_ph))/(max(ans_ph)-min(ans_ph)))
plt.plot(labelph)
plt.plot(ph_concentration_normailze,color='red')
plt.plot(concentration_normalization,color = 'blue')

# ==============projection===============
proj1 = (np.dot(datafix,concentration_normalization))#/np.linalg.norm(datafix,1)*temp)
plt.figure("projectoin with salinity concentration")
plt.plot(proj1)
proj2 = (np.dot(datafix,labelph))#/np.linalg.norm(datafix,1)*temp)
plt.figure("projectoin with ph concentration")

plt.plot(proj1)
plt.plot(proj2)
# ==========NMF===============
n_components= 5
model = NMF(n_components)
W = model.fit_transform(datafix)
H = model.components_
t_dataN = W
f_dataN = H
plt.figure("nmf up:time low:frequency ", figsize=(15, 10))
plt.subplot(121)#, figsize = (15, 10))
for i in range(n_components):
    plt.plot((t_dataN[:,i]-min(t_dataN[:,i]))/np.mean(max(t_dataN[:,i])-min(t_dataN[:,i]))-i,colour[i])
# plt.show()
plt.subplot(122)#, figsize=(15, 10))
for i in range(n_components):
    plt.plot(((f_dataN[i]-min(f_dataN[i]))/(max(f_dataN[i])-min(f_dataN[i]))-i),colour[i])  
plt.show()