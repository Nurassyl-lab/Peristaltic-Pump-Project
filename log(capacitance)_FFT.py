import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd #for excel files (xlsx)
from numpy.fft import fft 
from numpy.fft import fftshift
from numpy.fft import fftfreq
import scipy.signal as sci
import scipy.stats as stats


workbook = xlrd.open_workbook('device.xlsx', on_demand = True)

rpm_sup=[1,3,5,8,10,15,20]

# worksheet 2
# 2->1710

# worksheet 
# 2-> 670

# t_res=668
t_res=558

data=np.zeros((2,len(rpm_sup),t_res))

worksheet = workbook.sheet_by_index(2)
# first set workshop
for i in range(len(rpm_sup)):
    print("worksheet 2 - rpm",worksheet.cell_value(0,4*i))    
    data[0,i] = np.array([worksheet.cell_value(j,1+4*i) for j in range(1,t_res+1)])
    data[0,i] = data[0,i]-np.mean(data[0,i])

worksheet = workbook.sheet_by_index(3)
# second set workshop
for i in range(len(rpm_sup)):
    print("worksheet 3 - rpm",worksheet.cell_value(0,4*i))    
    data[1,i] =np.array([worksheet.cell_value(j,1+4*i) for j in range(1,t_res+1)])
    data[1,i] = data[1,i]-np.mean(data[1,i])

t_vec=[worksheet.cell_value(j,0) for j in range (1,t_res+1)]
    
# normalize the data
data=data/np.max(abs(data))

# plot all tha data
while False:
    plt.title("data recap: time evolution")    
    for j in range(len(rpm_sup)): 
        for i in range(2):
            # plt.plot(t_res,data[i,j],'rpm'+rpm_sup)
            s="rpm: "+str(rpm_sup[j])
            plt.plot(t_vec,data[i,j]+i+j/4,label=s)
    plt.legend()



# normalize the amplitude

plt.figure("time evolution")
plt.plot(t_vec,data_c)
plt.show()

# obtain the stfft




#print(workbook.nsheets)
#print(workbook.sheet_names())


worksheet = workbook.sheet_by_name('Sheet1')
plt.figure(9)
plt.title("LOG(|Capacitance|) FFT")

ar = []
time = [worksheet.cell_value(i, 0) for i in range (1709)]
capacitance_all = [worksheet.cell_value(i, 1) for i in range (1709)]

array1 = []
for i in range (1709):
    array1.append(capacitance_all[i] * pow(10, 12))

c = fft(array1)

c = abs(c)

c = np.log(1 + c)

timestep = 0.1
freq = np.fft.fftfreq(c.size, d=timestep)
plt.plot(freq, c) 
plt.show() 

plt.figure(10)
plt.title("LOG(|Capacitance|) FFT")
timestep = 0.1
freq = np.fft.fftfreq(c.size, d=timestep)
plt.plot(c) 
plt.show() 