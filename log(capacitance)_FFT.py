import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd #for excel files (xlsx)
from numpy.fft import fft 
from numpy.fft import fftshift
from numpy.fft import fftfreq
workbook = xlrd.open_workbook('data.xlsx', on_demand = True)
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