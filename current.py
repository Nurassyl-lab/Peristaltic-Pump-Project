import matplotlib.pyplot as plt
import numpy as np
import xlrd
from numpy.fft import fft, fftshift
from scipy.signal import butter,filtfilt
from scipy import signal

# def butter_lowpass_filter(data, cutoff, fs, order):
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients 
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

workbook = xlrd.open_workbook('5RPM.xlsx', on_demand = True)
worksheet = workbook.sheet_by_name('Sheet1') 

time = [worksheet.cell_value(i, 0) for i in range (679)]
capacitance_all = [worksheet.cell_value(i, 1) for i in range (679)]

cap = []

# plt.figure(1)
# plt.title("5 RPM")
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.plot(capacitance_all) 
# plt.show()

#here I'm making a window(24sec - 34sec)
# time = [worksheet.cell_value(i, 0) for i in range (220, 330, 1)]

#windows
#I did it very stupid, because I can't get used to for loop in python

#1

capacitance_all = [worksheet.cell_value(i, 1) for i in range (220, 240, 1)]#20
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#2
capacitance_all = [worksheet.cell_value(i, 1) for i in range (235, 255, 1)]#20
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#3
capacitance_all = [worksheet.cell_value(i, 1) for i in range (250, 270, 1)]#20
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#4
capacitance_all = [worksheet.cell_value(i, 1) for i in range (265, 285, 1)]#20
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#5
capacitance_all = [worksheet.cell_value(i, 1) for i in range (280, 300, 1)]#20
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#6
capacitance_all = [worksheet.cell_value(i, 1) for i in range (295, 315, 1)]#20
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#7
capacitance_all = [worksheet.cell_value(i, 1) for i in range (310, 330, 1)]#20
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all) 
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#FFT
cap = np.array(cap).flatten()
m = max(cap)
cap = cap / m
cap = fft(cap)
cap = fftshift(cap)
cap = abs(cap)

plt.figure(1)
plt.title("FFT")
plt.ylabel('Power')
plt.xlabel('Frequency')
plt.plot(cap) 
plt.show()

#working on low pass filter