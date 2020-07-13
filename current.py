import matplotlib.pyplot as plt
import numpy as np
import xlrd
from numpy.fft import fft, fftshift, ifft
from scipy.signal import butter,filtfilt
from scipy import signal

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y



# def butter_lowpass_filter(data, cutoff, fs, order):
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients 
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

workbook = xlrd.open_workbook('5RPM.xlsx', on_demand = True)
worksheet = workbook.sheet_by_name('Sheet1') 

# time = [worksheet.cell_value(i, 0) for i in range (679)]
# capacitance_all = [worksheet.cell_value(i, 1) for i in range (679)]

cap = []

# plt.figure(1)
# plt.title("5 RPM")
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.plot(capacitance_all) 
# plt.show()

#here I'm making a window(24sec - 34sec)
time = [worksheet.cell_value(i, 0) for i in range (220, 430, 1)]
#windows
#I did it very stupid, because I can't get used to for loop in python

#1

capacitance_all = [worksheet.cell_value(i, 1) for i in range (220, 250, 1)]
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#2
capacitance_all = [worksheet.cell_value(i, 1) for i in range (250, 280, 1)]
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#3
capacitance_all = [worksheet.cell_value(i, 1) for i in range (280, 310, 1)]
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#4
capacitance_all = [worksheet.cell_value(i, 1) for i in range (310, 340, 1)]
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#5
capacitance_all = [worksheet.cell_value(i, 1) for i in range (340, 370, 1)]
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#6
capacitance_all = [worksheet.cell_value(i, 1) for i in range (370, 400, 1)]
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all)
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

#7
capacitance_all = [worksheet.cell_value(i, 1) for i in range (400, 430, 1)]
capacitance_all = np.array(capacitance_all)
capacitance_all = np.array(capacitance_all) 
mean = np.mean(capacitance_all)
capacitance_all = capacitance_all - mean
cap.append(capacitance_all)
mean = 0

cap = np.array(cap).flatten()
#scaled the data
m = max(cap)
cap = cap / m
# plt.plot(time, cap)

fs = 10
nyq = 0.5 * fs
cutoff = 1.0
order = 2
y = butter_lowpass_filter(cap, cutoff, fs, order)

#Before FFT
plt.figure(0)
plt.title("Time domain")
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.plot(time, y) 
plt.show()

#FFT
y = fft(y)
y = fftshift(y)
y = abs(y)
plt.figure(1)
plt.title("FFT")
plt.ylabel('Power')
plt.xlabel('Frequency')
plt.plot(y) 
plt.show()

plt.figure(2)
plt.title("IFFT (impuls response LTI) h(t)")#if the x(y) (input) is impuls, out will be impulse response h(t)
y = ifft(y)
# plt.ylabel('Power')
plt.xlabel('TIME')
plt.plot(y) 
plt.show()
#working on low pass filter