import matplotlib.pyplot as plt
import numpy as np
import xlrd
from numpy.fft import fft, fftshift, ifft
from scipy.signal import butter,filtfilt

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

workbook = xlrd.open_workbook('5RPM.xlsx', on_demand = True)
worksheet = workbook.sheet_by_name('Sheet1') 

#remove 8 comments below if you want to plot full signal at 5 RPM

# time = [worksheet.cell_value(i, 0) for i in range (679)]
# capacitance_all = [worksheet.cell_value(i, 1) for i in range (679)]
# plt.figure(1)
# plt.title("5 RPM")
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.plot(capacitance_all) 
# plt.show()

cap = []
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

cap = np.array(cap).flatten()#capacitance 

#scale the data
m = max(cap)
cap = cap / m

fs = 10#I thought that I need to choose fs as 0.1, however, when fs is 0.1
#there is an error ValueError: Digital filter critical frequencies must be 0 < Wn < 1
#so I have choosen fs 10

nyq = 0.5 * fs#The Nyquist frequency is half the sampling rate.

cutoff = 1.0#The "cutoff" means the cut frequency. Guessing
order = 2#The "order" determine the accuracy of filter. Guessing


# y = cap#comment this line if you want to plot the signal with noise
y = butter_lowpass_filter(cap, cutoff, fs, order)#comment this line if you want to plot signal without noise

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