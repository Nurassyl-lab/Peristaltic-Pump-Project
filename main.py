import matplotlib.pyplot as plt
import numpy as np
import xlrd
from numpy.fft import fft, fftshift
from scipy.signal import butter,filtfilt, find_peaks
#-----------------------------------------------------------------------------
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
#-----------------------------------------------------------------------------
workbook = xlrd.open_workbook('DATA_FILE.xlsx', on_demand = True)
worksheet = workbook.sheet_by_name('Sheet1') 
#-----------------------------------------------------------------------------
"SELECT RPM: 0(1RPM), 1(3RPM), 2(5RPM), 3(8RPM), 4(10RPM), 5(15RPM), 6(20RPM)"
# SELECT = int(input())
SELECT = 0
while SELECT < 7:
    #-------------------------------------------------------------------------
    RPM_s = [0] * 7
    RPM_f = [0] * 7
    window = [0] * 7
    for_loop = [0] * 7
    s = [0] * 7
    c_col = [0] * 7
    cutoff = [0] * 7
    rps = [0] * 7
    rpm = [0] * 7
    #-------------------------------------------------------------------------
    #1RPM
    RPM_s[0] = 2
    RPM_f[0] = 602
    window[0] = 150
    for_loop[0] = 4
    s[0] = 1
    c_col[0] = 1
    cutoff[0] = 0.15
    rps[0] = 0.016
    #3RPM---------------------------------------------------------------------
    RPM_s[1] = 2
    RPM_f[1] = 389
    window[1] = 43
    for_loop[1] = 9
    s[1] = 3
    c_col[1] = 3
    cutoff[1] = 0.35
    rps[1] = 0.05
    #5RPM---------------------------------------------------------------------
    RPM_s[2] = 2
    RPM_f[2] = 389
    window[2] = 43
    for_loop[2] = 9
    s[2] = 5
    c_col[2] = 5
    cutoff[2] = 0.5
    rps[2] = 0.083
    #8RPM---------------------------------------------------------------------
    RPM_s[3] = 2
    RPM_f[3] = 162
    window[3] = 16
    for_loop[3] = 10
    s[3] = 8
    c_col[3] = 7
    cutoff[3] = 1
    rps[3] = 0.13
    #10RPM--------------------------------------------------------------------
    RPM_s[4] = 2
    RPM_f[4] = 132
    window[4] = 13
    for_loop[4] = 10
    s[4] = 10
    c_col[4] = 9
    cutoff[4] = 1
    rps[4] = 0.16
    #15RPM--------------------------------------------------------------------
    RPM_s[5] = 90
    RPM_f[5] = 270
    window[5] = 9
    for_loop[5] = 20
    s[5] = 15
    c_col[5] = 11
    cutoff[5] = 1.5
    rps[5] = 0.25
    #20RPM--------------------------------------------------------------------
    RPM_s[6] = 2
    RPM_f[6] = 62
    window[6] = 6
    for_loop[6] = 10
    s[6] = 20
    c_col[6] = 13
    cutoff[6] = 3
    rps[6] = 0.33
    
    t_col = c_col[SELECT] - 1
    #-------------------------------------------------------------------------
    time = [worksheet.cell_value(i, t_col) for i in range (RPM_s[SELECT], RPM_f[SELECT], 1)]
    time = np.array(time)
    capacitance_all = [worksheet.cell_value(i, c_col[SELECT]) for i in range (RPM_s[SELECT], RPM_f[SELECT], 1)]
    cap_to_comp = capacitance_all
    #-------------------------------------------------------------------------
    cap = []
    Fs = 10
    f = 2    
    sample = 1000
    fs = 10
    nyq = 0.5 * fs
    order = 2
    #-------------------------------------------------------------------------
    n = RPM_s[SELECT]
    b = n + window[SELECT]
    
    for i in range (for_loop[SELECT]):
        capacitance_all = [worksheet.cell_value(i, c_col[SELECT]) for i in range (n, b, 1)]
        capacitance_all = np.array(capacitance_all)
        mean = np.mean(capacitance_all)
        capacitance_all = capacitance_all - mean
        cap.append(capacitance_all)
        mean = 0
        n+=window[SELECT]
        b+=window[SELECT]
    
    cap = np.array(cap).flatten()
    cap = cap/np.max(np.abs(cap))
    y_comp = cap
    y = butter_lowpass_filter(cap, cutoff[SELECT], fs, order)
    peaks, _ = find_peaks(y)
    #=========================================================================
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title(str(s[SELECT]) + "RPM")
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.plot(time, cap_to_comp) 
    plt.show()
    #-------------------------------------------------------------------------
    plt.subplot(2, 2, 2)
    plt.title(str(s[SELECT]) + "RPM (Scaled and Denoised)")
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.plot(time, cap) 
    plt.plot(time, y)
    plt.show()
    # ------------------------------------------------------------------------
    plt.subplot(2, 2, 3)
    # plt.title(str(s[SELECT]) + "RPM peaks")
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.plot(time, y)
    plt.plot(time[peaks], y[peaks], "xr")
    plt.show()
    deltap = time[peaks][1:] - time[peaks][0:-1]
    p_mean = np.mean(deltap)
    p_var = np.var(deltap)
    print(str(s[SELECT]) + "RPM")
    print("MEAN of 1/4 rotation: " + str(p_mean) + "(sec)", " VARIANCE: " + str(p_var) + "(sec)" + " ERROR: " + str(abs((15 /s[SELECT]) - p_mean)) + "(sec)")
    print("MEAN of 1/4 rotation: " + str(((1/4 * (15/s[SELECT]))/p_mean)) + "(RPM)", "VARIANCE: " + str(((p_var * 1/4)/p_mean)) + "(RPM)" + " ERROR: " + str(abs(((1/4 * (15/s[SELECT]))/p_mean) - 0.25)) + "(RPM)")
    print("Real RPM: " + str(60 / (p_mean * 4)) + "(RPM)")
    print("=================================================================")
    #-------------------------------------------------------------------------
    y_comp = fft(y_comp)
    y_comp = fftshift(y_comp)
    y_comp = abs(y_comp)
    y = fft(y)
    y = fftshift(y)
    y = abs(y) 
    plt.subplot(2, 2, 4)
    # plt.title(str(s[SELECT]) + " FFT")
    plt.ylabel('Power')
    plt.xlabel('Frequency')
    plt.plot(y_comp)
    plt.plot(y) 
    plt.savefig(str(s[SELECT]) + " RPM")
    #-------------------------------------------------------------------------
    plt.figure(s[SELECT] + 10)
    plt.title(str(s[SELECT]) + 'RPM')
    plt.hist(deltap)
    con_interval_lo = p_mean - ((1.96 * p_var)/np.sqrt(100))
    con_interval_up = p_mean + ((1.96 * p_var)/np.sqrt(100))
    plt.axvline(x=p_mean, label="MEAN: " + str(p_mean), color="red")
    plt.legend()
    plt.savefig(str(s[SELECT]) + " RPM (histogram)")
    SELECT += 1