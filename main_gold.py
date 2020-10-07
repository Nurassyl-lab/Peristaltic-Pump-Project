"Gas Sensors"
#-----------------------------------------------------------------------------
import matplotlib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import math 
pi = math.pi 
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
#-----------------------------------------------------------------------------
def find_anomalies(random_data):
    anomalies = []
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def matrix_lstsqr(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

def classic_lstsqr(x_list, y_list):
    N = len(x_list)
    x_avg = sum(x_list)/N
    y_avg = sum(y_list)/N
    var_x, cov_xy = 0, 0
    for x,y in zip(x_list, y_list):
        temp = x - x_avg
        var_x += temp**2
        cov_xy += temp * (y - y_avg)
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)

def mse_loss(beta,X, Y,function):#found
    return np.sum((function(X,beta)-Y)**2)/Y.size

def func(x, A, B, x0, sigma):
    return A+B*np.tanh((x-x0)/sigma)    

def init_cos3(x_data, y_data):
    y_smoth = y_data    
    l_win_low  = np.argmax(y_smoth[0] < y_smoth)
    r_win_low =x_data.size- np.argmax(y_smoth[-1] < np.flip(y_smoth))   
    M=np.max(y_data[l_win_low:r_win_low])
    peak_pos=l_win_low+np.argmax(y_data[l_win_low:r_win_low])   
    L_int=y_smoth[0]
    R_int=y_smoth[-1]    
    beta_cos2_int=[L_int, M, R_int, l_win_low, peak_pos-l_win_low, r_win_low- peak_pos]   
    return beta_cos2_int   

def cos_window2(t,L,M,R,o,al,be):
    if t<o:
        out=L    
    elif o<=t & t<o+al:
        out=L+(M-L)/2*(1-np.cos(2*pi*(t-o)/(2*al)))
    elif o+al<= t & t<o+al+be:
        out=R+(M-R)/2*(1-np.cos(2*pi*(be-(t-o-al))/(2*be)))
    elif t>=o+al+be:
        out=R
    else:
        out=0
    return  out

def cos_win2(x_data,beta):#found
    L=beta[0]
    M=beta[1]
    R=beta[2]
    o=beta[3]    
    al=beta[4]
    be=beta[5]
    yvec=[cos_window2(t,L,M,R,o,al,be)  for t in range(x_data.size)]
    return yvec
#-----------------------------------------------------------------------------
Fs = 10
f = 2    
sample = 1000
fs = 100
nyq = 0.5 * fs
order = 5
cutoff = 3

n = 20

final_dataX = []
final_dataY = []
colors = ["r", "b", "g"]

cap = [0]
cap = np.array(cap)
colorr = ["red", "black", "yellow", "orange", "green", "aqua"]
c = 0
num_iter = 1
name = ["NTU", "NCU_11.87k_2", "NCU_11.87K", "NCU_10.95K", "NCU_8.12K", "NCU_7.48K" , "NCU_5.06K"]
colorrr = ["red", "purple", "yellow", "orange", "blue", "aqua", "pink"]
#-----------------------------------------------------------------------------
col = 2
beta_coordianates = []
n = 1
my_col = 0
while col < 38:
    time = pd.read_csv("gold2.csv", usecols=["T" + str(col)]) 
    time = np.array(time)
    capacitance = pd.read_csv("gold2.csv", usecols=["C"+ str(col)])
    capacitance = np.array(capacitance)
    plt.figure(1)
    plt.title("Gas sensor measurements")  
    plt.plot(time, capacitance, label=name[my_col], color=colorrr[my_col])#, color=colorr[c])
    plt.legend()
    plt.show()
#-----------------------------------------------------------------------------
#Peaks    
    plt.figure(2)
    plt.title("Scale")
    capacitance = capacitance / np.max(np.abs(capacitance))
    capacitance = abs(capacitance)
    
    plt.plot(time, capacitance)
    
    capacitance = capacitance.flatten()
    time = time.flatten()
    
    plt.plot(time, capacitance, label=name[my_col], color=colorrr[my_col])
    plt.legend() 
#==================================FITTING====================================
    beta_init = init_cos3 
    fit_fun= cos_win2
    loss_peak = mse_loss
    bounds_cos2_vec = np.array([np.zeros(6) , np.inf*np.ones(6)]).T
    bounds_cos2 = tuple(map(tuple, bounds_cos2_vec))
    
    "___Error in minimize function bounds___"
    result = minimize(loss_peak, beta_init(time, capacitance), args = (time, capacitance , fit_fun), tol=1e-12, bounds = bounds_cos2, method='Nelder-Mead')                  
    beta_hat_fit = result.x
    
    plt.figure(3)
    plt.plot(time, fit_fun(time, beta_hat_fit), label = name[my_col], color = colorrr[my_col])
    plt.legend()
    
    print(num_iter)
    
    num_iter += 1
    resY = beta_hat_fit[1] * (beta_hat_fit[4]/2 + beta_hat_fit[5]/2)
    resX = beta_hat_fit[2] - beta_hat_fit[0]
    
    final_dataX.append(resX)
    final_dataY.append(resY)
    
    beta_coordianates.append(beta_hat_fit)
#-----------------------------------------------------------------------------
    if (col - 2) % 5 == 0 and col != 2:
        my_col += 1
    plt.legend()
    col += 1
#-----------------------------------------------------------------------------
plt.figure(4)
my_col = 0
for i in range(36):
    plt.scatter(final_dataX[i], final_dataY[i], label = name[my_col], color = colorrr[my_col], marker = 'x')
    if (i % 5) == 0 and i != 0:
        my_col += 1
plt.legend()
#-----------------------------------------------------------------------------
ar_x = []
ar_y = []
#-----------------------------------------------------------------------------
an = find_anomalies(final_dataY)
indx = []
index = 0
for i in range(len(final_dataY)):
    for j in range (len(an)):
        if final_dataY[i] == an[j]:
            indx.append(index)
    index += 1
    
d_ind = 0
for i in range(len(indx)):
    final_dataY.pop(indx[i] - d_ind)
    final_dataX.pop(indx[i] - d_ind)
    d_ind += 1
    
my_col = 0
n = 0
b = 4

for g in range (7):
    ar_x = [0] * 4
    ar_y = [0] * 4
    j = 0
    for i in range (n, b, 1):
         ar_x[j] = final_dataX[i]
         ar_y[j] = final_dataY[i]
         j += 1
         
    n += 5
    b += 5
    alpha = 0.1
    lasso = Lasso(alpha=alpha)
    an = find_anomalies(ar_y)
    ar_x = np.array(ar_x)
    ar_y = np.array(ar_y)
    ar_x = np.reshape(ar_x, (-1, 2))
    ar_y = np.reshape(ar_y, (-1, 2))
    y_predicted = lasso.fit(ar_y, ar_x).predict(ar_y)
    plt.figure(5)
    plt.scatter(ar_y, y_predicted, label = name[my_col], marker="x" )
    ar_y, y_predicted = ar_y.reshape(-1,1), y_predicted.reshape(-1,1)
    ar_x = ar_x.reshape(-1, 1)
    plt.plot(ar_y, lasso.fit(ar_y, y_predicted).predict(ar_y))
    plt.legend()
    plt.show()
    my_col += 1
#-----------------------------------------------------------------------------
Mid = beta_coordianates
Mid = np.array(Mid)
scaler = StandardScaler()
Mid_test = scaler.fit_transform(Mid)
covariance = np.cov(Mid, rowvar = False)
eig_vals, eig_vecs = np.linalg.eig(covariance)
print("6D")
print("==============================")
print(eig_vecs.T)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

scree = [0] * 6
for i in range (6):
    scree[i] = (eig_vals[i] / sum(eig_vals))*100  

#CHOOSE DIMENSION ACCORDING TO EIGENVALUES
dimension_choose = 2
length_of_covariance_matrix = len(eig_vecs)

for i in range(length_of_covariance_matrix - dimension_choose):
    eig_vecs = np.delete(eig_vecs, dimension_choose, 1)
    
eig_vecs = eig_vecs.T
Mid_test = Mid_test.T
Final_PCA = np.matmul(eig_vecs, Mid_test).T
Final_PCA = Final_PCA.T
my_col = 0
for i in range (36):
    plt.figure(7)
    plt.title("2D Graph")
    plt.scatter(Final_PCA[i][0], Final_PCA[i][1], label=name[my_col], color=colorrr[my_col])
    if (i % 5) == 0 and i != 0:
        my_col += 1
plt.legend()

num_vars = 6
eigvals = eig_vals
#-----------------------------------------------------------------------------
fig = plt.figure(6, figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, scree, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component (eigenvalues)')
plt.ylabel('Percentage of how eigenvalues affect the result')

leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                  shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                  markerscale=0.4)

leg.get_frame().set_alpha(0.4)
plt.show()
print("2D")
print("==============================")
print(eig_vecs)