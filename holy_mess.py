"GOLDEN NANO PARTICLES"
#-----------------------------------------------------------------------------

"NOTES"
#eigen vectors

#-----------------------------------------------------------------------------
# from header_of_functions import COLO
import matplotlib.pyplot as plt
# import matplotlib._color_data as mcd
import numpy as np
# from numpy.fft import fft, fftshift
import pandas as pd
from scipy.signal import butter,filtfilt, find_peaks
import math 
pi = math.pi 
# from scipy.interpolate import interp1d
from scipy.optimize import minimize
# from itertools import cycle
from sklearn.linear_model import Lasso
# from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
def find_anomalies(random_data):
    anomalies = []
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    print(lower_limit)
    # Generate outliers
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
def find_anomalies(random_data):
    anomalies = []
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    print(lower_limit)
    # Generate outliers
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
    # plt.figure(1)
    # plt.title("Gas sensor measurements")  
    # plt.plot(time, capacitance, label=name[my_col], color=colorrr[my_col])#, color=colorr[c])
    # plt.legend()
    # plt.show()
#-----------------------------------------------------------------------------
#Peaks    
    plt.figure(2)
    plt.title("Scale")
    capacitance = capacitance / np.max(np.abs(capacitance))
    capacitance = abs(capacitance)
    
    plt.plot(time, capacitance)
    
    capacitance = capacitance.flatten()
    time = time.flatten()
    
    # peaks, _ = find_peaks(capacitance, distance=90)
    plt.plot(time, capacitance, label=name[my_col], color=colorrr[my_col])
    plt.legend() 
    # if col == 7 or col == 12 or col == 17 or col == 22 or col == 27 or col == 32 or col == 37:
    #     my_col += 1
    # plt.legend()
    # col += 1
    # plt.plot(time[peaks], capacitance[peaks], "xr")
#-----------------------------------------------------------------------------

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
    print("The number of iteration: " + str(num_iter))
    num_iter += 1
    print(beta_hat_fit)
    print("Length: " + str(len(beta_hat_fit)))
    resY = beta_hat_fit[1] * (beta_hat_fit[4]/2 + beta_hat_fit[5]/2)
    resX = beta_hat_fit[2] - beta_hat_fit[0]
    final_dataX.append(resX)
    final_dataY.append(resY)
    
    beta_coordianates.append(beta_hat_fit)
    
    # plt.plot(7)
    # plt.title("6D")
    # # print(beta_hat_fit)
    # # for i in range(6):
    # #     print(beta_hat_fit[i])
    # plt.scatter(beta_hat_fit[0], beta_hat_fit[1], beta_hat_fit[2], beta_hat_fit[3], beta_hat_fit[4], beta_hat_fit[5])
    # plt.show()
    plt.scatter(beta_hat_fit[0], beta_hat_fit[1], beta_hat_fit[2])
#-----------------------------------------------------------------------------
    if col == 7 or col == 12 or col == 17 or col == 22 or col == 27 or col == 32 or col == 37:
        my_col += 1
    plt.legend()
    col += 1
    
    # if(col == 8 or col == 13 or col == 18 or col == 23 or col == 28 or col == 33 or col == 38):
    #     c = 0
    # else:
    #     c += 1
#-----------------------------------------------------------------------------
plt.figure(4)

my_col = 0
for i in range(36):
    plt.scatter(final_dataX[i], final_dataY[i], label = name[my_col], color = colorrr[my_col], marker = 'x')
    if i == 5 or i == 10 or i == 15 or i == 20 or i == 25 or i == 30 or i == 35:
        my_col += 1
plt.legend()
#-----------------------------------------------------------------------------
ar_x = []
ar_y = []
#-----------------------------------------------------------------------------
an = find_anomalies(final_dataY)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
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
    print(an)
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
            
    # fig = plt.figure(6)
    # ax = fig.add_subplot(projection='3d')
    
    # ax.scatter(ar_x, ar_y,lasso.fit(ar_y, y_predicted).predict(ar_y), c='y', marker='o')
    
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    
    # plt.show()
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    
    # zs = [0.0, 1.0, 2.0]
    # t  = np.arange(1024)*1e-6
    # ones = np.ones(1024)
    # y1 = np.sin(t*2e3*np.pi) 
    # y2 = 0.5*y1
    # y3 = 0.25*y1
    
    # verts=[list(zip(t, y1))]
    
    # poly = PolyCollection(verts, facecolors = ['r','g','b'])
    # poly.set_alpha(0.7)
    # ax.add_collection3d(poly, zs=zs, zdir='y')
    # ax.set_xlabel('X')
    # ax.set_xlim3d(0, 1024e-6)
    # ax.set_ylabel('Y')
    # ax.set_ylim3d(-1, 3)
    # ax.set_zlabel('Z')
    # ax.set_zlim3d(-1, 1)
    
    # plt.show()
# print(final_dataX)










final_dataX = [0.1049180834771779, -141.7993536675172, 0.06999121122593192, 0.0352142572473213, 0.03196148689023669, 0.033492850964631574, 2.9038007540401316, 5.241116118232837, -0.006914659184557137, 3.8314567106686543, 0.01824950471421871, 0.1440305327482374, 0.2257413337312184, 0.030118343766482925, 2.4452196867312592, 1.2601557157827277, -0.004417684204923056, -0.028886311380549312, 3.8219153360286864, 0.018473059621451737, 2.7603300402005946, 10.447534740750822, 6.225021100780368, 8.054696817802515, 2.692856130102423, -26.237391165239067, 1.7983583819183526, 2.827573773812367, 0.06648559070355708, 6.500786171072487, 2.0574747833617657, -0.009964298815200001, 0.020166317675365475, 0.05559581063113228, 0.4043335111470952, 0.053599431014172194]
# print(final_dataY)
final_dataY = [438.417678169069, 15661.830278039562, 392.8078903137797, 271.2224520380711, 315.5675024360001, 435.6925709795137, 1192.7713236217064, 1854.5365354657893, 342.1988036840662, 2173.170028411074, 439.7756639310759, 534.2613724821023, 714.0548043185165, 487.7838520464866, 1994.0555101562747, 887.5630703277051, 378.82795665533547, 693.0389600317202, 2020.855834164435, 793.0237161208332, 1565.1508345992627, 3766.9888749743072, 2752.1117450259426, 2468.4114027403084, 1665.8469307668063, 13012.81174910835, 1018.3295450399688, 1586.7163482499625, 524.0458368799476, 2139.290669976549, 1294.370894560307, 542.65607493338, 401.5618411573114, 559.9738800107923, 729.0641845507485, 572.643896585693]
#-----------------------------------------------------------------------------
import matplotlib.ticker as ticker

def parallel_coordinates(data_sets, style=None):

    dims = len(data_sets[0])
    x    = range(dims)
    fig, axes = plt.subplots(1, dims-1, sharey=False)

    if style is None:
        style = ['r-']*len(data_sets)

    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        mn = min(m)
        mx = max(m)
        if mn == mx:
            mn -= 0.5
            mx = mn + 1.
        r  = float(mx - mn)
        min_max_range.append((mn, mx, r))

    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = [(value - min_max_range[dimension][0]) / 
                min_max_range[dimension][2] 
                for dimension,value in enumerate(ds)]
        norm_data_sets.append(nds)
    data_sets = norm_data_sets

    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, style[dsi])
        ax.set_xlim([x[i], x[i+1]])

    # Set the x axis ticks 
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        for i in range(ticks):
            v = mn + i*step
            labels.append('%4.2f' % v)
        axx.set_yticklabels(labels)

    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%4.2f' % (mn + i*step) for i in range(ticks)]
    axx.set_yticklabels(labels)

    # Stack the subplots 
    plt.subplots_adjust(wspace=0) 

    return plt

parallel_coordinates(beta_coordianates).show()
#-----------------------------------------------------------------------------
from sklearn import decomposition

pca = decomposition.PCA()
pca.n_components = 2
x_train_pca = pca.fit_transform(beta_coordianates)    

# plt.plot(x_train_pca)
plt.plot(final_dataX, x_train_pca)
plt.plot

for i in range(36):
    plt.scatter(36, beta_coordianates[i][0])
    
"NOTES"
#eigen vectors
# x = np.arange(-10, 10)
# y = 2*x + 1

plt.figure()
# plt.plot(x, y)
plt.xlim(-2, 10)
plt.ylim(-2, 10)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.show()

Mid = beta_coordianates
Mid = np.array(Mid)
# for i in range (36):
#     if i < 36 / 2:
#         Mid.append(beta_coordianates[i][0]) 
#     else:
#         Mid.append(30000)
# Mid = np.array(Mid)
# Mid = np.reshape(Mid, (-1, 2))

from matplotlib.cm import register_cmap
from scipy import stats
#from wpca import PCA
from sklearn.decomposition import PCA as PCA
import seaborn
# from wpca import PCA
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
Mid_test = scaler.fit_transform(Mid)
covariance = np.cov(Mid, rowvar = False)

# mean_vec = np.mean(Mid, axis=0)
# cov_mat = (Mid - mean_vec).T.dot((Mid - mean_vec)) / (Mid.shape[0]-1)
# cov_mat = np.cov(Mid.T)
eig_vals, eig_vecs = np.linalg.eig(covariance)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
# pca = PCA(n_components=2)
pca = PCA(n_components=2)
# pca.fit_transform(df1)
print (pca.explained_variance_ratio_ )

pca = PCA().fit(Mid)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

pca = decomposition.PCA()
pca.n_components = 3
pca.get_covariance()
Mid_train_pca = pca.fit_transform(Mid)

#-----------------------------------------------------------------------------
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plt.plot(Mid[0][0], Mid[0][1], Mid[0][2], projection = '3d')
for i in range (36):
    ax.scatter(Mid[i][0], Mid[i][1], Mid[i][2], label=str(i))

plt.legend()
plt.show()

plt.figure(9)
for i in range(36):
    plt.scatter(x_train_pca[i][0], x_train_pca[i][1])
       
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plt.plot(Mid[0][0], Mid[0][1], Mid[0][2], projection = '3d')
for i in range (31, 36):
    ax.scatter(Mid_train_pca[i][0], Mid_train_pca[i][1], Mid_train_pca[i][2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.legend()
plt.show()


scree = [0] * 6
for i in range (6):
    print((eig_vals[i] / sum(eig_vals))*100) 
    scree[i] = (eig_vals[i] / sum(eig_vals))*100
plt.scatter(6, scree)    
       
print(eig_vecs.shape)
print(eig_vecs[0])
print(eig_vecs[1])
print(eig_vecs[2])
for i in range(4):
    eig_vecs = np.delete(eig_vecs, 2, 1)
eig_vecs = eig_vecs.T
Mid_test = Mid_test.T
Final_PCA = np.matmul(eig_vecs, Mid_test).T

for i in range (36):
    plt.scatter(Final_PCA[i][0], Final_PCA[i][1])
from sklearn.decomposition import PCA as PCA

pca = PCA().fit(Mid)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

#Make a random array and then make it positive-definite
num_vars = 6
# num_obs = 9
# A = np.random.randn(num_obs, num_vars)
# A = np.asmatrix(A.T) * np.asmatrix(A)
# U, S, V = np.linalg.svd(A) 
eigvals = eig_vals
#-----------------------------------------------------------------------------
import matplotlib
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, scree, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
#I don't like the default legend so I typically make mine like below, e.g.
#with smaller fonts and a bit transparent so I do not cover up data, and make
#it moveable by the viewer in case upper-right is a bad place for it 
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                  shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                  markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()