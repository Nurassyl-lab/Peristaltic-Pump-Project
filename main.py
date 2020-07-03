import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd

#If your file has non-ASCII characters
#workbook = xlrd.open_workbook('my_file_name.xls', encoding='cp1252')
#workbook = xlrd.open_workbook('data.xlsx')
#If your spreadsheet is very large
workbook = xlrd.open_workbook('data.xlsx', on_demand = True)
print(workbook.nsheets)
print(workbook.sheet_names())

#opens sheet by its index
#worksheet = workbook.sheet_by_index(0)
#opens sheet by name
worksheet = workbook.sheet_by_name('Sheet1')

time = [worksheet.cell_value(i, 0) for i in range (1709)]
capacitance_all = [worksheet.cell_value(i, 1) for i in range (1709)]

#capacitance_all = str(capacitance_all)
#capacitance = [time, capacitance_all]
#capacitance = [capacitance_all, time]

plt.plot(time , capacitance_all) 
#plt.plot(capacitance_all, time) 
    
#plt.errorbar(time, capacitance_all)

#plt.axis([0,5,0,5])

#plt.axis([0, 186,  0.0000000000491116, 0.0000000000601116])
#plt.axis([0.0000000000561116, 0.0000000000601116, 0, 5])

plt.show()