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

worksheet = workbook.sheet_by_name('Sheet1')

time = [worksheet.cell_value(i, 0) for i in range (1709)]
capacitance_all = [worksheet.cell_value(i, 1) for i in range (1709)]
plt.figure(1)
plt.title("FULL TIME")
plt.plot(time , capacitance_all) 
plt.show()

plt.figure(2)
plt.title("ONLY 40 SECONDS")
time = [worksheet.cell_value(i, 0) for i in range (373)]
capacitance_all = [worksheet.cell_value(i, 1) for i in range (373)]
plt.plot(time , capacitance_all) 
plt.show()

plt.figure(3)
plt.title("ONLY 12 SECONDS WHEN PUMP IS OFF")
time = [worksheet.cell_value(i, 0) for i in range (116)]
capacitance_all = [worksheet.cell_value(i, 1) for i in range (116)]
plt.plot(time , capacitance_all) 
plt.show()

