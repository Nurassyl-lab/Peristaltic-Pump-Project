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

plt.plot(time , capacitance_all) 

plt.show()