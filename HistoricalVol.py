import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd

# plt.style.use('ggplot')

stock = pd.read_csv("data/tesla.csv")
stock["Last"] = stock["Last"].values[::-1] #Reverse order
stock['Volatility'] = stock.Last.rolling(5).std()
# Last = np.array(stock.Last)
# Vol = np.array(stock.Volatility)
# print(Last)
# print(Vol)
# # rho = np.corrcoef(Last, Vol)
# # print(rho)

CorrMatrix = stock.corr()
rho = round(CorrMatrix.Open['Volatility'], 2)

# Plotting historical volatility and closing prices
fig, axs = plt.subplots(2)
fig.set_size_inches(16, 8)

axs[0].plot(stock.Volatility)
axs[0].grid()
axs[0].set_xlabel('Trading days since 03/20/2019')
axs[0].set_ylabel('Historical Volatility (5 day)')
axs[0].set_xlim([0, 520])

axs[1].plot(stock.Last)
axs[1].grid()
axs[1].set_xlabel('Trading days since 03/20/2019')
axs[1].set_ylabel('Tesla stock')
axs[1].set_xlim([0, 520])
plt.savefig("fig/tesla " + str(rho) + ".jpg")