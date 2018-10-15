import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# new_table = pd.read_csv("SkillCraft1_Dataset.csv")
# header = list(new_table)
# NaN_index = [] 

# for h in header:
#     c = 0
#     for n in new_table[h]:
#         if n == '?':
#             new_table.drop(new_table.index[c],inplace=True)
#             continue
#         c = c + 1
# #Mostramos la tabla generada
# #new_table['Index'] = [x for x in range(len(y))]
# new_table.reset_index(drop=False,inplace=True)
# new_table[0:10]

# y = new_table['LeagueIndex']
# X = ['TotalHours','UniqueHotkeys','NumberOfPACs','ActionsInPAC','TotalMapExplored','WorkersMade','ComplexAbilitiesUsed']
# x_label = range(0,len(y))

# phi = np.zeros((len(y),len(X)+1))
# phi[:,0] = np.ones(len(y))
# for i,x in enumerate(X):
#     phi[:,i+1] = np.asarray(new_table[x]) 
# #phi[:,6] = phi[:,6]*(-1)
# phi_t = phi.T
# t = np.zeros((len(y),1))
# t[:,0] = y
# multiplicate_phi = np.dot(phi_t,phi)
# inverse_of_phi = inv(multiplicate_phi)
# phi_final = np.dot(inverse_of_phi,phi_t)
# w = np.dot(phi_final,t)

# accuracy = 0.1000000
# Y_result3 = []
# poly = PolynomialFeatures(2)
# for i in range(100):
#     clf = LinearRegression()
#     phi_transform = poly.fit_transform(phi[:,1:8])
#     fit = clf.fit(phi_transform,y)
#     Y_in_for = clf.predict(phi_transform)
#     accuracy_in_for = mean_squared_error(y,Y_in_for)
#     if accuracy_in_for > accuracy:
#         accuracy = accuracy_in_for
#         Y_result3 = Y_in_for

# beta = mean_squared_error(y,Y_result3)
# print('MSE:',beta)
# plt.figure(figsize =(20,4))
# plt.plot(x_label,y,'b*',label='real')
# plt.plot(x_label,Y_result3,'r-',label='POLY')
# plt.fill_between(range(0,len(y)), Y_result3+beta, Y_result3-beta, facecolor='yellow')
# plt.axis([100,200, 0, 8])
# plt.xlabel('Jugador nº')
# plt.ylabel('Liga')
# plt.legend()
# plt.show()

x = 0.4
print(round(x))