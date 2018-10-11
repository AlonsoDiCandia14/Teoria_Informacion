import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import classification_report

new_table = pd.read_csv("SkillCraft1_Dataset.csv")
header = list(new_table)
NaN_index = [] 

for h in header:
    c = 0
    for n in new_table[h]:
        if n == '?':
            new_table.drop(new_table.index[c],inplace=True)
            continue
        c = c + 1
#Mostramos la tabla generada
#new_table['Index'] = [x for x in range(len(y))]
new_table.reset_index(drop=False,inplace=True)
new_table.head()

y = new_table['LeagueIndex']
X = ['TotalHours','UniqueHotkeys','MinimapAttacks','NumberOfPACs','ActionsInPAC','TotalMapExplored','WorkersMade','ComplexAbilitiesUsed']

phi = np.zeros((len(y),len(X)+1))
phi[:,0] = np.ones(len(y))
for i,x in enumerate(X):
    phi[:,i+1] = np.asarray(new_table[x])
    
phi_t = phi.T
t = np.zeros((len(y),1))
t[:,0] = np.asarray(y)
multiplicate_phi = np.dot(phi_t,phi)
inverse_of_phi = inv(multiplicate_phi)
phi_final = np.dot(inverse_of_phi,phi_t)
w = np.dot(phi_final,t)

Y_result = []
for i in range(len(y)):
    y_predict = w[0]
    for j,x in enumerate(X):
        y_predict = y_predict + w[j+1]*float(new_table[x][i]) 
    Y_result.append(y_predict)

    t = range(0,len(y))
plt.figure(1)
plt.plot(t,y,t,Y_result,'--')
plt.show()