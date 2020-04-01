# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 07:49:36 2020

@author: suyog
"""

#Visualization of Multiple Regression

X = [ [150,100], [159,200], [170,350], [175,400], [179,500], [180,180], [189,159], [199,110], [199,400],[199,230], [235,120],[239,340],[239,360],[249,145]]
y = [0.73,1.39,2.03,1.45,1.82,1.32,0.83,0.53,1.95,1.27,0.49,1.03,1.24,0.55]

#Prepare the dataset
import pandas as pd
dataset = pd.DataFrame(X,columns=['Price','AdSpends'])
dataset['Sales']= pd.Series(y)

#Apply the multiple linear regression
import statsmodels.formula.api as smf
model = smf.ols(data=dataset,formula='Sales ~ Price + AdSpends')
results_formula = model.fit()
results_formula.params  #Check the intercept, price and adspends

#Prepare the visualization
import matplotlib.pyplot as plt
import numpy as np

X_surf,y_surf = np.meshgrid(np.linspace(dataset.Price.min(),dataset.Price.max(),100), \
                            np.linspace(dataset.AdSpends.min(),dataset.AdSpends.max(),100))

onlyX = pd.DataFrame({'Price' : X_surf.ravel(), 'AdSpends' : y_surf.ravel()})

fittedY = results_formula.predict(exog=onlyX)

#Convert the predicted result in an arraay since matplotlib takes array
fittedY = np.array(fittedY)

#Visualization
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(dataset['Price'],dataset['AdSpends'],dataset['Sales'],c='red',marker='o',alpha=0.5)
ax.plot_surface(X_surf,y_surf,fittedY.reshape(X_surf.shape),color='b',alpha=0.3)
ax.set_xlabel('Price')
ax.set_ylabel('Ads Spends')
ax.set_zlabel('Sales')
plt.show()


                            