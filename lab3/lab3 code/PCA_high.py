import numpy as np
import matplotlib.pyplot as plt
import cv2

# Generating the high dimensional data
np.random.seed(5)
Dim = 20 
mean = np.random.rand(1,Dim).squeeze()   
cov = np.random.rand(Dim,Dim)        
cov = cov + cov.T         
original_data = np.random.multivariate_normal(mean,cov,1000)

mean, eigenvectors, eigenvalues = cv2.PCACompute2(original_data, np.array([]))   
#conduct PCA and return eigenvectors and eigenvalues
lowDim_data= cv2.PCAProject(original_data,mean,eigenvectors) 
#Project original data into low dimension

plt.style.use('ggplot')          
plt.figure(figsize=(10, 6))
plt.plot(lowDim_data[:, 0], lowDim_data[:, 1], 'o') #Plot low-dimensional data based on first two principal components
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.axis([-20, 20, -10, 10])
plt.show()






