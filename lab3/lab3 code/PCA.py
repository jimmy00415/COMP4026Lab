import numpy as np
import matplotlib.pyplot as plt
import cv2

mean = [20, 20]                                           
cov = [[5 ,0], [25, 25]]                                  
x, y = np.random.multivariate_normal(mean,cov,1000).T   # generate synthetic data

plt.style.use('ggplot')
plt.plot(x,y,'o',zorder=1)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()        

X= np.vstack((x,y)).T                     #form a feature matrix by combining feature vectors x and y
mean, eigenvectors, eigenvalues = cv2.PCACompute2(X, np.array([]))   
#conduct PCA and return data mean, eigenvectors, and eigenvalues

# Display the extracted eigenvectors and eigenvalues on the original data
plt.plot(x, y, 'o', zorder=1)
plt.quiver([mean[0,0],mean[0,0]], [mean[0,1],mean[0,1]], eigenvectors[:, 0], eigenvectors[:, 1], zorder=3, scale=0.2, units='xy')

plt.text(mean[0,0] + 5 * eigenvectors[0, 0], mean[0,1] + 5 * eigenvectors[0, 1], str(int(eigenvalues[0].item())), zorder=5,
         fontsize=16, bbox=dict(facecolor='white', alpha=0.6))
plt.text(mean[0,0] + 7 * eigenvectors[1, 0], mean[0,1] + 4 * eigenvectors[1, 1], str(int(eigenvalues[1].item())), zorder=5,
         fontsize=16, bbox=dict(facecolor='white', alpha=0.6))
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()





