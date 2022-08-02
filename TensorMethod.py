
#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# Specify single image file name
filename = './data/NSC-Ph-1.tif'

# Pixel size of bluring kernel. Typically set at about 5 cell size.
sigma=40.0

# Threshold used for peak_local_max in detecting topological defects from chargedensity. Change if too many or too less defects are detected.
defthres=0.3

# Output figure size
fs=12

#%%
# Read image and normalize
I=np.float32(io.imread(filename))
I=(I-np.min(I))/np.max(I)-np.min(I)

# Using tensor method to calculate the phase
Ix = I[2:,1:-1]-I[0:-2,1:-1]
Iy = I[1:-1,2:]-I[1:-1,0:-2]

# Making tensor by & bluring with Gaussian filter
Gxx = ndi.gaussian_filter((Ix*Ix),sigma=sigma);
Gxy = ndi.gaussian_filter((Ix*Iy),sigma=sigma);
Gyy = ndi.gaussian_filter((Iy*Iy),sigma=sigma);

# Calculating coherence (the "amplitude" of nematic order) and phase (angle)
coh=((Gxx-Gyy)*(Gxx-Gyy)+4*Gxy*Gxy)/((Gxx+Gyy)*(Gxx+Gyy));
phi=np.arctan2(2*Gxy,Gxx-Gyy)/np.pi/2 

# Reconstructing tensor with amplitude set to 1
GGxx=np.sin(phi*np.pi)**2 - 0.5
GGyy=np.cos(phi*np.pi)**2 - 0.5
GGxy=np.cos(phi*np.pi)*np.sin(phi*np.pi)

# Calculating charge density to identify the position of topological defects
divXXX=GGxx[2:,1:-1]-GGxx[0:-2,1:-1]
divXYY=GGxy[1:-1,2:]-GGxy[1:-1,0:-2]
divXYX=GGxy[2:,1:-1]-GGxy[0:-2,1:-1]
divXXY=GGxx[1:-1,2:]-GGxx[1:-1,0:-2]
chargedensity=(divXXX*divXYY - divXYX*divXXY)

#%%
# Show image with detected angle of alignment
# Angle is from -0.5 to 0.5 (corresponding to -pi/2 to pi/2 where 0 is the x-axis)
plt.figure(figsize=(fs,fs))
plt.imshow(I,cmap='gray',vmin=0.0,vmax=0.3)
plt.imshow(phi,cmap=plt.cm.hsv,alpha=0.5)
plt.colorbar(orientation='horizontal',shrink=0.5)
plt.axis('off')

plt.savefig('./data/savefiles/Ph-with-Phase.png',bbox_inches='tight')

#%%
# Show coherence
plt.figure(figsize=(fs,fs))
plt.imshow(I, cmap=plt.cm.gray,vmin=0.0,vmax=0.3)
plt.imshow(coh,cmap='Greens',alpha=0.7)
plt.axis('off')

#%%
# Plotting topological defects
plt.figure(figsize=(fs,fs));
plt.axis('off')
plt.imshow(I, cmap=plt.cm.gray,vmin=0.0,vmax=0.3)
# Plus 1/2 toplogical defects
arrP=peak_local_max(-chargedensity, threshold_abs=defthres)
plt.plot(arrP[:,1],arrP[:,0],'ro',ms=10,alpha=0.7)
# Minus 1/2 toplogical defects
arrM=peak_local_max(chargedensity, threshold_abs=defthres)
plt.plot(arrM[:,1],arrM[:,0],'bo',ms=10,alpha=0.7)

plt.savefig('./data/savefiles/Ph-with-Defects.png',bbox_inches='tight')