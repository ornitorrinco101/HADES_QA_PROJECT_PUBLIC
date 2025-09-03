import numpy as np

from .definitions import Distribution
import os
from pathlib import Path
import re

import sys
from pathlib import Path
import pickle
from scipy.interpolate import SmoothBivariateSpline
from skimage.measure import block_reduce
from scipy.stats import beta
from scipy.stats import gamma

# Add project root to Python path
root_dir = Path(__file__).parents[2] # Goes up 2 levels to ANOMALYDETECTIONHADES
sys.path.append(str(root_dir))
print(root_dir)

def normal_dist(mean=0.0, sigma=1.0, size=10000):
    return np.random.normal(mean, sigma, size)

def uniform_dist(low=0.0, high=1.0, size=10000):
    return np.random.uniform(low, high, size)

# def landau_dist(mean=0.0, sigma=1.0, size=10000):
#     return np.array([ROOT.gRandom.Landau(mean, sigma) for _ in range(size)])

def exponential_dist(scale=1.0, size=10000):
    return np.random.exponential(scale, size)

def double_peak(mean1=-1.0, sigma1=0.5, mean2=2.0, sigma2=0.7, frac=0.5, size=10000):
    size1 = int(frac * size)
    size2 = size - size1
    return np.concatenate([
        np.random.normal(mean1, sigma1, size1),
        np.random.normal(mean2, sigma2, size2)
    ])

def START_beam_profile(mod,mean=None,std=None,nsamples=200,objective=1000,numbins=20):
    if mod==1 and (mean==None or std==None):
        mean=np.random.uniform(8, 10)
        std=np.random.uniform(1,1.5)
    if mod==2 and (mean==None or std==None):
        mean=np.random.uniform(8, 10)
        std=np.random.uniform(1.5,2.0)

    mod=np.clip(normal_dist(mean,std,nsamples).round(0),0,numbins)
    mod=np.repeat(mod,objective//nsamples)

    counts_mod=np.histogram(mod,np.arange(0,numbins+2,1))

    bins=counts_mod[1][:-1]
    dist=counts_mod[0]

    return bins,dist

def RICH_total_mult(minval=100,maxval=140,averaging_param=5,numbins=51):
    bins=np.arange(0,numbins,1)
    dist=uniform_dist(minval,maxval,(numbins,5)).mean(axis=1).round(0)
    return bins,dist

def load_spline_pickle(folder, filename):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def filter_area(intro,row_start,row_end,col_start,col_end,lmin=0,lmax=1,fill=False,fill_val=None):
    image=intro.copy()
    region = image[row_start:row_end, col_start:col_end]
 
    # Create a mask for values
    mask =(lmin<=region)&(lmax>=region)

    # Apply the mask to get the filtered values
    filtered_values = region[mask]

    if fill==True and fill_val!=None:
        image[row_start:row_end,col_start:col_end][mask]=fill_val
    elif fill==False:
        image[row_start:row_end,col_start:col_end][mask]=0
    else:
        print("Invalid fill value")
    return image

def RICH_CalsXY(root_dir=root_dir,og_size=572,kernel_downsample=6,add_noise=True,noise_factor=0.2,lmax=0.25):
    sx=572//kernel_downsample+1
    sy=sx

    x = np.arange(sx)  # [0, 1, ..., 49]
    y = np.arange(sy)  # [0, 1, ..., 49]

    spline_folder = os.path.join(root_dir, f'splinefits/kernel_{kernel_downsample}')
    spline_file = 'spline_fit.pkl'  # use .pkl to indicate pickle

    loaded_spline = load_spline_pickle(spline_folder, spline_file)

    Z_fit=loaded_spline(x,y,grid=True).T
    if add_noise==True:
        noise=np.random.uniform(0,Z_fit.mean(),Z_fit.shape)*noise_factor
        Z_fit+=noise
        Z_fit=filter_area(Z_fit,240//kernel_downsample,360//kernel_downsample,240//kernel_downsample,360//kernel_downsample,lmax=lmax)
    return Z_fit

def RICH_CalsXY_downsample(sample,kernel_size):
    box_size = (kernel_size, kernel_size)
    downsampled = block_reduce(sample, block_size=box_size, func=np.mean)

    return downsampled

def ToF_multiplicity(a=None,b=None,nsamples=4000,numbins=100):
    if a is None or b is None:
        a, b = np.random.uniform(1.5,2),20  # Symmetric parameters (a=b for bell shape)
    data = beta.rvs(a, b, size=nsamples)*numbins  # Scale to [0, 60]

    ToF_mult=np.histogram(data,np.arange(0,numbins+2,1))
    bins=ToF_mult[1][:-1]
    dist=ToF_mult[0]

    return bins,dist

def ToF_sum(k=None,theta=None,t0=None,nsamples=6000,noise_factor=0.15,numbins=150):
    if k is None or theta is None or t0 is None:
        k     = 3.0       # shape
        theta = 8.0       # scale
        t0    = 10.0      # shift in peak start (ns)

    # Generate shifted Gamma distribution
    gamma_samples = np.random.gamma(shape=k, scale=theta, size=nsamples)
    times = gamma_samples + t0
    noise=np.random.uniform(0,numbins+1,int(noise_factor*nsamples)) #introduction of an "acceptable" amount of noise
    times=np.concat([times,noise])

    # Plot for visual check
    ToF_sum=np.histogram(times,np.arange(0,numbins+2,1))
    bins=ToF_sum[1][:-1]
    dist=ToF_sum[0]

    return bins,dist



    