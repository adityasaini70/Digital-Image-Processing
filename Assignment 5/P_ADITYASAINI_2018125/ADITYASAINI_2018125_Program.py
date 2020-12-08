import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
from scipy import signal
from tabulate import tabulate

#Solution to 1b
print("Solution to 1b goes here")

img=cv2.imread('ADITYASAINI_2018125_Q1b_Barbara.bmp',0)

#Adding noise to image
mean = 0
variance = 500
noise = np.random.normal(mean,math.sqrt(variance),img.shape)

#Defining the degradation filter : Averaging filter
filter_size=3
h= (1/(filter_size)**2) * (np.ones([filter_size,filter_size]))

#Computing g
g=signal.fftconvolve(img,h,mode='same')+noise
error = np.mean((g-img)**2)
psnr_noise = 10 * math.log10((255**2)/error)

#Finding FFTs for g and h
G=np.fft.fftshift(np.fft.fft2(g))
H=np.fft.fftshift(np.fft.fft2(h,s=img.shape))

#Defining the Laplaccian
l=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
L=np.fft.fftshift(np.fft.fft2(l,s=img.shape))

#Finding results for given filter
min_thresh=math.inf
denoised_filter=0
psnr_filter=0
for k in np.arange(0.00095,4,0.25):
    for gamma in np.arange(0.001,4,0.25):
        given_filter = np.conj(H) / ((np.abs(H)**2 + k**2)*(1+ gamma * np.abs(L)**2))
        F = G * given_filter
        f_bar = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
        error = np.mean((f_bar-img)**2)
        if error < min_thresh:
            min_thresh = error
            denoised_filter=f_bar
            psnr_filter = 10 * math.log10((255**2)/error)

print("Filter Error[Min] ", error)
print("Filter PSNR ",psnr_filter)

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(g, cmap='gray')
plt.title("Noisy Image; PSNR= "+str(psnr_noise))

plt.subplot(1,3,3)
plt.imshow(denoised_filter, cmap='gray')
plt.title("Denoised using given filter; Best PSNR= " + str(psnr_filter))

plt.show()

#Finding results for Wiener Filter
min_thresh=math.inf
denoised_wiener=0
psnr_wiener=0
for k in np.arange(0.00095,4,0.02):
    wiener_filter = np.conj(H) / ((np.abs(H))**2 + k**2)
    F = G * wiener_filter
    f_bar = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    error = np.mean((f_bar-img)**2)
    if error < min_thresh:
        min_thresh = error
        denoised_wiener=f_bar
        psnr_wiener = 10 * math.log10((255**2)/error)

print("Wiener Filter Error[Min]",error)
print("Wiener Filter PSNR ",psnr_wiener)

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(g, cmap='gray')
plt.title("Noisy Image; PSNR= "+str(psnr_noise))

plt.subplot(1,3,3)
plt.imshow(denoised_wiener, cmap='gray')
plt.title("Denoised using Wiener filter; Best PSNR= " + str(psnr_wiener))

plt.show()

#Comparing Weiner and given filter
plt.subplot(1,3,1)
plt.imshow(g, cmap='gray')
plt.title("Noisy Image; PSNR= "+str(psnr_noise))

plt.subplot(1,3,2)
plt.imshow(denoised_wiener, cmap='gray')
plt.title("Denoised using Wiener filter;Best PSNR= " + str(psnr_wiener))

plt.subplot(1,3,3)
plt.imshow(denoised_filter, cmap='gray')
plt.title("Denoised using given filter;Best PSNR= " + str(psnr_filter))

plt.show()


#Solution to 3a
print("Solution to 3a goes here")

#Defining the array
f=np.array([[1,1,1,1],[0,10,10,1],[0,2,3,1],[0,5,15,8]])

#Defining the Sobel gradients
sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

#Calculating the x and y gradients
G_x = signal.correlate2d(f,sobel_x,mode='same')
G_y = signal.correlate2d(f,sobel_y,mode='same')

#Calculating the magnitude of the gradient
M = np.abs(G_x) + np.abs(G_y)

#Calculating the direction of the gradient
dir_M = np.arctan2(G_y, G_x) * 180 / np.pi

print("Original matrix:\n")
print(tabulate(f, tablefmt='pretty'))

print("Magnitude of Gradient:\n")
print(tabulate(M, tablefmt='pretty'))

print("Direction of Gradient:\n")
print(tabulate(dir_M, tablefmt='pretty'))

#Solution to 3b
print("Solution to 3b goes here")

#Computing different masks
horiz_mask = np.logical_or(np.logical_and(dir_M<=22.5,dir_M>=-22.5),np.logical_and(dir_M>=157.5,dir_M<=-157.5))
vert_mask = np.logical_or(np.logical_and(dir_M<=-67.5,dir_M>=-112.5),np.logical_and(dir_M<=112.5,dir_M>=67.5))
plus45_mask = np.logical_or(np.logical_and(dir_M<157.5,dir_M>112.5),np.logical_and(dir_M<-22.5,dir_M>-67.5))
minus45_mask = np.logical_or(np.logical_and(dir_M<67.5,dir_M>22.5),np.logical_and(dir_M<-112.5,dir_M>-157.5))

#Computing non-max suppression
g_n= np.zeros(M.shape)
for i in range(1,M.shape[0]-1):
    for j in range(1,M.shape[1]-1):
        if horiz_mask[i,j]:
            if M[i,j] < M[i,j+1] or M[i,j] < M[i,j-1]:
                g_n[i,j] = 0
            else:
                g_n[i,j] = M[i,j]
        elif vert_mask[i,j]:
            if M[i,j] < M[i-1,j] or M[i,j] < M[i+1,j]:
                g_n[i,j] = 0
            else:
                g_n[i,j] = M[i,j]
        elif plus45_mask[i,j]:
            if M[i,j] < M[i-1,j-1] or M[i,j] < M[i+1,j+1]:
                g_n[i,j] = 0
            else:
                g_n[i,j] = M[i,j]
        elif minus45_mask[i,j]:
            if M[i,j] < M[i-1,j+1] or M[i,j] < M[i+1,j-1]:
                g_n[i,j] = 0
            else:
                g_n[i,j] = M[i,j]

print("Non-max suppression output [g_n]:\n")
print(tabulate(g_n, tablefmt='pretty'))

#Solution to 3c
print("Solution to 3c goes here")

#Defining the thresholds
T_h = 90/100 * np.max(M.ravel())
T_l = 30/100 * np.max(M.ravel())

#Computing gnh & gnl
g_nh = np.zeros(g_n.shape)
g_nl = np.zeros(g_n.shape)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if g_n[i,j] >= T_h:
            g_nh[i,j] = 1
        else:
            g_nh[i,j] = 0
        
        if g_n[i,j] >= T_l:
            g_nl[i,j] = 1
        else:
            g_nl[i,j] = 0

print("After thresholding with T_h [g_nh]:\n")
print(tabulate(g_nh, tablefmt='pretty'))

print("After thresholding with T_l [g_nl]:\n")
print(tabulate(g_nl, tablefmt='pretty'))