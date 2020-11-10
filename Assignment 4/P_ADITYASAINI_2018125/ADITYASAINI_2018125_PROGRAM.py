## DIP Assignment - 4
## Aditya Saini
## 2018125, B.Tech ECE

import matplotlib.pyplot as plt
import numpy as np
import cv2

print("Solution to 1a goes here")


def filter_generate(m,n,type,scale_radius,x=0,y=0):
    filter = np.zeros([m,n])
    radius = scale_radius* filter.shape[0]

    if type=='butterworth':
        order = 4
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                dist = ((i - filter.shape[0]/2+y)**2 + (j - filter.shape[1]/2 + x)**2)**0.5
                filter[i][j]=1/(1+(dist/radius)**(2*order))
    if type =='idealpf':
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                dist = ((i - filter.shape[0]/2+y)**2 + (j - filter.shape[1]/2+x)**2)**0.5
                if dist < radius:
                    filter[i][j]=1
                    
    if type =='gauss':
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                dist = ((i - filter.shape[0]/2+y)**2 + (j - filter.shape[1]/2+x)**2)**0.5
                filter[i][j]=np.exp(-dist**2/(2*scale_radius*scale_radius))

    return filter
    

img = cv2.imread('ADITYASAINI_2018125_Q1_Noise-lines.jpg',0)
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
centered_fft=np.fft.fftshift(np.fft.fft2(img))
plt.imshow(1+np.log(np.abs(centered_fft)),cmap='gray')
plt.title('Centered FFT spectrum of image')
plt.show()
#Initializing the Ideal Low Pass filter
m=img.shape[0]
n=img.shape[1]
filter = np.ones([m,n])
center=n//2
filter[:,center-3:center+3]=0

filter = filter+filter_generate(m,n,'butterworth',0.01)
plt.imshow(1+np.log(np.abs(filter)),cmap='gray')
plt.title('Generated Filter')
plt.show()
#Beginning the Filtering process
filtered_img = centered_fft * filter
plt.imshow(1+np.log(np.abs(filtered_img)),cmap='gray')
plt.title("Magnitude Spectrum of the Filter")
plt.show()
#Obtaining the filtered image
final_img = np.fft.ifft2(np.fft.ifftshift(filtered_img))
plt.imshow(np.real(final_img),cmap='gray')
plt.title("Filtered Image(obtained through Fourier filtering)")
plt.show()

print("Solution to 1b goes here")

spatial_filter = np.real(np.fft.ifft2(np.fft.ifftshift(filter)))

plt.subplot(1,2,1)
plt.imshow(1+np.log(spatial_filter),cmap='gray')
plt.title("Spatial filter")

plt.subplot(1,2,2)
plt.imshow(1+np.log(np.fft.ifftshift(spatial_filter)),cmap='gray')
plt.title("Shifted Spatial filter")
plt.show()

from scipy import signal
final_img_conv=signal.fftconvolve(img,np.fft.ifftshift(spatial_filter),mode='same')
plt.imshow(np.real(final_img_conv),cmap='gray')
plt.title('Image obtained through spatial domain filtering')

print("Solution to 1c goes here")
plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(np.real(final_img),cmap='gray')
plt.title("Image obtained through Fourier filtering")

plt.subplot(1,3,3)
plt.imshow(np.real(final_img_conv),cmap='gray')
plt.title("Image obtained through spatial domain filtering")
plt.show()
print("Solution to 1d goes here")
plt.imshow(np.real(img-final_img),cmap='gray')
plt.title('Noise of the image')
plt.show()


print("Solution to 2a given in ADITYASAINI_2018125_Q2a.pdf")
print("Solution to 2b goes here")
img = cv2.imread('ADITYASAINI_2018125_Q2_Barbara.bmp',0)
plt.imshow(img,cmap='gray')
plt.title("Original Image")
plt.show()
#Padding the image and filter
m=img.shape[0]
n=img.shape[1]
img_padded = np.zeros([m+2,n+2])
img_padded[:m,:n]=img
filter_padded = np.zeros([m+2,n+2])

#Shifting the filter into the 1st quadrant
filter_padded[511][511]=0
filter_padded[511][0]=1
filter_padded[511][1]=0
filter_padded[0][511]=1
filter_padded[0][0]=-4
filter_padded[0][1]=1
filter_padded[1][511]=0
filter_padded[1][0]=1
filter_padded[1][1]=0

fft_filter = np.fft.fft2(filter_padded)
fft_img = np.fft.fft2(img_padded)

fft_filter_centered = np.fft.fftshift(fft_filter)
fft_img_centered = np.fft.fftshift(fft_img)

plt.subplot(2,2,1)
plt.imshow(np.abs(fft_filter),cmap='gray')
plt.title('Filter shifted to 1st quadrant')
plt.subplot(2,2,2)
plt.imshow(np.abs(fft_filter_centered),cmap='gray')
plt.title('FFT centered filter')
plt.subplot(2,2,3)
plt.imshow(1+np.log(np.abs(fft_img)),cmap='gray')
plt.title('FFT spectrum of img')

plt.subplot(2,2,4)
plt.imshow(1+np.log(np.abs(fft_img_centered)),cmap='gray')
plt.title('FFT centered img')

plt.show()


filtered_img_fft = fft_img_centered * fft_filter_centered
plt.imshow(1+np.log(np.abs(filtered_img_fft)), cmap='gray')
plt.title('FFT spectrum of filtered Image')
plt.show()
filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_img_fft))
filtered_img=filtered_img[:m,:n]
plt.imshow(np.real(filtered_img[:m,:n]), cmap='gray')
plt.title('Obtained Image')
plt.show()


print("Solution to 3 goes here")
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
centered_fft=np.fft.fftshift(np.fft.fft2(img))
plt.imshow(1+np.log(np.abs(centered_fft)),cmap='gray')
plt.title('Centered FFT spectrum of image')
plt.show()

#Initializing the filter
m=img.shape[0]
n=img.shape[1]
# print(m,n)
filter_lpf = filter_generate(m,n,'gauss',20)

plt.imshow(filter_lpf, cmap='gray')
plt.title('Gaussian Filter with Do = 20')
plt.show()
filtered_img_fft = centered_fft * filter_lpf
plt.imshow(1+np.log(np.abs(filtered_img_fft)),cmap='gray')
plt.title('FFT Spectrum of filtered_img')
plt.show()

filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_img_fft))
plt.imshow(np.real(filtered_img),cmap='gray')
plt.title('Obtained Image')
plt.show()