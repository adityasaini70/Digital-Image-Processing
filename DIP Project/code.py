import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import math
from skimage.io import imread_collection
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from skimage import data, exposure

X_dir = 'C:/Users/Aditya Saini/DIP Project/Dataset/Training_Images/*.jpg'
X_images = imread_collection(X_dir)

Y_dir = 'C:/Users/Aditya Saini/DIP Project/Dataset/Ground_Truth/*.png'
Y_images = imread_collection(Y_dir)

print(len(X_images), len(Y_images))

#Randomly show some images and their corresponding masks
randomlist = random.sample(range(0, len(X_images)), 5)
for i in range(1,len(randomlist)):
    plt.subplot(4,2,i)
    plt.imshow(X_images[i])

for i in range(1,len(randomlist)):
    plt.subplot(4,2,i)
    plt.imshow(Y_images[i])


#Calculating hog for each image
X_images_final = []
idx=0
for img in X_images:
    print(idx)
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print('Shape of image = ',img.shape)
    l=0
    w=0

    img = cv2.resize(img, (64,128), interpolation = cv2.INTER_AREA)
    print('Resize shape of image = ',img.shape)

    if(len(img)%16==0):
        l=len(img)
    else:
        l=len(img)-(len(img)%16)

    if(len(img[0])%16==0):
        w=len(img[0])
    else:
        w=len(img[0])-(len(img[0])%16)

    lm=len(img)%16
    wm=len(img[0])%16

    reshaped=np.zeros([l,w])
    reshaped=img[lm//2:len(img)-lm//2, wm//2:len(img[0])-wm//2]

    #reshaped=img.reshape(l,w)
    #re12=img.reshape(128,64)

    grx=np.zeros([l,w])
    gry=np.zeros([l,w])

    mag=np.zeros([l,w])
    ang=np.zeros([l,w])

    Fx=[[1,2,1],[0,0,0],[1,2,1]]
    Fy=[[-1,0,1],[-2,0,2],[-1,0,1]]

    for i in range(1,l-1):
        for j in range(1,w-1):
            tempx=0
            tempy=0
            for k1 in range(-1,2):
                for l1 in range(-1,2):
                    tempx=tempx+reshaped[i+k1][j+l1]*Fx[k1+1][l1+1]
                    tempy=tempy+reshaped[i+k1][j+l1]*Fy[k1+1][l1+1]

            grx[i][j]=tempx
            gry[i][j]=tempy

            mag[i][j]=math.sqrt(math.pow(grx[i][j],2)+math.pow(gry[i][j],2))
            if grx[i][j] == 0:
                ang[i][j] = math.atan(gry[i][j]/0.01)*(180/math.pi)
            else:
                ang[i][j]=math.atan(gry[i][j]/grx[i][j])*(180/math.pi)

    hog=[]

    for i in range(0,l,8):
        ro=[]
        for j in range(0,w,8):
            bini=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            for i1 in range(i,i+8):
                for j1 in range(j,j+8):
                    lb=int(ang[i1][j1]/20)
                    ab=ang[i][j]%20
                    if(lb>160):
                        bini[8]=bini[8]+mag[i1][j1]
                    else:
                        ba=20-ab
                        bini[lb]=bini[lb]+(ba*mag[i1][j1])/20
                        bini[lb+1]=bini[lb+1]+(ab*mag[i1][j1])/20
            ro.append(bini)
        hog.append(ro)

    out=[]


    for i in range(0,len(hog)-1):
        for j in range(0,len(hog[0])-1):
            temp=[]
            temp=temp+hog[i][j]
            temp=temp+hog[i+1][j]
            temp=temp+hog[i][j+1]
            temp=temp+hog[i+1][j+1]
            s=0
            for i1 in range(0,len(temp)):
                s=s+(temp[i1]*temp[i1])
            sq=math.sqrt(s)
            for i1 in range(0,len(temp)):
                temp[i1]=temp[i1]/sq
            out=out+temp
    print('Lenght of HOG vector =', len(out))
    X_images_final.append(out)
    idx+=1



np.savetxt('X_hog_images.csv', np.array(X_images_final), delimiter=',')

#Preprocessing target values
Y_images_final = []
idx=0
for img in Y_images:
    Y_images_final.append(cv2.resize(img, (30,30), interpolation = cv2.INTER_AREA).ravel())
    print(idx)
    idx+=1

np.savetxt('Y_target_images.csv', np.array(Y_images_final), delimiter=',')


#Preprocessing Done!


#Training ML model starts
X = pd.read_csv('X_hog_images.csv',header=None)
y = pd.read_csv('Y_target_images.csv',header=None)

#Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)


#Implementation of HOG
def hog_feature(orig_img):
    img = orig_img
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print('Shape of image = ',img.shape)
    l=0
    w=0

    img = cv2.resize(img, (64,128), interpolation = cv2.INTER_AREA)
    print('Resize shape of image = ',img.shape)

    if(len(img)%16==0):
        l=len(img)
    else:
        l=len(img)-(len(img)%16)

    if(len(img[0])%16==0):
        w=len(img[0])
    else:
        w=len(img[0])-(len(img[0])%16)

    lm=len(img)%16
    wm=len(img[0])%16

    reshaped=np.zeros([l,w])
    reshaped=img[lm//2:len(img)-lm//2, wm//2:len(img[0])-wm//2]

    #reshaped=img.reshape(l,w)
    #re12=img.reshape(128,64)

    grx=np.zeros([l,w])
    gry=np.zeros([l,w])

    mag=np.zeros([l,w])
    ang=np.zeros([l,w])

    Fx=[[1,2,1],[0,0,0],[1,2,1]]
    Fy=[[-1,0,1],[-2,0,2],[-1,0,1]]

    for i in range(1,l-1):
        for j in range(1,w-1):
            tempx=0
            tempy=0
            for k1 in range(-1,2):
                for l1 in range(-1,2):
                    tempx=tempx+reshaped[i+k1][j+l1]*Fx[k1+1][l1+1]
                    tempy=tempy+reshaped[i+k1][j+l1]*Fy[k1+1][l1+1]

            grx[i][j]=tempx
            gry[i][j]=tempy

            mag[i][j]=math.sqrt(math.pow(grx[i][j],2)+math.pow(gry[i][j],2))
            if grx[i][j] == 0:
                ang[i][j] = math.atan(gry[i][j]/0.01)*(180/math.pi)
            else:
                ang[i][j]=math.atan(gry[i][j]/grx[i][j])*(180/math.pi)

    hog=[]

    for i in range(0,l,8):
        ro=[]
        for j in range(0,w,8):
            bini=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            for i1 in range(i,i+8):
                for j1 in range(j,j+8):
                    lb=int(ang[i1][j1]/20)
                    ab=ang[i][j]%20
                    if(lb>160):
                        bini[8]=bini[8]+mag[i1][j1]
                    else:
                        ba=20-ab
                        bini[lb]=bini[lb]+(ba*mag[i1][j1])/20
                        bini[lb+1]=bini[lb+1]+(ab*mag[i1][j1])/20
            ro.append(bini)
        hog.append(ro)

    out=[]


    for i in range(0,len(hog)-1):
        for j in range(0,len(hog[0])-1):
            temp=[]
            temp=temp+hog[i][j]
            temp=temp+hog[i+1][j]
            temp=temp+hog[i][j+1]
            temp=temp+hog[i+1][j+1]
            s=0
            for i1 in range(0,len(temp)):
                s=s+(temp[i1]*temp[i1])
            sq=math.sqrt(s)
            for i1 in range(0,len(temp)):
                temp[i1]=temp[i1]/sq
            out=out+temp
    print('Length of HOG vector =', len(out))
    return out

#Implementing Gaussian Blurring
def blur(orimg, predmask):
    ker=[[1,1,1,1,1],[1,2,2,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,1,1,1,1]]
    out=copy.deepcopy(orimg)
    for i in range(5,len(predmask)-5):
        for j in range(5,len(predmask[0])-5):
            if(predmask[i][j]==0):
                temp0=0
                temp1=0
                temp2=0
                for k in range(-2,3):
                    for l in range(-2,3):
                        temp0=temp0+orimg[i+k][j+l][0]*ker[k+2][l+2]
                        temp1=temp1+orimg[i+k][j+l][1]*ker[k+2][l+2]
                        temp2=temp2+orimg[i+k][j+l][2]*ker[k+2][l+2]

                out[i][j][0]=min(int(temp0/35),255)
                out[i][j][1]=min(int(temp1/35),255)
                out[i][j][2]=min(int(temp2/35),255)

    return out

#Finally creating our output image
def predict_output(orig_img, model):
    #Obtaining mask for given image using HOGS and ML model
    mask=clf.predict(np.array(hog_feature(orig_img)).reshape(1, -1))
    temp_mask = mask.reshape(30,30)
    resized_mask=cv2.resize(temp_mask, (orig_img.shape[1],orig_img.shape[0]), interpolation = cv2.INTER_AREA)
    
    #Obtaining the image without background blurring
    ans= np.zeros(orig_img.shape)
    resized_mask=resized_mask/255
    for i in range(orig_img.shape[0]):
        for j in range(orig_img.shape[1]):
            for k in range(orig_img.shape[2]):
                ans[i][j][k] = orig_img[i][j][k] * resized_mask[i][j]
    ans = ans.astype('uint8')
    
    #Obtaining the original background and final image with blurred background
    final_img = blur(orig_img, resized_mask)
    
    #Also return the Hog image
    temp, hog_image = hog(orig_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return [final_img, ans, resized_mask, hog_image]


#Taking an example image
input_img = X_images[34]
[final_img, ans, resized_mask, hog_image] = predict_output(input_img, clf)

plt.imshow(input_img)
plt.title('Original Image')
plt.savefig('Fig1_orig',dpi=600)

plt.imshow(hog_image,cmap='gray')
plt.title('HOG Features')
plt.savefig('Fig1_hog',dpi=600)

plt.imshow(resized_mask,cmap='gray')
plt.title('resized_mask')
plt.savefig('Fig1_mask',dpi=600)

plt.imshow(final_img)
plt.title('Final Obtained Image')
plt.savefig('Fig1_final',dpi=600)