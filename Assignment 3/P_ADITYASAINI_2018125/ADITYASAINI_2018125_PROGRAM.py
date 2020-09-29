import matplotlib.pyplot as plt
import numpy as np
import cv2

## Question 1a Solution
print("Solution to 1a goes here")

def hist(img, L=256):
    ans = [0]*L
    for pixel in img.flatten():
        ans[pixel]+=1
    return ans

input_img = cv2.imread('ADITYASAINI_2018125_Q1_cameraman_InputImage.tif', cv2.IMREAD_GRAYSCALE)

input_hist=hist(input_img)
plt.subplot(1,2,1)
plt.imshow(input_img, cmap='gray')
plt.title('Input image')

plt.subplot(1,2,2)
plt.stem(input_hist, use_line_collection=True)
plt.title('Input image histogram')
plt.colorbar(orientation="horizontal",ticks=[])

plt.show()

print("Solution to 1b goes here")

specified_img = cv2.imread('ADITYASAINI_2018125_Q1_SpecifiedImage.jpg', cv2.IMREAD_GRAYSCALE)
plt.subplot(1,2,1)
plt.imshow(specified_img, cmap='gray')
plt.title('Specified Image')

plt.subplot(1,2,2)
specified_hist=hist(specified_img)
plt.stem(specified_hist, use_line_collection=True)
plt.title('Specified Image Histogram')
plt.colorbar(orientation="horizontal",ticks=[])

plt.show()

print("Solution to 1c goes here")

def compute_transfer(img_hist, img, L=256):
    cumsum=[0] * L
    cumsum[0] = img_hist[0]
    
    for idx in range(1,len(cumsum)):
        cumsum[idx]=cumsum[idx-1] + img_hist[idx]
    
    num_pixels= len(img.flatten())
    cumsum=[round( ((L-1)/num_pixels) * val) for val in cumsum]
    
    return cumsum
specified_transfer = compute_transfer(specified_hist, specified_img)
input_transfer = compute_transfer(input_hist, input_img)

plt.subplot(1,2,1)
plt.stem(input_transfer,use_line_collection=True)
plt.xlabel('r')
plt.ylabel('s=T(r)')
plt.subplot(1,2,2)
plt.stem(specified_transfer, use_line_collection=True)
plt.xlabel('z')
plt.ylabel('G(z)')
plt.show()

print("Solution to 1d goes here")
def compute_mapping(input_transfer, specified_transfer):
    specified_table={}
    
    for z in range(len(specified_transfer)):
        #When more than one value of G(z) satisfies s, choose the smallest value of z by convention.
        g_z = specified_transfer[z]
        
        if(not(g_z in specified_table)):
            specified_table[g_z] = z
    
    mapping = [0] * len(input_transfer)
    for r in range(len(input_transfer)):
        
        diff= 1000
        t_r = input_transfer[r] #t_r=s
        for g_z in specified_table:
            
            if abs(t_r-g_z) < diff:
                diff= abs(t_r-g_z)
                z = specified_table[g_z]
                mapping[r] = z
    
    return mapping

mapping=compute_mapping(input_transfer, specified_transfer)
plt.stem(mapping, use_line_collection=True)
plt.xlabel('r')
plt.ylabel('z')
plt.title('Mapping between r & z')
plt.show()


print("Solution to 1e goes here")
def compute_output_image(input_img, mapping):    
    flattened_img = input_img.flatten()
    
    output=np.zeros(len(flattened_img))
    
    for idx in range(len(flattened_img)):
        pixel=flattened_img[idx]
        output[idx] = mapping[pixel]
    return output

output=compute_output_image(input_img,mapping)
output=output.reshape(input_img.shape[0], input_img.shape[1])
output=output.astype('uint8')
output_hist=hist(output)

plt.subplot(2,3,1)
plt.imshow(input_img, cmap='gray')
plt.title('Input Image')
plt.subplot(2,3,2)
plt.imshow(specified_img, cmap='gray')
plt.title('Specified Image')
plt.subplot(2,3,3)
plt.imshow(output, cmap='gray')
plt.title('Output Image')
plt.subplot(2,3,4)
plt.stem(input_hist, use_line_collection=True)
plt.title('Input Image Histogram')
plt.colorbar(orientation="horizontal",ticks=[])
plt.subplot(2,3,5)
plt.stem(specified_hist, use_line_collection=True)
plt.title('Specified Image Histogram')
plt.colorbar(orientation="horizontal",ticks=[])
plt.subplot(2,3,6)
plt.stem(output_hist, use_line_collection=True)
plt.title('Output Image Histogram')
plt.colorbar(orientation="horizontal",ticks=[])
plt.show()

plt.subplot(1,2,1)
plt.imshow(output, cmap='gray')
plt.title('Output Image')
plt.subplot(1,2,2)
plt.stem(output_hist,use_line_collection=True)
plt.title('Output Image histogram')
plt.show()


print("Solution to 2 given in ADITYASAINI_2018125_Q2.pdf")
print("Solution to 3a goes here")
def conv2d(input_matrix, kernel,same=False):
    output = np.zeros([input_matrix.shape[0]+kernel.shape[0]-1, input_matrix.shape[1]+kernel.shape[1]-1])
    #Rotating kernel by 180
    rotated_kernel = np.zeros(kernel.shape)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            rotated_kernel[i][j] = kernel[kernel.shape[0]-1-i][kernel.shape[1]-1-j]
    
    padded_img = np.zeros([input_matrix.shape[0] + 4, input_matrix.shape[1] + 4])
    padded_img[2:2+input_matrix.shape[0], 2:2+input_matrix.shape[1]] = input_matrix
    
    for centre_row in range(output.shape[0]):
        for centre_col in range(output.shape[1]):
            padding_submat = padded_img[centre_row:kernel.shape[0]+centre_row, centre_col:kernel.shape[1]+centre_col]
            product = rotated_kernel.flatten().dot(padding_submat.flatten())
            output[centre_row][centre_col] = product
            
    if same==True:
        limit_row=output.shape[0]-input_matrix.shape[0]
        limit_col=output.shape[1]-input_matrix.shape[1]
        output=output[limit_row-1:output.shape[0]-limit_row+1,limit_col-1:output.shape[1]-limit_col+1]
        
    return output

kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
input_img=np.array([[6,7],[8,9]])
conv_output=conv2d(input_img,kernel)
print("\nThe convoluted output is:")
print(conv_output)
print("\n")

print("Solution to 3b given in ADITYASAINI_2018125_Q3_B.pdf")

print("Solution to 4a goes here")
img=plt.input_img = cv2.imread('ADITYASAINI_2018125_Q4_chandrayaan_InputImage.jpg', cv2.IMREAD_GRAYSCALE)
w_xy=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
print("\nThe values in filter/kernel w(x,y) are")
print(w_xy)
print("\n")

print("Solution to 4b goes here")
sharpened_img = conv2d(img, w_xy, True)
#Normalizing the pixel values
sharpened_img[sharpened_img>255]=255
sharpened_img[sharpened_img<0]=0
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('Original image')
plt.subplot(1,2,2)
plt.imshow(sharpened_img,cmap='gray')
plt.title('Sharpened image')
plt.show()