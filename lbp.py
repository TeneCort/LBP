import cv2
import numpy as np
import matplotlib.pyplot as plt

# loads file, resize and convert to gray
def prepareFile():
    # load image from file
    filename = 'test1.jpg'
    img = cv2.imread(filename)
    # convert grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize
    scale_percent = 10 # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    return resized

# feed a binary and return a decimal
def convertBinToDecimal(b_num):
    b_num = b_num
    value = 0

    for i in range(len(b_num)):
        digit = b_num.pop()
        if digit == '1':
            value = value + pow(2, i)
    print("The decimal value of the number is", value)
    return value

def countElements(seq) ->dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist

# set the image we work with and create a copy 
resized = prepareFile()
copy = np.copy(resized)

# get width and height of the image
rows = resized.shape[0]
cols = resized.shape[1]

#initialise the 2D array we will store pixel info and the binary output for each pixel in the LBP
values = [
    [0, 0 ,0],
    [0, 0 ,0],
    [0, 0 ,0]]
binary = ''

# Iterating each col and row of the image except the first and last values
for col in range(cols-1):
    if(col != 0):
        for row in range(rows-1):
            if(row != 0):
                
                # for each pixel iterated, we put the pixel and it's immediate neighbors in a 2D array
                for x in range(3):
                    for y in range(3):
                        indexX = x - 1
                        indexY = y - 1
                        values[x][y] = resized[row - indexX][col - indexY]
                
                # getting the grayscale value of the iterated pixel
                threshold = values[1][1]

                # converting the grayscale values of each neighbor into a binary one
                for x in range(3):
                    for y in range(3):
                        if(values[x][y] < threshold):
                            values[x][y] = 0
                        else:
                            values[x][y] = 1

                # retrieving each binary value from the 2D array and concatenate into a single value
                for x in range(3):
                    for y in range(3):
                        if((x is not 1) or (y is not 1)):
                            binary += str(values[x][y])
                
                # we change the value of the iterated pixel to the new image
                copy[row][col] = int(binary, 2)
                binary = ''
                

intensities = []

#copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2RGB)

for x in range(0, 10):
    for y in range(0, 10):
        #resized[x][y] = 255
        if copy[x][y] == 2:
            print("2 found")
        print(copy[x][y])
        intensities.append(copy[x][y])

hist = countElements(intensities)

#hist = []
print(hist)
'''
for i in range(256):
       hist.append(0)

for i in intensities:
    hist[i] += 1
'''
plt.hist(hist)
plt.show()

#cv2.imshow('image',resized)
cv2.imshow('copy',copy)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',gray)
    cv2.destroyAllWindows()


'''
var result

for int i = 0; i< hist1.length; i++{
    result += (hist1[i] - hist2[i])^2
}

D = sqr(result)
'''