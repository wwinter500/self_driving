#cv2.inRange() for color selection
#cv2.fillPoly() for regions selection
#cv2.line() to draw lines on an image given endpoints
#cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
#cv2.bitwise_and() to apply a mask to an image
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

solidWhiteRightimg = mpimg.imread('test_images/solidWhiteRight.jpg')
solidWhiteCurveimg = mpimg.imread('test_images/solidWhiteCurve.jpg')
solidYellowCurveimg = mpimg.imread('test_images/solidYellowCurve2.jpg')
swtichWhiteimg = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    slopeArr = []
    startP = []
    endP = []
    index = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            localslope = (y2 - y1) / (x2 - x1)
            slopeArr.append([localslope, index])
            startP.append([x1, y1])
            endP.append([x2, y2])

            index = index + 1
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    leftCount = 0
    rightCount = 0
    sortedArr = np.array(sorted(slopeArr, key=lambda x: x[0]))
    #print(sortedArr)
    for i in range(len(sortedArr)):
        if(sortedArr[i][0] < 0):
            leftCount = leftCount + 1
        if(sortedArr[i][0] > 0):
            rightCount = rightCount + 1
    
    leftM = leftCount / 2
    rightM = len(sortedArr) - rightCount/2
    leftC = 0
    rightC = 0
    leftSum = 0
    rightSum = 0
    for i in range(len(sortedArr)):
        if(sortedArr[i][0] < 0):
            if(leftCount > 0 and math.fabs((sortedArr[leftM][0] - sortedArr[i][0]) / sortedArr[leftM][0]) < 0.25):
                #print("__",sortedArr[i][0])
                leftSum = leftSum + sortedArr[i][0]
                leftC = leftC + 1
        if(sortedArr[i][0] > 0):
            if(rightCount > 0 and math.fabs((sortedArr[rightM][0] - sortedArr[i][0]) / sortedArr[rightM][0]) < 0.25):
                #print("__",sortedArr[i][0])
                rightSum = rightSum + sortedArr[i][0]
                rightC = rightC + 1
    
    leftSlope = sortedArr[leftM][0]
    rightSlope = sortedArr[rightM][0]
    if(leftCount != 0):
        leftSlope = leftSum / leftC
    if(rightCount != 0):
        rightSlope = rightSum / rightC
    
    #print("Average",leftSlope,"_" ,rightSlope)
    
    lp = int(sortedArr[leftM][1])
    rp = int(sortedArr[rightM][1])
    
    x0 = startP[lp][0]
    y0 = startP[lp][1]
    
    x1 = startP[rp][0]
    y1 = startP[rp][1]
    #print(x0,"__", y0)
    
    startLY = 512
    startLX = int((startLY - y0) / leftSlope + x0)
    endLY = 350
    endLX = int((endLY - y0) / leftSlope + x0)
    
    cv2.line(img,(startLX,startLY),(endLX,endLY),(255,0,0),10)
    
    startRY = 512
    startRX = int((startRY - y1) / rightSlope + x1)
    endRY = 350
    endRX = int((endRY - y1) / rightSlope + x1)
    
    cv2.line(img,(startRX,startRY),(endRX,endRY),(255,0,0),10)
            

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def laneDetection(img):
    grayimg = grayscale(img)
    #plt.imshow(grayimg, cmap='gray')
    
    smoothImg = gaussian_blur(grayimg, 3)
    #plt.imshow(smoothImg, cmap='gray')

    low_threshold = 50
    high_threshold = 150
    cannyimg = canny(smoothImg, low_threshold, high_threshold)
    #plt.imshow(cannyimg, cmap='gray')

    vertics = np.array([[80,512],[880,512],[480,280]],np.int32)
    regionImg = region_of_interest(cannyimg, [vertics])
    #plt.imshow(regionImg, cmap='gray')

    rho = 2
    theta = np.pi/180
    threshold = 10
    min_line_len = 10
    max_line_gap = 2
    houghimg = hough_lines(regionImg, rho, theta, threshold, min_line_len, max_line_gap)
    #plt.imshow(houghimg, cmap='gray')

    line_edge = weighted_img(houghimg, img, α=0.8, β=1., λ=0.)
    #plt.imshow(line_edge, cmap='gray')
    return line_edge

#resultimage = laneDetection(img)
#plt.imshow(line_edge)

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = laneDetection(image)
    
    return result

white_output = "white.mp4"
clip1 = VideoFileClip("solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#outputVideo = white_output.encode(encoding='UTF-8')
codecStr = 'libx264'
%time clip1.write_videofile(white_output, codec = codecStr,audio=False)