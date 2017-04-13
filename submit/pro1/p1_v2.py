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

    sortedArr = np.array(sorted(slopeArr, key=lambda x: x[0]))

    gapY = 50
    y0 = 550
    y1 = 300
    itr = y0 / gapY
    for i in itr:
        #search on start position
        localslopeArr = []
        for j in range(len(sortedArr)):
            p = sortedArr[j][1]
            if( y0 - (i+1)*gapY <= startP[p][1] <= y0 - i*gapY ):
                localslopeArr.append([sortedArr[j][0], sortedArr[1], startP[p][1]])
            if( y0 - (i+1)*gapY <= endP[p][1] <= y0 - i*gapY):
                localslopeArr.append([sortedArr[j][0], sortedArr[1], endP[p][1]])

        print(localslopeArr)
        print("------------")
    
    cv2.line(img,(startRX,startRY),(endRX,endRY),(255,0,0),10)