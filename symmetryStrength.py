# Gonzalo Muradás Odriozola  02-2023
# Python 3.11.0
# KU Leuven

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import warnings

from sewar.full_ref import rase, vifp, msssim,mse, rmse, psnr, uqi, ssim, ergas, scc, sam

### OPERATIONS ###

# Returns a list with all the transformmed keypoints
def transformKeypoints(keypoints, rotationMatrix):
    result = []
    for keypoint in keypoints:
        rotatedPoint = rotationMatrix.dot(np.array(keypoint + (1,)))
        result.append((int(rotatedPoint[0]),int(rotatedPoint[1])))

    return result

# Copies the pixels = [0,0,0] on both sides of the image
def copyMask(img1, img2):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if (img1[i,j] == [0,0,0]).all():
                img2[i,j] = [0,0,0]
            if (img2[i,j] == [0,0,0]).all():
                img1[i,j] = [0,0,0]

# Returns the metrics for the similarities between two images
def getMetrics(title, img1, img2):
    result = pd.DataFrame({
        'MSE': [mse(img1, img2)],
        'RMSE': [rmse(img1, img2)],
        'PSNR': [psnr(img1, img2)],
        'UQI': [uqi(img1, img2)],
        'MSSSIM': [msssim(img1, img2)],  # For 'valid' mode, one must be at least as large as the other in every dimension
        'ERGAS': [ergas(img1, img2)],
        'SCC': [scc(img1, img2)],
        'RASE': [rase(img1, img2)],  # CANNOT DIVIDE BY 0
        'SAM': [sam(img1, img2)],
        'VIF':[vifp(img1, img2)],  # DOES NOT WORK WITH LOWER RESOLUTIONS
        'SSIM': [ssim(img1, img2)]
    }, index=[title])

    return result

# Returns the metrics for the similarities between two images, None if a metric cannot be calculated
def getMetricsWithErrorHandling(title, img1, img2):
    warnings.filterwarnings("ignore")

    m = 0
    r = 0
    p = 0
    u = 0
    ms = 0
    e = 0
    s = 0
    ra = 0
    sa = 0
    v = 0
    ssi = 0
    try:
        m = mse(img1,img2)
    except:
        m = None
    try:
        r = rmse(img1,img2)
    except:
        r = None
    try:
        p = psnr(img1,img2)
    except:
        p = None
    try:
        u = uqi(img1,img2)
    except:
        u = None
    try:
        ms = msssim(img1,img2)
    except:
        ms = None
    try:
        e = ergas(img1,img2)
    except:
        e = None
    try:
        s = scc(img1,img2)
    except:
        s = None
    try:
        ra = rase(img1,img2)
    except:
        ra = None
    try:
        sa = sam(img1,img2)
    except:
        sa = None
    try:
        v = vifp(img1,img2)
    except:
        v = None
    try:
        ssi = ssim(img1,img2)
    except:
        ssi = None
    result = pd.DataFrame({
        'MSE':    [m],
        'RMSE':   [r],
        'PSNR':   [p],
        'UQI':    [u],
        'MSSSIM': [ms],  
        'ERGAS':  [e],
        'SCC':    [s],
        'RASE':   [ra], 
        'SAM':    [sa],
        'VIF':    [v], 
        'SSIM':   [ssi]
    }, index=[title])

    return result

# Returns statistics for the strength of symmetry of the selected bounding box on the image
def symmetryStrengthStartVertex(imgPath, index, startVertex, bbWidthAndLength, rotationDegrees, onlyUQUI = False, resolutionFactor = None):
    # Reading the image
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Rotating and cropping
    rotationMatrix = cv2.getRotationMatrix2D((img.shape[1]//2,img.shape[0]//2),-rotationDegrees,1)
    img = cv2.warpAffine(img,rotationMatrix,(img.shape[1], img.shape[0]))
    img = img[startVertex[1]:startVertex[1]+bbWidthAndLength[1], startVertex[0]:startVertex[0]+bbWidthAndLength[0]]

    #Dividing in left and right
    leftHalf = img[:, :bbWidthAndLength[0]//2]
    rightHalf = img[:, bbWidthAndLength[0]//2:]
    rightHalf = cv2.flip(rightHalf, 1)

    # Copying the black pixels
    copyMask(rightHalf,leftHalf)

    # Reducing resolution
    if resolutionFactor is not None:
        # Reducing resolution
        leftHalf = cv2.resize(leftHalf, (leftHalf.shape[1]//resolutionFactor, leftHalf.shape[0]//resolutionFactor))
        rightHalf = cv2.resize(rightHalf, (leftHalf.shape[1]//resolutionFactor, leftHalf.shape[0]//resolutionFactor))

    if onlyUQUI:
        return uqi(leftHalf, rightHalf)
    else:
        return getMetricsWithErrorHandling(index, leftHalf,rightHalf)
    
# Remove negative coordinates from list
def removeNegativeCoordinates(points, height, width):
    result = []
    for point in points:
        x = point[0]
        y = point[1]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > width:
            x = width
        if y > height:
            y = height
        result.append((x,y))
    return result

# Crops image with center of rectangle, dimensions and rotation
def symmetryStrength(img, centerX, centerY, width, height, rotation, axisHorizontal, index = "None", onlyUQUI = False, display = False):
    # Making width and height even
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1

    # Calculating the bounding box
    pts = [(centerX-width/2 , centerY-height/2), (centerX+width/2 , centerY-height/2), 
           (centerX+width/2 , centerY+height/2), (centerX-width/2 , centerY+height/2)]

    # Removing negative coordinates
    pts = removeNegativeCoordinates(pts, img.shape[0], img.shape[1])

    # Rotating and cropping image
    rotationMatrix = cv2.getRotationMatrix2D((centerX,centerY),rotation,1)
    img = cv2.warpAffine(img,rotationMatrix,(img.shape[1], img.shape[0]))
    img = img[int(pts[0][1]):int(pts[0][1])+int(height), int(pts[0][0]):int(pts[0][0]+width)]

    # Flipping image if axis is horizontal
    if axisHorizontal:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    if img is None:
        if onlyUQUI:
            return -1
        else:
            return pd.DataFrame({
                'MSE':    [None],
                'RMSE':   [None],
                'PSNR':   [None],
                'UQI':    [None],
                'MSSSIM': [None],  
                'ERGAS':  [None],
                'SCC':    [None],
                'RASE':   [None], 
                'SAM':    [None],
                'VIF':    [None], 
                'SSIM':   [None]
            })

    # Dividing image
    leftHalf = img[:, :img.shape[1]//2]
    rightHalf = img[:, img.shape[1]//2:]
    rightHalf = cv2.flip(rightHalf, 1)

    # Making sure both sizes are the same
    if leftHalf.shape[1] > rightHalf.shape[1]:
        leftHalf = leftHalf[:,:-1]
    if leftHalf.shape[1] < rightHalf.shape[1]:
        rightHalf = rightHalf[:,:-1]
    if leftHalf.shape[0] > rightHalf.shape[0]:
        leftHalf = leftHalf[:-1]
    if leftHalf.shape[0] < rightHalf.shape[0]:
        rightHalf = rightHalf[:-1]

    # Copying the black pixels
    copyMask(rightHalf,leftHalf)

    # Display
    if display:
        diff = cv2.subtract(leftHalf, rightHalf)
        _, ax = plt.subplots(1,4, figsize=(15, 5))
        ax[0].imshow(img), ax[0].set_title(f'Bounding Box')
        ax[1].imshow(leftHalf), ax[1].set_title(f'Left half')
        ax[2].imshow(rightHalf), ax[2].set_title(f'Right half')
        ax[3].imshow(diff), ax[3].set_title(f'Difference')
        plt.show()

    if onlyUQUI:
        return uqi(leftHalf, rightHalf)
    else:
        return getMetricsWithErrorHandling(index, leftHalf,rightHalf)