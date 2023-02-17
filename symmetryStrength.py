# Gonzalo Muradás Odriozola  02-2023
# Python 3.11.0
# KU Leuven

import numpy as np
import pandas as pd
import cv2

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, sam, msssim
# from sewar.full_ref import rase, vifp

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
        for j in range(img1.shape[0]):
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
        'MSSSIM': [msssim(img1, img2)],
        'ERGAS': [ergas(img1, img2)],
        'SCC': [scc(img1, img2)],
        # 'RASE': [rase(img1, img2)],  # CANNOT DIVIDE BY 0
        'SAM': [sam(img1, img2)],
        # 'VIF':[vifp(img1, img2)],  # DOES NOT WORK WITH LOWER RESOLUTIONS
        'SSIM': [ssim(img1, img2)]
    }, index=[title])

    return result

### MAIM FUNCTION ###
# Returns statistics for the strength of symmetry of the selected bounding box on the image
def symmetryStrength(imgPath, index, startVertex, bbWidthAndLength, rotationDegrees, onlyUQUI = False, resolutionFactor = None):
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
        return getMetrics(index, leftHalf,rightHalf)