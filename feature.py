import cv2 as cv
import numpy as np
import sys

def find_features(images):
    """Finds the descriptors and their keypoints for every image using the ORB feature detector

    Arguments:
        images -- The sequence of images 
    """

    #TODO: Change this back to 50000
    orb = cv.ORB_create(10000, 1.2, nlevels=9)
    key_points = []
    des = []
    matches = []
    for i in range(len(images)):
        kp, d = orb.detectAndCompute(images[i], None)
        key_points.append(kp)
        des.append(d)

    return des, key_points

def match_features(des0, des1, kp1, kp2):
    """Find the descriptors which match in a pair of images
    
    Arguments:
        des0 -- Descriptors for image 1
        des1 -- Descriptors for image 2
        kp1  -- Key points for image 1
        kp2  -- Key points for image 2
    
    Returns:
        matches -- The list of matches for this image pair
    """

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    des0 = np.float32(des0)
    des1 = np.float32(des1)

    bf = cv.BFMatcher()
    matches = flann.knnMatch(des0, des1, k=2)

    good_matches = []
    for m, n in  matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    

    pts0 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts1 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    des1 = ([des1[m.trainIdx] for m in good_matches])
    kp2 = ([kp2[m.trainIdx] for m in good_matches])

    _, mask = cv.findHomography(pts0, pts1, cv.RANSAC, 2)

    mask = mask.ravel()

    pts0 = pts0[mask == 1]
    pts1 = pts1[mask == 1]

    if len(pts0) < 8:
        print("Not enough matches")
        sys.exit(0)

    d1 = []
    k2 = []

    for i in range(0, len(mask)):
        if(mask[i] == 1):
            d1.append(des1[i])
            k2.append(kp2[i])

    print(len(pts0))
    return pts0.T, pts1.T, good_matches, d1, k2
