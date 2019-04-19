import cv2 as cv
import numpy as np
import sys

# ?Might be beneficial to get SIFT to work
def find_features(images):
    """Finds the descriptors and their keypoints for every image using the ORB feature detector

    Arguments:
        images -- The sequence of images 
    """

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
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        print("Not enough matches")
        sys.exit(0)

    print(len(good_matches))
    pts0 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts1 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    _, mask = cv.findHomography(pts0, pts1, cv.RANSAC, 2)

    mask = mask.ravel()

    pts0 = pts0[mask == 1]
    pts1 = pts1[mask == 1]

    return pts0.T, pts1.T, good_matches


def features_to_array(matches, kp1, kp2):
    """Match the keypoints in the 2 images based on the matches
    
    Arguments:
        matches -- The list of matches for the pair of images
        kp1 -- Keypoints for the first image
        kp2 -- Keypoints for the second image
    
    Returns:
        coords 1 -- x & y coordinates of features in image 1
        coords 2 -- x & y coordinates of features in image 2
    """
    num_matches = len(matches)

    coords1 = np.zeros((2, num_matches))
    coords2 = np.zeros((2, num_matches))


    for i in range(0, num_matches):
        q = matches[i].queryIdx # Get the id for this match in the query image

        # Index the key points to get the x & y coordinates
        coords1[0][i] = kp1[q].pt[0]
        coords1[1][i] = kp1[q].pt[1]

        t = matches[i].trainIdx # Get the id for this match in thw target image
        coords2[0][i] = kp2[q].pt[0]
        coords2[1][i] = kp2[q].pt[1]

    return coords1, coords2
