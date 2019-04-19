import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv


import feature
import geometry
import camera

def main():
    path = "dino/"
    files = [f for f in os.listdir(path)]
    images = []

    # Load all the images in the folder specified by 'path'
    for f in files:
        images.append(cv.imread(path + f, cv.IMREAD_COLOR))

    height, width = images[0].shape[:2]
    des, kp = feature.find_features(images) # FInd descriptors and key points for the images

    coords1, coords2, matches = feature.match_features(des[0], des[1], kp[0], kp[1])

    img = cv.drawMatches(images[0], kp[0], images[1], kp[1], matches, None, flags=2)
    plt.imshow(img)
    #plt.show()

    
    # coords1, coords2 = feature.features_to_array(matches, kp[0], kp[1])

    mm = (height + width) / 2
    

    # Convert coordinates to homogenous coordindates
    coords1 = np.vstack((coords1, np.ones(len(coords1[0]))))
    coords2 = np.vstack((coords2, np.ones(len(coords2[0]))))

    # TODO: Normalise our feature coordinates to be in the range [-1, 1]    
    coords1 = (coords1 - np.ones(len(coords1[0]))*width)/mm
    coords2 = (coords2 - np.ones(len(coords1[0]))*height)/mm
    
    # Calculate the fundamental matrix
    F = geometry.fundamental_matrix(coords1, coords2)

    # Calculate the epipole for the left and right views
    epipole_left = geometry.epipole(F.T)
    epipole_right = geometry.epipole(F)

    # Calculate the homography that maps the left epipole to the right
    H = geometry.homography(epipole_right, F)

    #Projection matric for the first view is [I|0]
    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    views = []
    view = camera.Camera()
    view.P = P
    views.append(view)

    # Calculate the plane at infinity
    p = geometry.reference_frame(epipole_right, H)

    # Estimate the initial projection matrix for view 2
    view = camera.Camera()
    view.projection_matrix(H, epipole_right, p, len(images))
    views.append(view)
    
    coords3D = geometry.triangulate(coords1, coords2, views[0].P, views[1].P)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(coords3D[0], coords3D[1], coords3D[2], 'b.')
    ax.view_init(elev=135, azim=90)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    #ax.set_xlim3d(-1, 1)
    #ax.set_ylim3d(-1, 1)
    #ax.set_zlim3d(-1, 1)

    plt.show()

main()
