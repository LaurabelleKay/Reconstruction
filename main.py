import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import sys


import feature
import geometry
import camera
import utils


def main():
    path = "ice/"
    files = [f for f in os.listdir(path)]
    images = []

    # Load all the images in the folder specified by 'path'
    for f in files:
        images.append(cv.imread(path + f, cv.IMREAD_COLOR))

    height, width = images[0].shape[:2]

    # Find descriptors and key points for the images
    des, kp = feature.find_features(images)

    coords1, coords2, matches, des1, kp2 = feature.match_features(
        des[0], des[1], kp[0], kp[1])

    img = cv.drawMatches(images[0], kp[0], images[1],
                         kp[1], matches, None, flags=2)
    plt.imshow(img)
    # plt.show()

    mm = (height + width) / 2

    co = coords2

    # Normalise our feature coordinates to be in the range [-1, 1]
    # FIXME: Normalise rows ans columns separately!!!
    coords1 = (coords1 - np.ones(len(coords1[0]))*width)/mm
    coords2 = (coords2 - np.ones(len(coords1[0]))*height)/mm

    # Convert coordinates to homogenous coordindates
    coords1 = np.vstack((coords1, np.ones(len(coords1[0]))))
    coords2 = np.vstack((coords2, np.ones(len(coords2[0]))))

    # Calculate the fundamental matrix
    F = geometry.fundamental_matrix(coords1, coords2)

    # Calculate the epipole for the left and right views
    epipole_left = geometry.epipole(F.T)
    epipole_right = geometry.epipole(F)

    # Calculate the homography that maps the left epipole to the right
    H = geometry.homography(epipole_right, F)

    # Projection matrix for the first view is [I|0]
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

    # Perform triangulation of the first 2 views
    coords3D = geometry.triangulate(coords1, coords2, views[0].P, views[1].P)

    coords = []
    coords.append(coords1)
    coords.append(coords2)

    add_des = []
    add_kp = []

    add_des.append(des[0])
    add_kp.append(kp[0])

    add_des.append(des1)
    add_kp.append(kp2)

    for i in range(1, 2):
        pts0, pts1, matches, des1, kp1 = feature.match_features(add_des[i], des[i + 1], add_kp[i], kp[i + 1])
        
        if(len(matches) < 6):
            print("Not enough matches to estimate pose")
            sys.exit(0)

        add_des.append(des1)
        add_kp.append(kp2)

        #pts0 = (pts0 - np.ones(len(pts0[0])) * height)/mm
        coords_temp = co.T
        print(coords_temp)
        print("\n")
        pts_temp = pts0.T
        print(pts_temp)
        vals = []
        for j in range(0, len(pts0[0])):
            queryx = pts_temp[j, 0]
            queryy = pts_temp[j, 1]

            mask = (coords_temp == (queryx, queryy))
            val = coords_temp[mask == True]
            if val.size != 0:
                vals.append(val)

        print(val)

        # TODO: Additional match (get add match but also normal matches)
        # TODO: Calculate P using add matches
        # TODO: Triangluate other matches using normal triangulation (IF YOU HAVE TIME)
        # TODO: Bundle adjust (add matches with previous one as well)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(coords3D[0], coords3D[1], coords3D[2], 'b.')
    ax.view_init(elev=125, azim=90)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    plt.show()


main()
