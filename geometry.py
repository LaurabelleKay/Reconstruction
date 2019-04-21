import numpy as np
from numpy.linalg import inv, svd, eig
from scipy.optimize import least_squares, fmin


def fundamental_matrix(coords1, coords2):
    """Compute the fundamental matrix based on the matched features in the
    image pair using the 8-point algorithm

    Arguments:
        coords1 -- Feature coordinates for image 1
        coords2 -- Feature coordinates for image 2

    Returns:
        F -- The fundamental matrix 
    """

    num_features = len(coords1[0])

    # The matrix used to solve mFm' = 0, where m and m' are the image
    # features, and F is the fundamental matrix
    A = np.zeros((num_features, 9))

    # Fill out our equation matrix
    A[:, 0] = np.transpose(coords1[0, :] * coords2[0, :])
    A[:, 1] = np.transpose(coords1[1, :] * coords2[0, :])
    A[:, 2] = np.transpose(coords2[0, :])
    A[:, 3] = np.transpose(coords1[0, :] * coords1[1, :])
    A[:, 4] = np.transpose(coords1[1, :] * coords2[1, :])
    A[:, 5] = np.transpose(coords2[1, :])
    A[:, 6] = np.transpose(coords1[0, :])
    A[:, 7] = np.transpose(coords1[1, :])
    A[:, 8] = np.ones(num_features)

    # Solve for F
    U, s, Vh = svd(A)
    V = np.transpose(Vh)

    # The right null space of A is the estimate for F, reshape to be 3 x 3
    F = np.reshape(V[:, 8], (3, 3))
    U, s, V = svd(F)

    # Enforce that F is of rank 2
    s[2] = 0
    S = np.diag(s)

    F = np.dot(U, np.dot(S, V))

    return F


def fundamental_minimise(f, coords1, coords2):

    F = np.asarray([np.transpose(f[0:3]), np.transpose(f[3:6]), [
                   f[6], -(-f[0]*f[4]+f[6]*f[2]*f[4]+f[3]*f[1]-f[6]*f[1]*f[5])/(-f[3]*f[2]+f[0]*f[5]), 1]])
    print(F)

    num_features = len(coords1[0])

    err = np.zeros((num_features), 1)
    for i in range(0, num_features):
        err[i] = np.dot(coords2[:, i]), np.dot(F, np.transpose(coords1[:, i]))

    return err.flatten()


def epipole(F):
    """Calculates the epipoles on 2 images based on the fundamental matrix

    Arguments:
        F -- The fundamental matrix

    Returns:
        epi -- The epipole for this image
    """

    # Fe = 0 -> e is the right null space of F
    U, S, V = np.linalg.svd(F)
    epi = V[:, 2]
    epi = epi / epi[2]

    return epi


def homography(epi, F):
    """Calculate the homography matrix that maps one epipole to the other
    using the equation [e] x F

    Arguments:
        epi -- Epipole for the second view
        F   -- The fundamental matrix

    Returns:
        H -- The homography matrix for the pair
    """

    # Use the skew-symmetric form to calculate the cross product
    epi = np.asarray([[0, -epi[2], epi[1]],
                      [epi[2], 0, -epi[0]],
                      [-epi[1], epi[0], 0]])

    # Matrix multiply
    H = np.dot(epi, F)
    H = H * np.sign(np.trace(H))

    return H

# ! Need some way to verify that this works!!


def reference_frame(epi, H):
    p = np.sum(np.divide(np.eye(3) - H,
                         np.transpose(np.asarray([epi, epi, epi]))), axis=0)/3
    #p = fmin(init_plane, np.append(p, 1), xtol=1e-25,ftol=1e-25, args=(h.real, epi.real))
    p = p[0:3]
    return p


def triangulate(coords1, coords2, P1, P2):
    """Triangulate the 3D points for the features using the projection matrices

    Arguments:
        coords1 -- feature coordinates for image 1
        coords2 -- feature coordinates for image 2
        P1 -- projection matrix for view 1
        P2 -- projection matrix for view 2

    Returns:
        coords3D -- The 3D coordinates for this pair of views
    """
    num_points = len(coords1[0])
    coords3D = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (P1[2, :] * coords1[0, i] - P1[0, :]),
            (P1[2, :] * coords1[1, i] - P1[1, :]),
            (P2[2, :] * coords2[0, i] - P2[0, :]),
            (P2[2, :] * coords2[1, i] - P2[1, :])
        ])

        # For a 3D point X, x1 = P1 * X and x2 = P2 * x, to find X, solve for AX = 0
        _, _, Vh = svd(A)
        V = np.transpose(Vh)

        # Right null space of A is the solution for X
        coords = V[:, V.shape[0]-1]

        # Convert to homogenous
        coords = coords/coords[3]

        coords3D[:, i] = coords

        # Minimise the reprojection error for the caclulated 3D points
        #x = fmin(reprojection_error, x, xtol=1e-25, ftol=1e-25, full_output=0, args=(
        #    P1, P2, coords1[0:1,i], coords2[0:1,i]), disp=False)

    return coords3D


def reprojection_error(coords3D, P1, P2, coords1, coords2):
    # Project the 3D points to 2D using the calculated projection matrix
    coords_1 = P1[0:2, :].dot(coords3D) / P1[2, :].dot(coords3D)

    # Calculate the error between the projected point and the actual point coordinates
    err1 = np.power((coords_1 - coords1), 2)

    coords_2 = P2[0:2, :].dot(coords3D) / P2[2, :].dot(coords3D)
    err2 = np.power((coords_2 - coords2), 2)

    return sum(err1 + err2)
