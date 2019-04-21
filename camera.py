import numpy as np
from numpy.linalg import inv, svd, eig


class Camera(object):

    def __init__(self, P=None, K=None, T=None):

        self.P = P
        K = np.asarray([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        self.K = K
        self.T = T

    def projection_matrix(self, H, epi, p, num_images):

        P = np.zeros((3, 4))

        # !Reshape p & epi so we can multiply them
        e = np.reshape(epi, (3, 1))
        p_t = np.reshape(p, (1, 3))

        # P = [H + p*epi| epi]
        P[:, :3] = H + e.dot(p_t)
        P[:, 3] = epi
        P[:, :] = P[:, :] / P[2, 2]  # Convert to homogenous coordinates

        self.P = P

    def pose_estimation(self, P, coords1, coords2, coords3D):
        num_points = len(coords1[0])

        A = np.zeros((2*num_points, 12))

        for i in range(0, num_points):
            A[2 * i, 0:4] = np.transpose(coords3D[:, i])
            A[2 * i, 8:12] = -coords1[0, i] * np.transpose(coords3D[:, i])
            A[2 * i + 1, 4:8] = np.transpose(coords3D[:, i])
            A[2 * i + 1, 8:12] = -coords1[1, i] * np.transpose(coords3D[:, i])

        U, s, Vh = svd(A)
        V = np.transpose(Vh)

        V = V[0:12, 11]
        V = V/V[10]
        V = np.delete(V, 10)

        P = np.zeros((3, 4))
        P[0, :] = V[0:4]
        P[1, :] = V[4:8]
        P[2, :] = np.append(np.append(V[8:10], 1), V[10])

        self.P = P

    def project(self, points3D):
        """Apply the projection matrix for this camera to convert the 3D 
        coordinates to the 2D image space

        Arguments:
            points3D -- 3D coordinates of the features

        Returns:
            points2D -- The 2D projection of the points
        """

        points2D = np.zeros(len(points3D[0]))

        return points2D
