import numpy as np

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
        p_t =  np.reshape(p, (1, 3))

        # P = [H + p*epi| epi]
        P[:, :3] = H + e.dot(p_t)
        P[:, 3] = epi
        P[:, :] = P[:, :] / P[2, 2] # Convert to homogenous coordinates

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
