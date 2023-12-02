import os
import cv2
import numpy as np

from tf.transformations import euler_from_matrix


point2D = np.array([[458, 228],
                    [428, 226], 
                    [397, 227],
                    [363, 228],
                    [335, 229],
                    [299, 226]], dtype=np.float32)
point3D = np.array([[1.933904, 0.055708, -0.033737],
                    [1.931475, 0.178497, -0.033824],
                    [1.935202, 0.275420, -0.034085],
                    [1.938919, 0.423817, -0.034608],
                    [1.932009, 0.515873, -0.034870],
                    [1.937722, 0.620643, -0.035480]
                    ], dtype=np.float32)


def calibrate(points2D=None, points3D=None):
    # Load corresponding points
    # Check points shape
    assert(points2D.shape[0] == points3D.shape[0])
    if not (points2D.shape[0] >= 5):
        print('PnP RANSAC Requires minimum 5 points')
        return

    # Obtain camera matrix and distortion coefficients
    camera_matrix = np.array([[919.84, 0.0, 667.42],
                                [0.0, 919.97, 375.73],
                                [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([[-0.0, 0.0, 0.0, 0.0, -0.00]])

    # Estimate extrinsics
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(points3D, 
        points2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # Compute re-projection error.
    points2D_reproj = cv2.projectPoints(points3D, rotation_vector,
        translation_vector, camera_matrix, dist_coeffs)[0].squeeze(1)
    assert(points2D_reproj.shape == points2D.shape)
    error = (points2D_reproj - points2D)[inliers].reshape(point2D.shape)  # Compute error only over inliers.
    rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
    print('Re-projection error before LM refinement (RMSE) in px: ' + str(rmse))

    # Refine estimate using LM
    if not success:
        print('Initial estimation unsuccessful, skipping refinement')
    elif not hasattr(cv2, 'solvePnPRefineLM'):
        print('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
    else:
        assert len(inliers) >= 3, 'LM refinement requires at least 3 inlier points'
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(points3D[inliers],
            points2D[inliers], camera_matrix, dist_coeffs, rotation_vector, translation_vector)

        # Compute re-projection error.
        points2D_reproj = cv2.projectPoints(points3D, rotation_vector,
            translation_vector, camera_matrix, dist_coeffs)[0].squeeze(1)
        assert(points2D_reproj.shape == points2D.shape)
        error = (points2D_reproj - points2D)[inliers].reshape(point2D.shape)  # Compute error only over inliers.
        rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
        print('Re-projection error after LM refinement (RMSE) in px: ' + str(rmse))

    # Convert rotation vector
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    euler = euler_from_matrix(rotation_matrix)
    
    # Save extrinsics
    np.savez(os.path.join('data', 'extrinsics.npz'),
        euler=euler, R=rotation_matrix, T=translation_vector.T)

    # Display results
    print('Euler angles (RPY):', euler)
    print('Rotation Matrix:', rotation_matrix)
    print('Translation Offsets:', translation_vector.T)



calibrate(points2D=point2D, points3D=point3D)