import cv2
import numpy as np
from scipy import spatial


def get_putative_matching_keypoints(
    test_keypoints, test_descriptors,
    train_keypoints, train_descriptors,
    max_distance=0.8
):
    test_descriptor_tree = spatial.cKDTree(test_descriptors)
    _, matches = test_descriptor_tree.query(train_descriptors, distance_upper_bound=max_distance)

    test_kp_count = test_keypoints.shape[0]
    train_kp_count = train_keypoints.shape[0]

    test_matching_keypoints = np.array(
        [test_keypoints[matches[i], ] for i in range(train_kp_count) if (
            matches[i] != test_kp_count)]
    )
    train_matching_keypoints = np.array(
        [train_keypoints[i, ] for i in range(train_kp_count) if matches[i] != test_kp_count])

    return test_matching_keypoints, train_matching_keypoints, matches


def count_tentative_matches(desc_0, desc_1, th=0.9):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_0, desc_1, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [False for _ in range(len(matches))]
    # SNN ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            matchesMask[i] = True
    tentative_matches = [m[0] for i, m in enumerate(matches) if matchesMask[i]]
    return len(tentative_matches)
