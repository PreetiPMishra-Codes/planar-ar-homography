"""match.py - keypoint detection + matching (ORB / SIFT) with Lowe's ratio test."""
import argparse
import cv2
import numpy as np


def _make_detector(method, n_features):
    """Return (detector, BFMatcher norm). The norm must fit the descriptor type."""
    if method == "orb":
        return cv2.ORB_create(nfeatures=n_features), cv2.NORM_HAMMING
    if method == "sift":
        return cv2.SIFT_create(nfeatures=n_features), cv2.NORM_L2
    raise ValueError(f"unknown method: {method}")


def match_features(ref, frame, method="sift", ratio=0.75, n_features=2000):
    """
    Match keypoints between a reference image and a frame.

    Returns
    -------
    pts_ref, pts_frame : (N, 2) float32 arrays, row i of one matches row i of the other
    info : dict with keypoint and match counts (used later in evaluation)
    good : list of cv2.DMatch (only needed for drawing)
    """
    detector, norm = _make_detector(method, n_features)

    # Detectors expect grayscale
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if ref.ndim == 3 else ref
    g_frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    kp_r, des_r = detector.detectAndCompute(g_ref, None)
    kp_f, des_f = detector.detectAndCompute(g_frm, None)

    info = {"kp_ref": len(kp_r), "kp_frame": len(kp_f), "matches": 0}
    empty = np.empty((0, 2), np.float32)

    # Need >= 2 descriptors in the frame to have a "second nearest" neighbour
    if des_r is None or des_f is None or len(kp_f) < 2:
        return empty, empty, info, []

    # k=2: for each reference descriptor, its two nearest frame descriptors
    knn = cv2.BFMatcher(norm).knnMatch(des_r, des_f, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:  # Lowe ratio test
            good.append(best)

    pts_ref = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts_frame = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 2)
    info["matches"] = len(good)
    return pts_ref, pts_frame, info, good


if __name__ == "__main__":
    # Quick demo: python src/match.py data/reference.jpg data/frame.jpg
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("frame")
    args = ap.parse_args()

    ref, frame = cv2.imread(args.ref), cv2.imread(args.frame)
    assert ref is not None and frame is not None, "could not read an image"

    for method in ("orb", "sift"):
        p_r, p_f, info, good = match_features(ref, frame, method)
        print(method, info)
        # Draw keypoints too, to see what the detector actually found
        kp_r = [cv2.KeyPoint(x, y, 1) for x, y in p_r]
        kp_f = [cv2.KeyPoint(x, y, 1) for x, y in p_f]
        dm = [cv2.DMatch(i, i, 0) for i in range(len(kp_r))]
        vis = cv2.drawMatches(ref, kp_r, frame, kp_f, dm[:100], None)
        cv2.imwrite(f"results/matches_{method}.jpg", vis)
