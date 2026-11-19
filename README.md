# planar-ar-homography

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-homography%20from%20scratch-013243)
![OpenCV](https://img.shields.io/badge/OpenCV-matching%20%26%20warping%20only-5C3EE8)

Planar AR overlay in Python. I estimate the homography myself (normalized DLT + RANSAC in plain NumPy) and use it to pin an image onto a flat object in a video. OpenCV's `findHomography` is used only as a baseline to compare against.

## Contents

- [Why I built this](#why-i-built-this)
- [What's implemented](#whats-implemented)
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Testing](#testing)
- [Results](#results)
- [Limitations](#limitations)

## Why I built this

I wanted to understand what happens inside `cv2.findHomography` instead of treating it as a black box. Planar AR is a good testbed because errors are visible: if the homography is wrong, the overlay slides off the object.

The rule I set myself: OpenCV may be used for feature detection/matching, video I/O and warping. The homography estimation and the RANSAC loop are my own NumPy code, and I check them against OpenCV's implementation.

## What's implemented

| Part | File | Implementation |
|------|------|----------------|
| ORB / SIFT matching + Lowe ratio test | `src/match.py` | OpenCV detectors and `BFMatcher` |
| Normalized DLT (Hartley) + SVD | `src/dlt.py` | NumPy only |
| RANSAC with optional adaptive iteration count | `src/ransac.py` | NumPy only |
| Warp overlay + write video | `src/overlay.py` | `cv2.warpPerspective` with my H |
| Comparison, robustness tests, plots | `src/evaluate.py` | NumPy / Matplotlib |
| Synthetic-data unit tests | `tests/` | NumPy / pytest |

## How it works

### 1. Matching (`src/match.py`)

Keypoints are detected in the reference image and in each frame (ORB or SIFT), descriptors are matched with a brute-force matcher, and a match is kept only if it passes Lowe's ratio test: best distance < 0.75 x second-best distance. Ambiguous matches (repetitive texture) have two similar distances and get rejected.

ORB descriptors are binary, so they use Hamming distance. SIFT descriptors are float vectors, so they use L2.

### 2. Homography via DLT (`src/dlt.py`)

A homography $H$ is a 3x3 matrix mapping a point on the reference plane to its position in the frame, in homogeneous coordinates:

$$ \lambda \begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix} $$

$H$ is only defined up to scale, so it has 8 degrees of freedom and 4 point pairs are enough. Each pair gives two linear equations in the entries $h_1 \dots h_9$ of $H$:

$$
\begin{aligned}
-x h_1 - y h_2 - h_3 + x'x\,h_7 + x'y\,h_8 + x' h_9 &= 0 \\
-x h_4 - y h_5 - h_6 + y'x\,h_7 + y'y\,h_8 + y' h_9 &= 0
\end{aligned}
$$

Stacking $N$ pairs gives a $2N \times 9$ matrix $A$ and the system $A\mathbf{h} = 0$. With noise there is no exact solution, so I minimize $\lVert A\mathbf{h} \rVert$ subject to $\lVert \mathbf{h} \rVert = 1$. The minimizer is the right singular vector of $A$ for the smallest singular value (the last row of $V^T$ from the SVD), reshaped to 3x3.

**Hartley normalization.** Pixel coordinates are in the hundreds, so entries of $A$ range from 1 to about $10^5$ and the problem is badly conditioned. Before solving, each point set is translated so its centroid is the origin and scaled so the mean distance from the origin is $\sqrt{2}$. This gives similarity transforms $T$ (reference) and $T'$ (frame). I solve for $\hat{H}$ on the normalized points, then undo the normalization:

$$ H = T'^{-1} \, \hat{H} \, T $$

### 3. RANSAC (`src/ransac.py`)

Matches always contain wrong pairs, and DLT is least squares, so even a few outliers ruin the fit. The loop:

1. Sample 4 random pairs and fit $H$ with DLT.
2. Project every reference point with $H$ and measure the reprojection error against its matched frame point:

$$ e_i = \left\lVert \mathbf{x}'_i - \pi(H\mathbf{x}_i) \right\rVert_2 $$

where $\pi$ divides by the third homogeneous coordinate.
3. Points with $e_i$ below the threshold are inliers. Keep the model with the most inliers.
4. After the loop, refit $H$ with DLT on all inliers of the best model. The 4-point fit is noisy; the refit uses every good match.

The number of iterations needed to reach success probability $p$ when a fraction $w$ of matches are inliers is

$$ N = \frac{\log(1-p)}{\log(1-w^4)} $$

In adaptive mode, $w$ is updated from the best inlier ratio found so far and $N$ is recomputed.

### 4. Overlay (`src/overlay.py`)

The overlay image is resized to the reference image's size so the same $H$ applies, warped into each frame with `cv2.warpPerspective`, blended on top, and written to an output video.

## Getting started

Clone the repo, then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the reference image, input video and overlay image in `data/`, then:

```bash
# overlay on a video
python src/overlay.py --ref data/reference.jpg --video data/video.mp4 \
    --overlay data/overlay.jpg --out results/output.mp4 --method sift

# full evaluation (writes results/results.csv, plots and failure-case frames)
python src/evaluate.py --ref data/reference.jpg --video data/video.mp4
```

## Testing

```bash
pytest tests/
```

The tests use synthetic data where the true $H$ is known: apply a known $H$ to random points, add Gaussian noise, replace a fraction of the points with random outliers, then check that DLT and RANSAC recover $H$ within a tolerance. Random seeds are fixed so runs are reproducible.

## Results

Evaluation is in progress. The comparison against `cv2.findHomography` (mean reprojection error, inlier ratio, runtime), the ORB vs SIFT comparison, the robustness tests (rotation, scale, blur, low light) and the failure-case analysis will be added here once `src/evaluate.py` has produced measured numbers.

## Limitations

- Assumes a planar, rigid object. A curved or bending page breaks the homography model.
- Each frame is processed independently. There is no tracking or temporal smoothing, so the overlay can jitter.
- Needs enough texture on the reference image. Plain or highly repetitive surfaces give few reliable matches.
- Estimates only the 2D homography, not full 3D camera pose, so no 3D objects.
