#!/usr/bin/env python3
"""
compare_sfm_poses.py

Compares incremental and global SfM pose estimates against ground-truth
poses stored in camera_poses.pkl.

Pose convention: all poses are cam-to-world 4x4 transforms; T[:3,3] is the
camera centre in world coordinates. recoverPose returns world-to-cam, so
poses are inverted before chaining or comparison.

Scale and alignment: cam_a is used as the anchor (estimated pose replaced by
GT exactly). The single scale factor is set so that the cam_a -> cam_b
baseline length matches GT.
"""

import os, glob, pickle, math, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr as sp_lsqr

# user config
CAM_A          = 0
CAM_B          = 10
GLOBAL_K       = 3
IMAGE_FILES    = '/home/aayushi/Documents/Lepton/turntable/proc'
POSES_PKL      = 'camera_poses.pkl'
CAM_PARAMS_NPZ = 'camera_params.npz'


# I/O

def load_gt_poses(path):
    """Return a list of 4x4 cam-to-world arrays."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    def _invert(T):
        R = T[:3, :3]; t = T[:3, 3]
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3]  = -R.T @ t
        return out

    if isinstance(data, (list, tuple)):
        return [np.asarray(p, dtype=float) for p in data]
    if isinstance(data, np.ndarray):
        return [data[i].astype(float) for i in range(data.shape[0])]
    if isinstance(data, dict):
        for key in ('T_cam_to_world', 'poses_cam_to_world'):
            if key in data:
                arr = np.asarray(data[key])
                return [arr[i].astype(float) for i in range(arr.shape[0])]
        for key in ('T_world_to_cam', 'poses_world_to_cam'):
            if key in data:
                arr = np.asarray(data[key])
                return [_invert(arr[i].astype(float)) for i in range(arr.shape[0])]
        if 'R_world_to_cam' in data and 't_world_to_cam' in data:
            R_arr = np.asarray(data['R_world_to_cam']).astype(float)
            t_arr = np.asarray(data['t_world_to_cam']).astype(float)
            poses = []
            for i in range(R_arr.shape[0]):
                T = np.eye(4)
                T[:3, :3] = R_arr[i]; T[:3, 3] = t_arr[i]
                poses.append(_invert(T))
            return poses
        raise RuntimeError(f'Unknown pose dict format in {path}')
    raise RuntimeError(f'Cannot parse poses from {path}')


def find_images(root='.', exts=('png', 'jpg', 'jpeg', 'ppm', 'tif', 'tiff')):
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, f'**/*.{e}'), recursive=True))
    return sorted(files)


def load_camera_matrix(npz_path):
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    for key in ('K', 'cam_K', 'camera_matrix', 'intrinsics'):
        if key in data:
            return data[key].astype(float)
    if {'fx', 'fy', 'cx', 'cy'}.issubset(data.keys()):
        return np.array([[float(data['fx']), 0, float(data['cx'])],
                         [0, float(data['fy']), float(data['cy'])],
                         [0, 0, 1.0]])
    return None


# feature matching

def clahe_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def detect_and_match(img1, img2, ratio=0.75):
    """KAZE detection + BFMatcher with Lowe ratio test.
    Returns (pts1, pts2, n_lowe_matches) or (None, None, 0)."""
    g1, g2 = clahe_gray(img1), clahe_gray(img2)
    det = cv2.KAZE_create()
    kp1, d1 = det.detectAndCompute(g1, None)
    kp2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None, 0
    raw = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if len((m, n)) == 2 and m.distance < ratio * n.distance]
    if len(good) < 8:
        return None, None, 0
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, len(good)


def estimate_relative_pose(pts1, pts2, K=None):
    """Estimate Essential matrix via RANSAC and recover (R, t) with cheirality check.
    recoverPose gives the world-to-cam relative transform T_w2c such that
    X_B = R @ X_A + t. Returns T_w2c (4x4) and RANSAC inlier count."""
    if K is None:
        K = np.array([[500., 0., 0.], [0., 500., 0.], [0., 0., 1.]])
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, 0
    n_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K, mask=mask)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t.ravel()
    return T, int(n_inliers)


def invert_T(T):
    """Invert a cam-to-world or world-to-cam 4x4 rigid transform."""
    R = T[:3, :3]; t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3]  = -R.T @ t
    return out


# shared edge building

def build_edges(imgs, k, K=None):
    """Estimate relative pose for all pairs (i, i+offset), offset in [1..k].
    Stores T_c2w_rel: T_c2w_j = T_c2w_i @ T_c2w_rel, i.e. R_j = R_i @ R_rel.
    Also stores the reverse edge (j, i) = invert(T_c2w_rel).
    Returns edges dict and per-pair match stats."""
    n = len(imgs)
    edges, stats = {}, {}
    for i in range(n):
        for offset in range(1, k + 1):
            j = i + offset
            if j >= n:
                break
            pts1, pts2, n_lowe = detect_and_match(imgs[i], imgs[j])
            if pts1 is None:
                stats[(i, j)] = {'lowe_matches': 0, 'ransac_inliers': 0, 'inlier_ratio': 0.0}
                continue
            T_w2c_rel, n_inliers = estimate_relative_pose(pts1, pts2, K=K)
            stats[(i, j)] = {'lowe_matches': n_lowe, 'ransac_inliers': n_inliers,
                             'inlier_ratio': n_inliers / n_lowe if n_lowe > 0 else 0.0}
            if T_w2c_rel is None:
                continue
            T_c2w_rel = invert_T(T_w2c_rel)
            edges[(i, j)] = T_c2w_rel
            edges[(j, i)] = invert_T(T_c2w_rel)
    return edges, stats


# incremental SfM

def chain_poses(n, edges, anchor=0):
    """BFS chaining over k=1 consecutive edges.
    poses[anchor] = I; each subsequent pose is accumulated as
    T_c2w[b] = T_c2w[a] @ T_c2w_rel, mirroring dino_test2 initialise/register logic."""
    poses = [None] * n
    poses[anchor] = np.eye(4)
    queue, visited = [anchor], {anchor}
    while queue:
        u = queue.pop(0)
        for (a, b), T_rel in edges.items():
            if a != u or b in visited:
                continue
            poses[b] = poses[u] @ T_rel
            visited.add(b)
            queue.append(b)
    return poses


# SO(3) utilities for rotation averaging

def so3_log(R):
    """Rotation matrix -> angle-axis vector."""
    angle = math.acos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    return (angle / (2.0 * math.sin(angle))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def so3_exp(v):
    """Angle-axis vector -> rotation matrix."""
    R, _ = cv2.Rodrigues(np.asarray(v, dtype=np.float64))
    return R


def average_rotations(R_list):
    """Geodesic mean on SO(3) via iterative first-order retraction."""
    if len(R_list) == 1:
        return R_list[0].copy()
    R_mean = R_list[0].copy()
    for _ in range(50):
        delta = np.mean([so3_log(R @ R_mean.T) for R in R_list], axis=0)
        if np.linalg.norm(delta) < 1e-9:
            break
        R_mean = so3_exp(delta) @ R_mean
    return R_mean


# global SfM: rotation averaging

def global_rotation_averaging(n, edges_raw, anchor=0):
    """Recover absolute rotations by minimising (uniform weights, ±k neighbourhood):

        min_{R_i}  sum_{(i,j) in E_k}  d( R_ij,  R_i R_j^T )^2

    where d(·,·) is the geodesic distance on SO(3).

    Edge convention: edges_raw[(i,j)][:3,:3] = R_rel with R_j = R_i @ R_rel.

    Per-camera subproblem (coordinate descent, all others fixed):
      Forward edge (a -> cam): R_rel = R_a^T R_cam  =>  R_cam predicted as R_abs[a] @ R_rel
      Reverse edge (cam -> b): R_rel = R_cam^T R_b  =>  R_cam predicted as R_abs[b] @ R_rel^T
    Geodesic mean of all predictions minimises the per-camera residual sum.

    Initialised by BFS from anchor, then iterated until convergence."""
    R_rel = {(i, j): T[:3, :3].copy() for (i, j), T in edges_raw.items()}

    # BFS initialisation from anchor
    R_abs = [None] * n
    R_abs[anchor] = np.eye(3)
    queue, visited = [anchor], {anchor}
    while queue:
        u = queue.pop(0)
        for (a, b) in list(R_rel.keys()):
            if a != u or b in visited:
                continue
            R_abs[b] = R_abs[u] @ R_rel[(a, b)]
            visited.add(b)
            queue.append(b)

    # coordinate descent over non-anchor cameras
    for _iter in range(30):
        changed = False
        for cam in range(n):
            if cam == anchor:
                continue
            predictions = []
            for (i, j), Rr in R_rel.items():
                # forward edge (i -> cam): R_cam = R_abs[i] @ Rr
                if j == cam and R_abs[i] is not None:
                    predictions.append(R_abs[i] @ Rr)
                # reverse edge (cam -> j): R_cam = R_abs[j] @ Rr^T
                elif i == cam and R_abs[j] is not None:
                    predictions.append(R_abs[j] @ Rr.T)
            if not predictions:
                continue
            R_new = average_rotations(predictions)
            if R_abs[cam] is not None:
                if np.linalg.norm(so3_log(R_new @ R_abs[cam].T)) > 1e-6:
                    changed = True
            else:
                changed = True
            R_abs[cam] = R_new
        if not changed:
            break

    return R_abs


# global SfM: translation synchronisation

def global_translation_sync(n, edges_raw, R_abs, anchor=0):
    """Recover camera centres {c_i} given fixed rotations {R_i} by solving:

        min_{c_i, s}  sum_{(i,j) in E_k}  || R_i^T (c_j - c_i) - s * t_ij ||^2

    where t_ij = T_c2w_rel[:3,3] is the relative translation in camera-i frame and
    s > 0 is a SINGLE global scale shared by all edges (recoverPose gives unit-norm t,
    so all edges share the same unknown metric scale).
    Gauge: c_anchor = 0, s = 1 (scale is absorbed into c_i magnitudes; anchor_and_scale
    will recover the true metric scale from GT afterwards).
    Solved as a sparse linear least-squares system via scipy lsqr.

    Using one shared s instead of per-edge s_ij is the key change: per-edge scales
    are under-constrained when edges are few or noisy, causing wildly different
    reconstruction units per edge. A single s forces all edges to agree on one unit,
    so anchor_and_scale can correct it with a single scalar."""

    # collect undirected edges (i < j only, to avoid duplicate constraints)
    edge_list = []
    for (i, j), T in edges_raw.items():
        if i >= j or R_abs[i] is None or R_abs[j] is None:
            continue
        t_rel = T[:3, 3].copy()
        if np.linalg.norm(t_rel) < 1e-12:
            continue
        edge_list.append((i, j, t_rel))

    M = len(edge_list)
    if M == 0:
        return [np.zeros(3)] * n

    # unknowns: [c_0 .. c_{n-1}, s]  (3n + 1 total)
    # one shared scale variable at index 3*n
    dim = 3 * n + 1
    rows, cols_idx, vals = [], [], []
    b_vec = np.zeros(3 * M)

    for e_idx, (i, j, t_rel) in enumerate(edge_list):
        Ri = R_abs[i]
        # constraint: -R_i^T c_i + R_i^T c_j - s * t_ij = 0  (3 rows)
        # R_i^T[d,k] = Ri[k,d]
        for d in range(3):
            row = 3 * e_idx + d
            if i != anchor:
                for kk in range(3):
                    rows.append(row); cols_idx.append(3 * i + kk); vals.append(-Ri[kk, d])
            if j != anchor:
                for kk in range(3):
                    rows.append(row); cols_idx.append(3 * j + kk); vals.append(Ri[kk, d])
            # single global scale at column 3*n
            rows.append(row); cols_idx.append(3 * n); vals.append(-t_rel[d])

    # gauge: c_anchor = 0
    for d in range(3):
        row = 3 * M + d
        rows.append(row); cols_idx.append(3 * anchor + d); vals.append(1.0)
    b_vec = np.append(b_vec, [0.0, 0.0, 0.0])

    # gauge: s = 1 (unit scale; anchor_and_scale corrects to metric afterwards)
    rows.append(3 * M + 3); cols_idx.append(3 * n); vals.append(1.0)
    b_vec = np.append(b_vec, [1.0])

    A = csr_matrix((vals, (rows, cols_idx)), shape=(3 * M + 4, dim))
    x = sp_lsqr(A, b_vec, iter_lim=10000, atol=1e-9, btol=1e-9)[0]

    s_solved = x[3 * n]
    print(f'  [trans_sync] solved global scale s = {s_solved:.6f}')

    return [x[3 * i: 3 * i + 3].copy() if R_abs[i] is not None else np.zeros(3)
            for i in range(n)]


def build_global_poses(n, edges_raw, anchor=0):
    """Full global SfM: rotation averaging -> translation sync -> assemble T_c2w."""
    R_abs  = global_rotation_averaging(n, edges_raw, anchor=anchor)
    centres = global_translation_sync(n, edges_raw, R_abs, anchor=anchor)
    poses = []
    for i in range(n):
        if R_abs[i] is None:
            poses.append(None)
            continue
        T = np.eye(4)
        T[:3, :3] = R_abs[i]
        T[:3,  3] = centres[i]
        poses.append(T)
    return poses


# scale and alignment

def anchor_and_scale(poses_c2w_est, gt_poses, cam_a, cam_b):
    """Align estimated poses to GT at cam_a, then scale so the cam_a->cam_b
    baseline length matches GT. Returns aligned poses and the scale factor."""
    T_est_a = poses_c2w_est[cam_a]
    if T_est_a is None:
        raise RuntimeError(f'Estimated pose for anchor cam {cam_a} is None')

    # rigid alignment: T_align @ T_est_a = T_gt_a
    T_align = gt_poses[cam_a] @ invert_T(T_est_a)
    aligned = [T_align @ p if p is not None else None for p in poses_c2w_est]

    if aligned[cam_b] is None:
        print(f'Warning: no estimated pose for cam_b={cam_b}; scale=1')
        return aligned, 1.0

    baseline_gt  = np.linalg.norm(gt_poses[cam_b][:3, 3] - gt_poses[cam_a][:3, 3])
    baseline_est = np.linalg.norm(aligned[cam_b][:3, 3]  - aligned[cam_a][:3, 3])
    if baseline_est < 1e-9:
        print('Warning: estimated baseline ~0; scale=1')
        return aligned, 1.0

    scale = baseline_gt / baseline_est
    t_anchor = gt_poses[cam_a][:3, 3]
    scaled = []
    for p in aligned:
        if p is None:
            scaled.append(None)
        else:
            T = p.copy()
            T[:3, 3] = t_anchor + scale * (p[:3, 3] - t_anchor)
            scaled.append(T)
    return scaled, scale


# plotting utilities

def draw_frustum(ax, T_c2w, size=0.05, color='k', label=None):
    """Draw a camera frustum. T_c2w[:3,3] = camera centre, columns of T_c2w[:3,:3]
    are right/up/forward axes. Frustum opens along +Z (OpenCV convention)."""
    origin  = T_c2w[:3, 3]
    R       = T_c2w[:3, :3]
    right, up, forward = R[:, 0], R[:, 1], R[:, 2]
    center  = origin + forward * size * 1.5
    hw      = size * 0.6
    corners = [center + right * hw + up * hw,
               center - right * hw + up * hw,
               center - right * hw - up * hw,
               center + right * hw - up * hw]
    for c in corners:
        ax.plot(*zip(origin, c), color=color, linewidth=0.8)
    loop = corners + [corners[0]]
    ax.plot([p[0] for p in loop], [p[1] for p in loop], [p[2] for p in loop],
            color=color, linewidth=0.8)
    if label:
        ax.text(*origin, label, color=color, fontsize=8)


def set_equal_axes(ax, centers):
    """Equal-aspect 3D axis limits centred on the point cloud."""
    c = np.array(centers)
    c = c[~np.isnan(c).any(axis=1)]
    if c.shape[0] == 0:
        return
    mid = c.mean(axis=0)
    rng = max(0.05, np.abs(c - mid).max() * 1.4)
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


def rotation_error_deg(R_gt, R_est):
    dR  = R_est @ R_gt.T
    val = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(val))


def translation_error(T_gt, T_est):
    return float(np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3]))


def reprojection_error_for_pair(pts1, pts2, T_c2w_i, T_c2w_j, K):
    """Symmetric reprojection error: triangulate with GT poses, reproject into both views.
    Returns (mean, median) error in pixels and the per-point array."""
    T_w2c_i = invert_T(T_c2w_i)
    T_w2c_j = invert_T(T_c2w_j)
    R1, t1  = T_w2c_i[:3, :3], T_w2c_i[:3, 3]
    R2, t2  = T_w2c_j[:3, :3], T_w2c_j[:3, 3]
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64),
                                           pts2.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3]).T

    # cheirality filter
    d1 = (R1 @ pts3d.T + t1.reshape(3, 1))[2]
    d2 = (R2 @ pts3d.T + t2.reshape(3, 1))[2]
    valid = (d1 > 0) & (d2 > 0)
    if valid.sum() < 4:
        return np.nan, np.nan, np.array([])

    X = pts3d[valid]

    def project(P, X):
        Xh = np.hstack([X, np.ones((X.shape[0], 1))]).T
        px = P @ Xh
        return (px[:2] / px[2]).T

    err1 = np.linalg.norm(project(P1, X) - pts1[valid], axis=1)
    err2 = np.linalg.norm(project(P2, X) - pts2[valid], axis=1)
    sym_err = (err1 + err2) / 2.0
    return float(np.mean(sym_err)), float(np.median(sym_err)), sym_err


# main

def compare_and_plot(
    cam_a=CAM_A, cam_b=CAM_B, global_k=GLOBAL_K,
    image_files=IMAGE_FILES,
    poses_pkl=POSES_PKL,
    cam_params_npz=CAM_PARAMS_NPZ,
):
    if isinstance(image_files, str):
        if os.path.isdir(image_files):
            image_files = find_images(image_files)
        elif os.path.isfile(image_files):
            image_files = [image_files]
        else:
            raise RuntimeError(f'IMAGE_FILES path not found: {image_files}')
    if not image_files:
        raise RuntimeError('No images found')

    imgs, valid_paths = [], []
    for p in image_files:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            print(f'Warning: cannot read {p}, skipping')
            continue
        imgs.append(im); valid_paths.append(p)

    n = len(imgs)
    if n == 0:
        raise RuntimeError('No readable images')
    print(f'Loaded {n} images')

    gt_poses = load_gt_poses(poses_pkl)
    L = min(n, len(gt_poses))
    gt_poses = gt_poses[:L]
    imgs     = imgs[:L]
    print(f'Using {L} pose/image pairs')

    if cam_a >= L or cam_b >= L:
        raise RuntimeError(f'cam_a={cam_a} or cam_b={cam_b} out of range (L={L})')

    K = load_camera_matrix(cam_params_npz)
    if K is not None:
        print(f'Loaded K from {cam_params_npz}:\n{K}')
    else:
        print('No camera matrix found; using K=diag(500,500,1)')

    # incremental SfM: consecutive (k=1) edges, BFS chaining
    print('Building incremental edges (k=1)…')
    edges_inc, stats_inc = build_edges(imgs, k=1, K=K)
    poses_inc_raw = chain_poses(L, edges_inc, anchor=cam_a)

    # global SfM: rotation averaging + translation sync over ±k edges
    print(f'Building global edges (k={global_k})…')
    edges_glob, stats_glob = build_edges(imgs, k=global_k, K=K)
    print('Running rotation averaging and translation synchronisation…')
    poses_glob_raw = build_global_poses(L, edges_glob, anchor=cam_a)

    # anchor both methods to GT at cam_a and match cam_a->cam_b baseline scale
    poses_inc,  s_inc  = anchor_and_scale(poses_inc_raw,  gt_poses, cam_a, cam_b)
    poses_glob, s_glob = anchor_and_scale(poses_glob_raw, gt_poses, cam_a, cam_b)

    print(f'\nScale factors — Incremental: {s_inc:.4f}   Global: {s_glob:.4f}')

    def fmt_errors(poses_method, name):
        p = poses_method[cam_b]
        if p is None:
            print(f'{name}: no pose for cam_b={cam_b}')
            return
        t_err = translation_error(gt_poses[cam_b], p)
        r_err = rotation_error_deg(gt_poses[cam_b][:3, :3], p[:3, :3])
        print(f'{name:15s}  trans={t_err * 100:.2f} cm   rot={r_err:.2f}°')

    print(f'\nErrors at cam_b={cam_b} (after anchor+scale):')
    fmt_errors(poses_inc,  'Incremental')
    fmt_errors(poses_glob, 'Global')

    out_dir = 'output_sfm_compare_final'
    os.makedirs(out_dir, exist_ok=True)

    # per-pair CSV
    print('\nComputing per-pair reprojection errors and writing CSV…')
    all_pairs  = sorted(set(list(stats_inc.keys()) + list(stats_glob.keys())))
    K_reproj   = K if K is not None else np.array([[500., 0., 0.], [0., 500., 0.], [0., 0., 1.]])
    csv_rows   = []

    for (i, j) in all_pairs:
        si = stats_inc.get((i, j), {})
        sg = stats_glob.get((i, j), {})
        s  = sg if sg else si
        lowe    = s.get('lowe_matches',   0)
        inliers = s.get('ransac_inliers', 0)
        inl_rat = s.get('inlier_ratio',   0.0)

        reproj_mean = reproj_med = float('nan')
        if i < len(gt_poses) and j < len(gt_poses):
            pts1, pts2, _ = detect_and_match(imgs[i], imgs[j])
            if pts1 is not None and pts1.shape[0] >= 8:
                reproj_mean, reproj_med, _ = reprojection_error_for_pair(
                    pts1, pts2, gt_poses[i], gt_poses[j], K_reproj)

        def _pose_errs(poses_method, cam_idx):
            p = poses_method[cam_idx] if cam_idx < len(poses_method) else None
            if p is None:
                return float('nan'), float('nan')
            return (translation_error(gt_poses[cam_idx], p) * 100.0,
                    rotation_error_deg(gt_poses[cam_idx][:3, :3], p[:3, :3]))

        t_inc_i,  r_inc_i  = _pose_errs(poses_inc,  i)
        t_inc_j,  r_inc_j  = _pose_errs(poses_inc,  j)
        t_glo_i,  r_glo_i  = _pose_errs(poses_glob, i)
        t_glo_j,  r_glo_j  = _pose_errs(poses_glob, j)

        def _fmt(v): return f'{v:.3f}' if not math.isnan(v) else 'nan'
        csv_rows.append({
            'cam_i': i, 'cam_j': j,
            'lowe_matches':        lowe,
            'ransac_inliers':      inliers,
            'inlier_ratio':        f'{inl_rat:.4f}',
            'reproj_err_mean_px':  _fmt(reproj_mean),
            'reproj_err_med_px':   _fmt(reproj_med),
            'inc_trans_err_cm_i':  _fmt(t_inc_i),
            'inc_rot_err_deg_i':   _fmt(r_inc_i),
            'inc_trans_err_cm_j':  _fmt(t_inc_j),
            'inc_rot_err_deg_j':   _fmt(r_inc_j),
            'glob_trans_err_cm_i': _fmt(t_glo_i),
            'glob_rot_err_deg_i':  _fmt(r_glo_i),
            'glob_trans_err_cm_j': _fmt(t_glo_j),
            'glob_rot_err_deg_j':  _fmt(r_glo_j),
        })

    csv_path = os.path.join(out_dir, 'pairwise_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader(); writer.writerows(csv_rows)
    print(f'CSV saved -> {csv_path}  ({len(csv_rows)} pairs)')

    print(f'\n{"Pair":>7}  {"Lowe":>6}  {"RANSAC":>7}  {"Inl%":>5}  '
          f'{"Reproj(mean)":>13}  {"Reproj(med)":>12}')
    for r in csv_rows:
        inl_pct = f'{float(r["inlier_ratio"]) * 100:.1f}'
        print(f'({r["cam_i"]},{r["cam_j"]}){"":<2}  {r["lowe_matches"]:>6}  '
              f'{r["ransac_inliers"]:>7}  {inl_pct:>5}  '
              f'{r["reproj_err_mean_px"]:>13}  {r["reproj_err_med_px"]:>12}')

    gt_a   = gt_poses[cam_a]
    gt_b   = gt_poses[cam_b]
    inc_b  = poses_inc[cam_b]
    glob_b = poses_glob[cam_b]

    def two_cam_range():
        pts = np.vstack([gt_a[:3, 3], gt_b[:3, 3]])
        mid = pts.mean(axis=0)
        rng = max(0.08, np.linalg.norm(pts[0] - pts[1]) * 1.2)
        return mid, rng

    frustum_sz = np.linalg.norm(gt_a[:3, 3] - gt_b[:3, 3]) * 0.15
    frustum_sz = max(0.02, min(frustum_sz, 0.15))
    mid, rng   = two_cam_range()

    def _set_lims(ax):
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # figure 1: incremental
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Incremental SfM  |  GT A={cam_a}, GT B={cam_b}  (scale {s_inc:.4f})')
    draw_frustum(ax, gt_a,  size=frustum_sz, color='black',   label=f'GT {cam_a}')
    draw_frustum(ax, gt_b,  size=frustum_sz, color='black',   label=f'GT {cam_b}')
    if inc_b is not None:
        draw_frustum(ax, inc_b, size=frustum_sz, color='magenta', label=f'Inc {cam_b}')
    _set_lims(ax)
    f1 = os.path.join(out_dir, f'incremental_A{cam_a}_B{cam_b}.png')
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close(fig)
    print(f'Saved {f1}')

    # figure 2: global
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Global SfM (k={global_k})  |  GT A={cam_a}, GT B={cam_b}  (scale {s_glob:.4f})')
    draw_frustum(ax, gt_a,  size=frustum_sz, color='black',      label=f'GT {cam_a}')
    draw_frustum(ax, gt_b,  size=frustum_sz, color='black',      label=f'GT {cam_b}')
    if glob_b is not None:
        draw_frustum(ax, glob_b, size=frustum_sz, color='darkorange', label=f'Glob {cam_b}')
    _set_lims(ax)
    f2 = os.path.join(out_dir, f'global_A{cam_a}_B{cam_b}.png')
    plt.tight_layout(); plt.savefig(f2, dpi=150); plt.close(fig)
    print(f'Saved {f2}')

    # figure 3: side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
    for ax, method, poses_m, s_m, color, tag in [
        (axes[0], 'Incremental',         poses_inc,  s_inc,  'magenta',    'Inc'),
        (axes[1], f'Global k={global_k}', poses_glob, s_glob, 'darkorange', 'Glob'),
    ]:
        ax.set_title(f'{method}  (scale={s_m:.4f})')
        draw_frustum(ax, gt_a, size=frustum_sz, color='black', label=f'GT {cam_a}')
        draw_frustum(ax, gt_b, size=frustum_sz, color='black', label=f'GT {cam_b}')
        p = poses_m[cam_b]
        if p is not None:
            draw_frustum(ax, p, size=frustum_sz, color=color, label=f'{tag} {cam_b}')
        _set_lims(ax)
    fig.suptitle(f'SfM comparison — anchor cam {cam_a}, evaluating cam {cam_b}', fontsize=11)
    f3 = os.path.join(out_dir, f'side_by_side_A{cam_a}_B{cam_b}.png')
    plt.tight_layout(); plt.savefig(f3, dpi=150); plt.close(fig)
    print(f'Saved {f3}')

    # figure 4: full trajectories
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title('Full trajectories: GT (black) | Incremental (magenta) | Global (orange)')

    def centers(poses):
        return np.array([p[:3, 3] if p is not None else [np.nan] * 3 for p in poses])

    for c, col, lbl in [(centers(gt_poses),    'black',      'GT'),
                        (centers(poses_inc),   'magenta',    'Incremental'),
                        (centers(poses_glob),  'darkorange', 'Global')]:
        ok = ~np.isnan(c).any(axis=1)
        if ok.any():
            ax.plot(c[ok, 0], c[ok, 1], c[ok, 2], '-', color=col, label=lbl)
            ax.scatter(c[ok, 0], c[ok, 1], c[ok, 2], color=col, s=8)

    set_equal_axes(ax, centers(gt_poses))
    ax.legend(); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    f4 = os.path.join(out_dir, f'trajectories_A{cam_a}_B{cam_b}.png')
    plt.tight_layout(); plt.savefig(f4, dpi=150); plt.close(fig)
    print(f'Saved {f4}')

    print('\nDone.')


if __name__ == '__main__':
    compare_and_plot()