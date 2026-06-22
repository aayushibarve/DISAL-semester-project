import os, sys, glob
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# 1.  CAMERA INTRINSICS

IMAGE_W, IMAGE_H = 740, 555
F  = 2360.0
CX, CY = IMAGE_W / 2.0, IMAGE_H / 2.0

K = np.array([[F,  0, CX],
              [0,  F, CY],
              [0,  0,  1]], dtype=np.float64)

def load_calibration(npz_path="./automated_calibration/camera_params.npz"):
    cal  = np.load(npz_path)
    K    = cal['K'].astype(np.float64)
    dist = cal['dist'].astype(np.float64)
    return K, dist
  
# Uncomment if trying with thermal imgs. The above code is for the dino images
# K, dist = load_calibration()

# 2.  LOAD IMAGES
 

def load_images(folder="dino_imgs"):
    paths = sorted(glob.glob(os.path.join(folder, "*.tiff")) +
                   glob.glob(os.path.join(folder, "*.tif"))  +
                   glob.glob(os.path.join(folder, "*.ppm"))  +
                   glob.glob(os.path.join(folder, "*.PPM")))
    if not paths:
        sys.exit(f"[ERROR] No images found in '{folder}'.")

    bgr, gray = [], []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            sys.exit(f"[ERROR] Could not read '{p}'.")

        # Normalise to uint8 if 16-bit (common for thermal TIFFs)
        if im.dtype == np.uint16:
            im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Ensure 3-channel BGR for colouring the point cloud
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

        bgr.append(im)
        gray.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

    print(f"Loaded {len(paths)} images from '{folder}'")
    return paths, bgr, gray

 
# 3.  FEATURE DETECTION
 

# def detect_features(gray_imgs, n_features=6000):
#     sift = cv2.SIFT_create(nfeatures=n_features)
#     kps, descs = [], []
#     for g in tqdm(gray_imgs, desc="SIFT detection"):
#         kp, des = sift.detectAndCompute(g, None)
#         kps.append(kp)
#         descs.append(des if des is not None else np.zeros((0, 128), np.float32))
#         print(len(kp))
#     return kps, descs

def detect_features(gray_imgs, n_features=6000):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kaze = cv2.KAZE_create()
    
    kps, descs = [], []
    for g in tqdm(gray_imgs, desc="KAZE detection"):
        enhanced = clahe.apply(g)
        kp, des = kaze.detectAndCompute(enhanced, None)
        kps.append(kp)
        descs.append(des if des is not None else np.zeros((0, 64), np.float32))
        print(len(kp))
    return kps, descs
 
# 4.  MATCHING  (FLANN + Lowe ratio test)
 

def match_pair(des1, des2, ratio=0.75):
    """
    Returns (idx1, idx2) arrays of matched keypoint indices, or None.
    """
    if des1.shape[0] < 2 or des2.shape[0] < 2:
        return None
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=100)
    )
    raw = flann.knnMatch(des1.astype(np.float32),
                         des2.astype(np.float32), k=2)
    good_i, good_j = [], []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_i.append(m.queryIdx)
                good_j.append(m.trainIdx)
    if len(good_i) < 8:
        return None
    return np.array(good_i), np.array(good_j)

 
# 5.  TRIANGULATION  (type-safe, correct)
 

def triangulate(P1, P2, pts1, pts2):
    """
    DLT triangulation via cv2.triangulatePoints.

    P1, P2  : (3,4) projection matrices
    pts1    : (N,2) pixel coords in image 1
    pts2    : (N,2) pixel coords in image 2

    Returns
    -------
    pts3d  : (N,3)  world points
    valid  : (N,)   boolean — finite + in front of BOTH cameras
    """
    P1 = np.asarray(P1, np.float32)
    P2 = np.asarray(P2, np.float32)
    p1 = np.asarray(pts1, np.float32).reshape(-1, 2).T   # (2, N)
    p2 = np.asarray(pts2, np.float32).reshape(-1, 2).T   # (2, N)
    N  = p1.shape[1]

    if N == 0:
        return np.empty((0, 3)), np.zeros(0, bool)

    hom   = cv2.triangulatePoints(P1, P2, p1, p2)        # (4, N)
    w     = hom[3]
    nz    = np.abs(w) > 1e-8
    pts3d = np.full((3, N), np.nan, dtype=np.float64)
    pts3d[:, nz] = hom[:3, nz] / w[nz]
    pts3d = pts3d.T                                       # (N, 3)

    # Cheirality: depth in both cameras must be positive
    # depth = r3 · X_world + t3   where r3 = P[2, :3], t3 = P[2, 3]
    d1 = (P1[2, :3].astype(np.float64) @ pts3d.T) + float(P1[2, 3])
    d2 = (P2[2, :3].astype(np.float64) @ pts3d.T) + float(P2[2, 3])

    valid = nz & np.isfinite(pts3d).all(axis=1) & (d1 > 0) & (d2 > 0)
    return pts3d, valid

 
# 6.  POSE FROM ESSENTIAL MATRIX
 

def pose_from_essential(pts1, pts2, K):
    """
    Estimate E with RANSAC, recover (R, t) with cheirality check built into
    cv2.recoverPose.

    pts1, pts2 : (N,2) pixel coordinates
    Returns R (3,3), t (3,1), inlier_mask (N,) bool   or  (None, None, None)
    """
    pts1 = np.asarray(pts1, np.float64)
    pts2 = np.asarray(pts2, np.float64)
    E, mask_e = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC, prob=0.999, threshold=2.0
    )
    if E is None or mask_e is None:
        return None, None, None
    mask_e = mask_e.ravel().astype(bool)

    _, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, K,
                                       mask=mask_e.astype(np.uint8))
    inliers = mask_p.ravel().astype(bool)
    if inliers.sum() < 8:
        return None, None, None
    return R, t, inliers

 
# 7.  GLOBAL 3-D MAP
#     Each entry: xyz (3,) + observations {img_idx: kp_idx}
#     This lets us build exact 2D-3D correspondences for PnP.
 

class PointMap:
    def __init__(self):
        self.xyz          = []   # list of (3,) float64
        self.rgb          = []   # list of (3,) float64  [0,1]
        self.observations = []   # list of dict {img_idx: kp_idx}

    # add new triangulated points 
    def add(self, pts3d, valid, ki_arr, kj_arr, img_i, img_j,
            all_kps, bgr_imgs):
        """
        pts3d  : (N,3)
        valid  : (N,) bool
        ki_arr : (N,)  keypoint indices in image img_i
        kj_arr : (N,)  keypoint indices in image img_j
        """
        bgr = bgr_imgs[img_i]
        h, w = bgr.shape[:2]
        for loc in np.where(valid)[0]:
            xyz = pts3d[loc].copy()
            pid = len(self.xyz)
            self.xyz.append(xyz)
            self.observations.append({img_i: int(ki_arr[loc]),
                                       img_j: int(kj_arr[loc])})
            # colour from img_i keypoint
            kp = all_kps[img_i][int(ki_arr[loc])]
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= x < w and 0 <= y < h:
                b, g, r = bgr[y, x]
                self.rgb.append(np.array([r/255., g/255., b/255.]))
            else:
                self.rgb.append(np.array([0.5, 0.5, 0.5]))

    # propagate map visibility via matches
    def propagate(self, img_new, img_ref, all_kps, all_descs):
        """
        Match img_new ↔ img_ref.  Where img_ref keypoints are already in the
        map, record that img_new also sees those map points.
        Returns number of newly linked observations.
        """
        match = match_pair(all_descs[img_new], all_descs[img_ref])
        if match is None:
            return 0
        kn_idx, kr_idx = match

        # kp_idx_in_ref → list of map point ids
        kp_to_pid = {}
        for pid, obs in enumerate(self.observations):
            if img_ref in obs:
                kp_to_pid.setdefault(obs[img_ref], []).append(pid)

        added = 0
        for kni, kri in zip(kn_idx, kr_idx):
            if kri in kp_to_pid:
                for pid in kp_to_pid[kri]:
                    if img_new not in self.observations[pid]:
                        self.observations[pid][img_new] = int(kni)
                        added += 1
        return added

    # get 2D-3D correspondences for PnP 
    def correspondences(self, img_idx, all_kps):
        """
        Returns (pts3d, pts2d) for all map points seen by img_idx.
        pts3d: (M,3)  pts2d: (M,2)
        """
        pts3d, pts2d, pids = [], [], []
        for pid, (xyz, obs) in enumerate(
                zip(self.xyz, self.observations)):
            if img_idx in obs:
                kp = all_kps[img_idx][obs[img_idx]]
                pts3d.append(xyz)
                pts2d.append(kp.pt)
                pids.append(pid)
        if not pts3d:
            return np.empty((0,3)), np.empty((0,2)), []
        return (np.array(pts3d, np.float64),
                np.array(pts2d, np.float64),
                pids)

    def array(self):
        if not self.xyz:
            return np.empty((0, 6))
        return np.hstack([np.array(self.xyz),
                          np.array(self.rgb)])

 
# 8.  INCREMENTAL SfM  (correct architecture)
 

class IncrementalSfM:
    def __init__(self, K):
        self.K    = K
        self.poses = {}          # img_idx → (R 3×3, t 3×1)
        self.map   = PointMap()

    def P(self, idx):
        R, t = self.poses[idx]
        return self.K @ np.hstack([R, t])   # (3,4)

    # A: initialise from a seed pair 
    def initialise(self, i, j, all_kps, all_descs, bgr_imgs):
        match = match_pair(all_descs[i], all_descs[j])
        if match is None:
            return False
        ki_idx, kj_idx = match

        pts_i = np.array([all_kps[i][k].pt for k in ki_idx], np.float64)
        pts_j = np.array([all_kps[j][k].pt for k in kj_idx], np.float64)

        R, t, mask = pose_from_essential(pts_i, pts_j, self.K)
        if R is None or mask.sum() < 8:
            return False

        # Camera i = world origin
        self.poses[i] = (np.eye(3),        np.zeros((3, 1)))
        self.poses[j] = (R,                t)

        pts3d, valid = triangulate(
            self.P(i), self.P(j),
            pts_i[mask], pts_j[mask]
        )
        if valid.sum() < 8:
            return False

        self.map.add(pts3d, valid,
                     ki_idx[mask], kj_idx[mask],
                     i, j, all_kps, bgr_imgs)

        print(f"  Init ({i},{j}): {mask.sum()} inliers → "
              f"{valid.sum()} 3-D pts")
        return True

    # B: register one new camera via PnP 
    def register(self, cam_idx, all_kps, all_descs, bgr_imgs,
                 registered_list):
        """
        1. Propagate map observations from every registered camera.
        2. Solve PnP (RANSAC) with the resulting 2D-3D pairs.
        3. Triangulate new points with every registered camera.
        """
        # 1. Propagate visibility
        for ref in registered_list:
            self.map.propagate(cam_idx, ref, all_kps, all_descs)

        # 2. PnP
        pts3d, pts2d, pids = self.map.correspondences(cam_idx, all_kps)
        if len(pts3d) < 10:
            return False

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float32),
            pts2d.astype(np.float32),
            self.K, None,
            iterationsCount=2000,
            reprojectionError=4.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok or inliers is None or len(inliers) < 10:
            return False

        R, _ = cv2.Rodrigues(rvec)
        self.poses[cam_idx] = (R, tvec)

        # 3. Triangulate NEW points against each registered camera
        n_new = 0
        kp_in_map = {obs[cam_idx] for obs in self.map.observations
                     if cam_idx in obs}

        for ref in registered_list:
            match = match_pair(all_descs[cam_idx], all_descs[ref])
            if match is None:
                continue
            kn_idx, kr_idx = match

            kp_in_map_ref = {obs[ref] for obs in self.map.observations
                             if ref in obs}

            # Truly new pairs: neither endpoint already in the map
            sel = [(kni, kri) for kni, kri in zip(kn_idx, kr_idx)
                   if kni not in kp_in_map and kri not in kp_in_map_ref]
            if len(sel) < 8:
                continue

            kni_arr = np.array([x[0] for x in sel])
            kri_arr = np.array([x[1] for x in sel])
            pts_new = np.array([all_kps[cam_idx][k].pt for k in kni_arr], np.float64)
            pts_ref = np.array([all_kps[ref][k].pt      for k in kri_arr], np.float64)

            # Triangulate: P_ref is camera 1, P_new is camera 2
            pts3d_t, valid = triangulate(self.P(ref), self.P(cam_idx),
                                         pts_ref, pts_new)

            self.map.add(pts3d_t, valid,
                         kri_arr, kni_arr,   # note: ref is "img_i" here
                         ref, cam_idx, all_kps, bgr_imgs)

            # Update local set
            kp_in_map |= set(kni_arr[valid])
            n_new += valid.sum()

        reproj = self._reproj_error(cam_idx, all_kps)
        print(f"  Cam {cam_idx:2d}: inliers={len(inliers):4d} "
              f"new_pts={n_new:4d}  map={len(self.map.xyz):6d} "
              f"reproj={reproj:.2f}px")
        return True

    def _reproj_error(self, cam_idx, all_kps):
        pts3d, pts2d, _ = self.map.correspondences(cam_idx, all_kps)
        if len(pts3d) == 0:
            return 0.0
        R, t = self.poses[cam_idx]
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(pts3d.astype(np.float32),
                                    rvec, t, self.K, None)
        return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1).mean()

    def camera_centres(self):
        """World position C = -R^T t for each registered camera."""
        return {idx: (-R.T @ t).ravel()
                for idx, (R, t) in self.poses.items()}

 
# 9.  VISUALISATION
 

def save_ply(map_arr, path="dino_cloud.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(map_arr[:, :3])
    if map_arr.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(map_arr[:, 3:], 0, 1))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved '{path}'  ({len(pcd.points)} pts after outlier removal)")
    return pcd


def visualise(pcd, cam_centres_dict):
    centres = np.array(list(cam_centres_dict.values()))
    cam_pcd = o3d.geometry.PointCloud()
    cam_pcd.points = o3d.utility.Vector3dVector(centres)
    cam_pcd.paint_uniform_color([1, 0, 0])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries(
        [pcd, cam_pcd, coord],
        window_name="Dino 3-D Reconstruction",
        width=1280, height=720
    )


def plot_trajectory(cam_centres_dict, path="camera_trajectory.png"):
    centres = np.array([cam_centres_dict[k]
                        for k in sorted(cam_centres_dict)])
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(centres[:,0], centres[:,1], centres[:,2],
            'ro-', ms=5, lw=1, label='cameras')
    for k in sorted(cam_centres_dict):
        c = cam_centres_dict[k]
        ax.text(c[0], c[1], c[2], str(k), fontsize=6)
    ax.set_title("Camera trajectory")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved '{path}'")

 
# 10.  MAIN
 

def main():
    paths, bgr_imgs, gray_imgs = load_images("./datasets/turntable/proc")
    #paths, bgr_imgs, gray_imgs = load_images("/home/aayushi/semester-project/dino_imgs")
    N = len(gray_imgs)

    print("\n=== Feature detection ===")
    all_kps, all_descs = detect_features(gray_imgs, n_features=6000)

    sfm = IncrementalSfM(K)

    # Initialisation: pick best seed pair 
    #print("\n=== Initialisation ===")
    candidates = ([(i, i+1) for i in range(min(6, N-1))] +
                  [(i, i+2) for i in range(min(6, N-2))])
    init_done, registered = False, []
    for i, j in candidates:
        if sfm.initialise(i, j, all_kps, all_descs, bgr_imgs):
            registered = [i, j]
            init_done  = True
            break

    if not init_done:
        sys.exit("[ERROR] Could not initialise SfM from any pair.")

    # Incremental registration 
    for cam_idx in range(N):
        if cam_idx in registered:
            continue
        ok = sfm.register(cam_idx, all_kps, all_descs, bgr_imgs, registered)
        if ok:
            registered.append(cam_idx)

    print(f"\nRegistered {len(registered)}/{N} cameras")
    print(f"Total map points (before filtering): {len(sfm.map.xyz)}")

    map_arr = sfm.map.array()
    cam_cen = sfm.camera_centres()
    flip = np.diag([1., -1., -1.])
    map_arr_flipped = map_arr.copy()
    map_arr_flipped[:, :3] = map_arr[:, :3] @ flip.T
    plot_trajectory(cam_cen)
    pcd = save_ply(map_arr_flipped, 'dino2_cloud.ply')

    #print("\nLaunching Open3D viewer …")
    #visualise(pcd, cam_cen)


if __name__ == "__main__":
    main()
