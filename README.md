# Thermal Multi-view Reconstruction

A semester project at EPFL's Distributed Intelligent Systems and Algorithms Lab (DISAL) investigating **3D reconstruction from thermal (infrared) imagery**. The project evaluates the full pipeline from camera calibration to dense reconstruction, with particular focus on the challenges that arise when applying standard RGB computer vision techniques to thermal data.

---

## Repository Structure

```
thermal-multiview/
│
├── approximate_camera_poses/
│   ├── camera_poses.pkl          # Pre-computed approximate ground-truth poses
│   └── turntable.py              # Turntable trajectory estimation
│
├── automated_calibration/
│   ├── calib-auto.py             # Mask-based calibration (blob detection on thermal grid)
│   ├── calib-chessboard.py       # Chessboard-based calibration with CLAHE enhancement
│   ├── calib-trial-march.ipynb   # Calibration trials and analysis
│   ├── camera_params.npz         # Saved intrinsics from mask calibration
│   └── camera_params_chessboard.npz  # Saved intrinsics from chessboard calibration
│
├── datasets/
│   ├── capture_march_20/         # Raw thermal captures
│   ├── capture_march_20_proc/    # Preprocessed captures
│   ├── capture_table/            # Table scene captures
│   ├── chessboard/               # Chessboard calibration images
│   ├── kettle/                   # Kettle turntable dataset (ground-truth evaluation)
│   ├── laptop/                   # Laptop scene captures
│   ├── person2/                  # Person scene captures
│   ├── test/                     # Miscellaneous test images
│   └── turntable/                # Turntable rotation dataset
│
├── feature_evaluation/
│   ├── evaluate_features.py      # Benchmarks detector+preprocessor combinations
│   ├── superpoint_trial.ipynb    # SuperPoint + LightGlue evaluation on thermal images
│   └── turntable_detector_results/  # Saved evaluation outputs
│
├── manual_calibration/
│   ├── annotate.py               # Manual point annotation tool
│   ├── calib-manual.py           # Manual calibration from annotated correspondences
│   └── camera.py                 # Camera capture script (Lepton thermal sensor, Y16 format)
│
├── reconstructions/
│   ├── dino2_cloud.ply           # Sparse point cloud (SfM)
│   ├── dino2_cloud_viz.html      # Interactive 3D viewer — sparse cloud
│   ├── dsi_pointcloud.ply        # Dense point cloud (plane-sweep stereo)
│   ├── dsi_pointcloud_viz.html   # Interactive 3D viewer — dense cloud
│   ├── voxel_cloud.ply           # Voxel carving reconstruction
│   └── voxel_cloud_viz.html      # Interactive 3D viewer — voxel cloud
│
├── temperature_monitoring/
│   └── temperature_monitoring.ino  # Arduino sketch for thermal reference monitoring
│
├── dense_recon.ipynb             # Dense reconstruction (plane-sweep stereo + voxel carving)
├── global_vs_inc.py             # Comparing global vs incremental SfM to generate csv with pairwise rotation and translation estimation errors compared to ground truth
├── sparse_recon.py               # Incremental SfM pipeline (SIFT + Essential Matrix + PnP)
└── visualize_ply.py              # Point cloud visualization utility
```

---

## Pipeline Details

### 1. Camera Calibration

Two calibration approaches designed for thermal sensors are implemented and compared:

- **Mask-based** (`automated_calibration/calib-auto.py`): A perforated mask placed in front of a heated background produces a regular grid of bright thermal markers. Blob centroids are detected via thresholding and connected components.
- **Chessboard-based** (`automated_calibration/calib-chessboard.py`): A thermal chessboard made from heated and insulated regions, detected using OpenCV's `findChessboardCornersSB` with CLAHE contrast enhancement.

Both methods output standard camera intrinsics (`K`) and distortion coefficients (`dist`) via `cv2.calibrateCamera`.

### 2. Feature Detection & Matching Evaluation

`feature_evaluation/evaluate_features.py` benchmarks all combinations of:

| Preprocessors | Detectors |
|---|---|
| Linear normalisation | SIFT |
| Log normalisation | ORB |
| Histogram equalisation | AKAZE |
| CLAHE | KAZE |
| Gaussian sharpening | BRISK |
| Gamma correction (γ=0.5) | FAST+BRIEF |
| | FAST+FREAK |

Pairs are evaluated on precision and recall using geometric consistency (RANSAC Essential Matrix). Results are saved as CSV files for analysis.


### 3. Sparse Reconstruction (SfM)

`sparse_recon.py` implements an incremental Structure-from-Motion pipeline:
- KAZE feature detection with CLAHE (this block can be replaced with other feature detectors depending on the results of the detector evaluation)
- Essential matrix estimation + `recoverPose` for the seed pair
- Incremental camera registration via PnP RANSAC
- Triangulation with cheirality check

A controlled kettle turntable dataset (full 360° rotation, fixed camera) provides approximate ground-truth poses for quantitative evaluation.


### 3. Comparison of Global vs Incremental SfM

Using the approximate ground truth, the results of incremental and global SfM (rotations and translations averaged in world frame over multiple viewpoints) are compared. 


### 4. Dense Reconstruction

`dense_recon.ipynb` implements two dense methods using known camera poses:

- **Plane-sweep stereo (DSI)**: Photometric consistency across views produces dense depth estimates from thermal intensities.
- **Voxel carving**: Reconstructs the visual hull from segmented silhouettes.

Outputs (`.ply` files + interactive HTML viewers) are saved in `reconstructions/`.

---

## Hardware

- **Thermal camera**: FLIR Lepton (160×120, radiometric Y16 output via V4L2)
- **Capture script**: `manual_calibration/camera.py` — streams Y16 frames, supports interactive capture (`c` to save, `q` to quit)

---

## Dependencies

```bash
pip install opencv-python numpy open3d matplotlib tqdm tifffile pandas
```

For deep learning features:
```bash
pip install torch torchvision
# SuperPoint + LightGlue — see superpoint_trial.ipynb for setup
```


## Acknowledgements

This project was conducted as a semester project at DISAL, EPFL under the supervision of Alexander Wallén Kiessling and Prof. Alcherio Martinoli.
