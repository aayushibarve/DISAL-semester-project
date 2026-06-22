import cv2
import numpy as np
import glob
import os

IMAGE_FOLDER   = "./datasets/chessboard" #Change depending on device
IMAGE_GLOB     = "*.tiff"
SQUARE_SIZE_MM = 20
BOARD_COLS     = 4
BOARD_ROWS     = 4
CLAHE_CLIP     = 4.0 #Increase limit for better contrast
CLAHE_TILE     = (8, 8) #Default
SAVE_FILE      = "camera_params_chessboard.npz"

PATTERN_SIZE  = (BOARD_COLS, BOARD_ROWS)
SB_FLAGS      = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
clahe_op      = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

def process(path):
    raw  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw.copy()
    gray8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    enhanced = clahe_op.apply(gray8)
    return enhanced

def detect(enhanced):
    #Check detection of chessboard corners for 3 different versions - CLAHE, CLAHE+Gaussian blur, CLAHE+edhe sharpening
    candidates = [
        ("enhanced",  enhanced),
        ("blurred",   cv2.GaussianBlur(enhanced, (5, 5), 0)),
        ("sharpened", cv2.filter2D(enhanced, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))),
    ]
    for name, candidate in candidates:
        found, corners = cv2.findChessboardCornersSB(candidate, PATTERN_SIZE, SB_FLAGS)
        if found:
            return True, corners, name
    return False, None, None

image_paths = sorted(glob.glob(IMAGE_FOLDER + "/*.tiff"))

obj_points  = []
img_points  = []
image_size  = None

for path in image_paths:
    enhanced = process(path)
    found, corners, candidate_name = detect(enhanced)
    if found:
        obj_points.append(objp)
        img_points.append(corners)
        if image_size is None:
            image_size = (enhanced.shape[1], enhanced.shape[0])
        #Uncomment the below lines to see which processed version worked/ which images the chessboard wasn't detected for
        #print(f"  {os.path.basename(path):<40}  {candidate_name}")
    # else:
    #     print(f"  {os.path.basename(path):<40}  not detected")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

np.savez(SAVE_FILE, ret=ret, K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)

print(f"\nImages used: {len(obj_points)}/{len(image_paths)}")
print(f"RMS error: {ret:.4f} px")
print(f"Camera matrix K:\n{np.array2string(K, precision=4, suppress_small=True)}")
print(f"Distortion coeffs: {np.array2string(dist, precision=6, suppress_small=True)}")
print(f"Saved to: {os.path.abspath(SAVE_FILE)}")
