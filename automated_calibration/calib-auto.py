import cv2
import numpy as np
import glob

#This code is for the cardboard mask

folder = "./datasets/capture_march_20" #Chnage based on device
THRESH = 200 #Tune this to make sure 16 (or however many) blobs are detected
rows, cols = 4, 4 #Change based on the grid you use
spacing = 200/7      # mm

kernel = np.ones((3,3), np.uint8)

objp = np.zeros((rows*cols, 3), np.float32)

objp[:, :2] = np.mgrid[
    0:cols,
    0:rows
].T.reshape(-1,2) * spacing

objpoints = []   # 3D points
imgpoints = []   # 2D points

def order_points(pts, rows=4, cols=4):

    pts = np.array(pts)

    # identify TL, TR, BL, BR using sums and differences
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    TL = pts[np.argmin(s)]
    BR = pts[np.argmax(s)]

    TR = pts[np.argmin(diff)]
    BL = pts[np.argmax(diff)]

    vx = TR - TL
    vy = BL - TL

    vx = vx / np.linalg.norm(vx)
    vy = vy / np.linalg.norm(vy)

    coords = []

    for p in pts:
        v = p - TL
        x = np.dot(v, vx)
        y = np.dot(v, vy)
        coords.append([x,y])

    coords = np.array(coords)
    order = np.lexsort((coords[:,0], coords[:,1]))
    pts_sorted = pts[order]
    ordered = []

    for r in range(rows):
        row_pts = pts_sorted[r*cols:(r+1)*cols]
        # ensure left → right ordering
        row_pts = row_pts[np.argsort(row_pts[:,0])]
        ordered.append(row_pts)
    return np.vstack(ordered)

image_paths = sorted(glob.glob(folder + "/*.tiff"))

for idx,path in enumerate(image_paths):

    img16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # normalize for thresholding
    img8 = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX)
    img8 = np.uint8(img8)

    _, binary = cv2.threshold(img8, THRESH, 255, cv2.THRESH_BINARY)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    if num_labels-1 != 16:
        print("Skipping:", path)
        continue

    pts = centroids[1:]   # remove background
    pts=order_points(pts)
    objpoints.append(objp)
    imgpoints.append(pts.astype(np.float32))


print("Images used:", len(objpoints))

h, w = img16.shape

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,(w, h),None,None)

print("Camera matrix:", K)
print("Distortion:", dist)
print("Reprojection error:", ret)

#Sanity check: If reprojection error is >1px, order_points probably did not work correctly. May need to exclude such images from the calibration dataset
save_file = "camera_params.npz"

np.savez(save_file,ret=ret,K=K,dist=dist,rvecs=rvecs,tvecs=tvecs)

print(f"Camera parameters saved to {save_file}")
