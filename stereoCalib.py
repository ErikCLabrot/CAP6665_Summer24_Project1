import cv2 as cv
import numpy as np
import os
import glob
if __name__ == "__main__":

	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	fs = cv.FileStorage("left_cal.txt", cv.FILE_STORAGE_READ)
	left_mtx = fs.getNode("camera_matrix").mat()
	left_dist = fs.getNode("dist_coeffs").mat()
	fs.release()

	fs = cv.FileStorage("right_cal.txt", cv.FILE_STORAGE_READ)
	right_mtx = fs.getNode("camera_matrix").mat()
	right_dist = fs.getNode("dist_coeffs").mat()
	fs.release()

	chessboardSize = (9,6)

	objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
	objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

	size_of_chessboard_squares_mm = 20
	objp = objp * size_of_chessboard_squares_mm

	objpoints = [] # 3d point in real world space
	imgpointsL = [] # 2d points in image plane.
	imgpointsR = [] # 2d points in image plane.

	imagesLeft = sorted(glob.glob('stereosamp/l/*.jpg'))
	imagesRight = sorted(glob.glob('stereosamp/r/*.jpg'))

	i = 0

	for imgLeft, imgRight in zip(imagesLeft, imagesRight):

		imgL = cv.imread(imgLeft)
		imgR = cv.imread(imgRight)
		grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
		grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

		# Find the chess board corners
		retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
		retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

		cv.drawChessboardCorners(imgL,chessboardSize,cornersL,retL)
		fname = 'stereosamp/l/corners/corner_' + str(i) + '.jpg'
		cv.imwrite(fname,imgL)

		cv.drawChessboardCorners(imgR,chessboardSize,cornersR,retR)
		fname = 'stereosamp/r/corners/corner_' + str(i) + '.jpg'
		cv.imwrite(fname,imgR)

		i = i + 1

	# If found, add object points, image points (after refining them)
		if retL and retR == True:

			objpoints.append(objp)

			cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
			imgpointsL.append(cornersL)

			cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
			imgpointsR.append(cornersR)

	err, Kl, Dl, Kr, Dr, R, T, E, F = cv.stereoCalibrate(
		objpoints, imgpointsL, imgpointsR, left_mtx, left_dist, right_mtx, right_dist, grayL.shape[::-1], flags=cv.CALIB_FIX_INTRINSIC)

	print(err)
	print(R)
	print(T)
	print(E)
	print(F)

	fs = cv.FileStorage("stereoCal.txt", cv.FILE_STORAGE_WRITE)
	fs.write("err",err)
	fs.write("Kl",Kl)
	fs.write("Dl",Dl)
	fs.write("Kr",Kr)
	fs.write("Dr",Dr)
	fs.write("R",R)
	fs.write("T",T)
	fs.write("E",E)
	fs.write("F",F)
	fs.release()