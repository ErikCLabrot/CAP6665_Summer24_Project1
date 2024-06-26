import cv2
import os
import numpy as np

if __name__ == "__main__":
	#Load calib data
	fs = cv2.FileStorage("stereoCal.txt", cv2.FILE_STORAGE_READ)
	left_mtx = fs.getNode("Kl").mat()
	left_dist = fs.getNode("Dl").mat()
	right_mtx  = fs.getNode("Kr").mat()
	right_dist = fs.getNode("Dr").mat()
	R = fs.getNode("R").mat()
	T = fs.getNode("T").mat()
	E = fs.getNode("E").mat()
	F = fs.getNode("F").mat()
	fs.release()

	#Get stereo sample
	rcam = cv2.VideoCapture(0)
	lcam = cv2.VideoCapture(1)

	cv2.namedWindow("right")
	cv2.namedWindow("left")

	while True:
		ret1, rframe = rcam.read()
		ret2, lframe = lcam.read()

		cv2.imshow("right", rframe)
		cv2.imshow("left", lframe)
		key = cv2.waitKey(1)

		if key % 256 == 27:
			break
		elif key % 256 == 32:
			imgR = rframe
			imgL = lframe
			break

	imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
	imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

	rL, rR, pMatL, pMatR, Q, roiL, roiR = cv2.stereoRectify(left_mtx, left_dist, right_mtx, right_dist, imgL.shape[::-1],R,T,1,(0,0))

	l_mapx,l_mapy = cv2.initUndistortRectifyMap(left_mtx, left_dist, rL, pMatL, imgL.shape[::-1], cv2.CV_16SC2)
	r_mapx, r_mapy = cv2.initUndistortRectifyMap(right_mtx, right_dist, rR, pMatR, imgR.shape[::-1], cv2.CV_16SC2)

	fs = cv2.FileStorage("stereoMap.txt", cv2.FILE_STORAGE_WRITE)
	fs.write("lmapx",l_mapx)
	fs.write("lmapy",l_mapy)
	fs.write("rmapx",r_mapx)
	fs.write("rmapy",r_mapy)
	fs.release

	rect_imgL = cv2.remap(imgL, l_mapx, l_mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
	rect_imgR = cv2.remap(imgR, r_mapx, r_mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

	cv2.imshow('Rectified Left Image', rect_imgL)
	cv2.imshow('Rectified Right Image', rect_imgR)
	cv2.waitKey(0)
	cv2.destroyAllWindows()