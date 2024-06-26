import cv2
import os
from matplotlib import pyplot as plt

if __name__ == "__main__":
	fs = cv2.FileStorage("stereoMap.txt", cv2.FILE_STORAGE_READ)
	lx = fs.getNode("lmapx").mat()
	ly = fs.getNode("lmapy").mat()
	rx = fs.getNode("rmapx").mat()
	ry = fs.getNode("rmapy").mat()

	cv2.namedWindow("right")
	cv2.namedWindow("left")
	cv2.namedWindow("depth")

	rcam = cv2.VideoCapture(0)
	lcam = cv2.VideoCapture(1)

	stereo = cv2.StereoBM.create(numDisparities=32, blockSize=5)

	while True:
		retr, rframe = rcam.read()
		retl, lframe = lcam.read()

		rectr = cv2.remap(rframe, rx, ry, cv2.INTER_LINEAR)
		rectl = cv2.remap(lframe, lx, ly, cv2.INTER_LINEAR)

		gr = cv2.cvtColor(rectr,cv2.COLOR_BGR2GRAY)
		gl = cv2.cvtColor(rectl,cv2.COLOR_BGR2GRAY)
		disparity = stereo.compute(gr,gl)
		norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		cv2.imshow("right",rectr)
		cv2.imshow("left",rectl)
		cv2.imshow("depth",norm_image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	rcam.release()
	lcam.release()
	cv2.destroyAllWindows()