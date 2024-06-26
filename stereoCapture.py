import cv2
import os

if __name__ == "__main__":

	sc = 15

	rcam = cv2.VideoCapture(0)
	lcam = cv2.VideoCapture(1)

	cv2.namedWindow("right")
	cv2.namedWindow("left")

	i = 0
	numSamples = 15

	while True:
		ret1, rframe = rcam.read()
		ret2, lframe = lcam.read()

		cv2.imshow("right",rframe)
		cv2.imshow("left",lframe)
		key = cv2.waitKey(1)
		if key % 256 == 27:
			break
		elif key % 256 == 32:
			img = f"stereosamp/r/rsamp_{i}.jpg"
			cv2.imwrite(img,rframe)
			img = f"stereosamp/l/lsamp_{i}.jpg"
			cv2.imwrite(img,lframe)
			i = i + 1
			print(i)
		if i >= numSamples:
			break