import cv2

if __name__ == "__main__":

	cap = cv2.VideoCapture(1)
	cv2.namedWindow("stream")
	i = 0
	numSamples = 15
	while True:
		ret, frame = cap.read()
		cv2.imshow("stream",frame)
		key = cv2.waitKey(1)
		if key % 256 == 27:
			break
		elif key % 256 == 32:
			img = f"samp_{i}.jpg"
			cv2.imwrite(img,frame)
			i = i + 1
			print(i)
		if i > numSamples:
			break

	cap.release()
	cv2.destroyAllWindows()
