import os
import cv2
import numpy as np

class monoCalibration:
	def __init__(self,cb_size,folder):
		self.cb_size = cb_size
		self.worldpoints = []
		self.imgpoints = []
		self.objp = []
		self.folder = folder
		self.imgSize = 0
		self._prepare_object_points()

	def _prepare_object_points(self):
		objp = np.zeros((self.cb_size[0] * self.cb_size[1], 3), np.float32)
		objp[:, :2] = np.mgrid[0:self.cb_size[0], 0:self.cb_size[1]].T.reshape(-1, 2)
		self.objp = objp*20

    #Load images into memory
    #Refactor: save paths as strings, reduce mem consumption
    ##Maybe combine load & process into one func?
	def load_cb_images(self,folder):
		imgs = []
		for file in os.listdir(folder):
			if file.endswith('.jpg'):
				path = os.path.join(folder,file)
				print(path)
				img = cv2.imread(path)
				imgs.append(img)
		return imgs

	def process_images(self):
		imgs = self.load_cb_images(self.folder)
		i = 1
		for sample in imgs:
			sg = cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(sg, self.cb_size, None)
			self.imgSize = sg.shape
			if ret:
				self.worldpoints.append(self.objp)
				self.imgpoints.append(corners)

				cv2.drawChessboardCorners(sample,self.cb_size,corners,ret)
				fname = 'corners/corner_' + str(i) + '.jpg'
				fname = os.path.join(self.folder,fname)
				print(fname)
				cv2.imwrite(fname,sample)
				i=i+1
			else:
				return False
		return True

	def calibrate(self):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.worldpoints, self.imgpoints, self.imgSize[::-1], None, None)		

		return {
			"ret": ret,
			"mtx": mtx,
			"dist": dist,
			"rvecs": rvecs,
			"tvecs": tvecs
		}

	#def undistort(self,calib):
		#Do undistortion here

	def calc_repro_error(self,calib):
		mean_e = 0
		for i in range(len(self.worldpoints)):
			ip2,_ = cv2.projectPoints(self.worldpoints[i],calib["rvecs"][i],calib["tvecs"][i],calib["mtx"],calib["dist"])
			e = cv2.norm(self.imgpoints[i],ip2,cv2.NORM_L2)/len(ip2)
			mean_e += e
		print("total error: {}".format(mean_e/len(self.worldpoints)))

	def run(self):
		if(self.process_images()):
			cal = self.calibrate()
			self.calc_repro_error(cal)
		else:
			print("Error loading and processing images!")

		return cal


if __name__ == "__main__":
	cb_size = (9,6)
	print("doing right")
	mc = monoCalibration(cb_size,"samples/right")
	cal = mc.run()
	print(cal)
	fs = cv2.FileStorage('right_cal.txt', cv2.FILE_STORAGE_WRITE)
	fs.write("camera_matrix", cal["mtx"])
	fs.write("dist_coeffs", cal["dist"])
	fs.release()

	print("doing left")
	mc = monoCalibration(cb_size, "samples/left")
	cal = mc.run()
	print(cal)

	fs = cv2.FileStorage('left_cal.txt', cv2.FILE_STORAGE_WRITE)
	fs.write("camera_matrix", cal["mtx"])
	fs.write("dist_coeffs", cal["dist"])
	fs.release()