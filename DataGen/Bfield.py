import DataGen.BB_packaging as BB
import numpy as np

def parseData(path):
	lines = []
	with open(path) as f:
		lines = f.readlines()

	actData = []
	for l in lines:
		lsplit = l.split()
		arr = []
		for ls in lsplit:
			arr.append(float(ls))
		actData.append(arr)

	return actData

actData = np.array(parseData('DataGen/matlab/actuatorPosition.txt'))
MMData = np.array(parseData('DataGen/matlab/MM.txt'))

def getOri(pos, curr):

	pos = np.array(pos)
	curr = np.array(curr)

	BM = np.matmul(BB.packageBB(pos,actData), MMData)

	b_des = np.matmul(curr, np.transpose(BM))

	return b_des/(3e-3)

def getCurrents(pos, ori):
	
	pos = np.array(pos)
	ori = np.array(ori)

	ori = ori/np.linalg.norm(ori,2)
	b_des = ori * 3e-3

	BM = np.matmul(BB.packageBB(pos,actData), MMData)

	II = np.linalg.pinv(BM)*b_des

	return II

if __name__ == "__main__":

	# print( getOri( [[0.05],[0.02],[0.15]] , [0.5, 2.5, 2.2, 2.1, -1.5, -1.3, 2, 2, 2] ) )

	pos = np.array( [[0.05],[0.02],[0.15]] )
	curr = np.array( [0.5, 2.5, 2.2, 2.1, -1.5, -1.3, 2, 2, 2] )

	ori = getOri(pos, curr)

	gcurr = getCurrents(pos, ori)
	gcurr = np.array(gcurr)

	print(curr - gcurr)

