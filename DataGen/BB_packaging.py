import numpy as np
import math

def packageBB(ri, rj):
	n_s = ri.shape[1]
	n_a = rj.shape[1]

	BB = np.zeros((3*n_s,3*n_a))
	mu = 4.0*math.pi*1e-7

	for i_s in range(n_s):
		r_s = ri[:,i_s]

		for i_a in range(n_a):
			r_a = rj[:,i_a]

			p = r_s-r_a
			
			x = p[0]
			y = p[1]
			z = p[2]
			
			r = np.sqrt(np.square(x)+np.square(y)+np.square(z))
			x = x/r
			y = y/r
			z = z/r
			
			P = [ 	[3*np.square(x) - 1,     3*x*y,     3*x*z],
            		[3*x*y, 3*np.square(y) - 1,     3*y*z],
            		[3*x*z,     3*y*z, 3*np.square(z) - 1] 	]
				
			P = P/np.power(r,3)
			
			
			BB[3*(i_s+1)-3:3*(i_s+1) , 3*(i_a+1)-3:3*(i_a+1)] =  P
	
	BB = (mu/4/math.pi) *BB
	return BB