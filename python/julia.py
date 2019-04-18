import time

def timefn(fn):
	def measure_time(*args, **kwargs):
		t1 = time.time()
		result = fn(*args, **kwargs)
		t2 = time.time()
		print("@timefn:" + fn.__name__ + " took " + str(t2-t1) + " seconds")
		return result
	return measure_time


# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193

def calc_pure_python(desired_width, max_iterations):
	"""
	Create a list of complex coordinates (zs) and complex
	parameters (cs), build Julia set, and display 
	"""
	x_step = (float(x2 - x1) / float(desired_width))
	y_step = (float(y1 - y2) / float(desired_width))
	x = []
	y = []
	ycoord = y2
	while ycoord > y1:
		y.append(ycoord)
		ycoord += y_step
	xcoord = x1
	while xcoord < x2:
		x.append(xcoord)
		xcoord += x_step

	zs = []
	cs = []
	for ycoord in y:
		for xcoord in x:
			zs.append(complex(xcoord, ycoord))
			cs.append(complex(c_real, c_imag))

	print("Length of x:", len(x))
	print("Total elements:", len(zs))
	
	start_time = time.time()
	output = calculate_z_serial_purepython(max_iterations, zs, cs)
	end_time = time.time()
	secs = end_time - start_time
	print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

	assert sum(output) == 33219980

@timefn
def calculate_z_serial_purepython(maxiter, zs, cs):
	"""
	Calculate output list using Julia update rule
	"""
	output = [0] * len(zs)
	for i in range (len(zs)):
		n = 0
		z = zs[i]
		c = cs[i]
		while abs(z) < 2 and n < maxiter:
			z = z * z + c
			n += 1
		output[i] = n

	return output

if __name__ == "julia":
	calc_pure_python(desired_width=1000, max_iterations=300)
