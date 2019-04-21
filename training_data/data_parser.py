
FILE_NAME = "./training_data/data.txt"

class Parser():

	def __init__(self):
		self.parsed_data = list()

	def parsefile(self):
		self.parsed_data = list()
		with open(FILE_NAME, "r") as ins:
			for line in ins:
				array = []
				spline = line.split(",")
				for val in spline:
					array.append(float(val))
				self.parsed_data.append(array)

	def get_data(self):
		assert(len(self.parsed_data)>0)
		return self.parsed_data


