# from data_parser import Parser
from .data_parser import Parser
import numpy as np

class DataReader():

    def __init__(self):
        self.parser = Parser()
        self.parser.parsefile()
        parsed_data = self.parser.get_data()
        parsed_data = np.array(parsed_data)
        self.data = parsed_data
        self.reset()
        
    def get_all_data(self):
        return self.data

    def reset(self):

        # Shuffling
        permutations = np.random.permutation(self.data.shape[0])
        self.data = self.data[permutations]
        
        # Resetting iter
        self.iter = -1
        self.total_data = self.data.shape[0]

    def get_next(self):
        self.iter += 1
        assert(not self.iter < 0)

        if self.iter < self.total_data:
            return self.data[self.iter]
        else:
            self.reset()
            return self.get_next()

    def get_data(self):
        return self.data[self.iter]
    
if __name__ == '__main__':
    DR = DataReader()

    min = 1000
    max = -1000
    data =  DR.get_all_data()
    for j in data:
        arr = np.array(j)
        arr_min = arr.min()
        arr_max = arr.max()

        if(min > arr_min):
            min = arr_min

        if(max < arr_max):
            max = arr_max
    
    print("Min/Max: {}/{}".format(min,max))
