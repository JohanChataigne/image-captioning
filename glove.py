import pandas as pd

class Glove:
    
    def __init__(self, directory='./', dimension=300):
        
        assert dimension in [50, 100, 200, 300]
        
        self.get_glove(directory, dimension)
        
        
        
    def get_glove(self, directory, dimension):
        
        glove = pd.read_csv(directory + 'glove.6B.' + str(dimension) +'d.txt', sep=' ', quoting=3, header=None, index_col=0)
        self.embedding_dict = {key: val.values for key, val in glove.T.items()}
        
        print(self.embedding_dict)
        
        
        
        
g = Glove()
        