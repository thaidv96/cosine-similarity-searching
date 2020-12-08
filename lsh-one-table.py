import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from copy import deepcopy
import time
from itertools import combinations
import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)



class CosineLSH(object):
    def __init__(self, base_vectors = None, n_vectors = 16, dim = 512, db_config= None):
        if base_vectors == None:
            self.base_vectors = np.random.randn(n_vectors,dim)
            self.n_vectors = n_vectors
        else:
            self.n_vectors = base_vectors.shape[0]
        self.table = defaultdict(list)
    
    def index_one(self, vector, name):
        index = vector.dot(self.base_vectors.T) > 0
        index = (2**(np.array(range(self.n_vectors))) * index).sum()
        self.table[index].append(name)

    # def insert_one(self, index, name):
    #     current_set = self.table.get(index,[])
    #     current_set.append(name)
    #     self.table[index] = current_set


    def index_batch(self, vectors, names):
        indices = vectors.dot(self.base_vectors.T) > 0
        indices = indices.dot(2**(np.array(range(self.n_vectors))))
        for index, name in tqdm(zip(indices, names), total = len(names)):
            self.table[index].append(name)

    def get_index_with_radius(self,index, radius):
        res = []
        for r in range(radius):
            nearby_index = index.copy()
            for _change in combinations(range(self.n_vectors),r):
                _index = list(_change)
                nearby_index[_index] = np.logical_not(nearby_index[_index])
                _index = (2**(np.array(range(self.n_vectors))) * nearby_index).sum()
                res.append(_index)
        return res

    def query_one(self, vector, top_k = 5, radius = 7):
        res = set()
        index = vector.dot(self.base_vectors.T) > 0
        nearby_indices = self.get_index_with_radius(index, radius)
        print("# Nearby indices", len(nearby_indices))
        for _idx in nearby_indices:
            res |= set(self.table[_idx])
        return res

if __name__ == '__main__':
    lsh = CosineLSH(n_vectors=16)
    base_vectors = np.random.randn(100000,512)                                                                                                                                   
    names = range(100000)                                                                                                                                                        
    lsh.index_batch(base_vectors, names)      
    base_vector = deepcopy(base_vectors[10])
    noise = np.random.randn(512)/3
    vec = base_vector + noise
    print("BASE VECTORS SIZE:", sizeof_fmt(sys.getsizeof(lsh.base_vectors)))
    print("LSH TABLE SIZE:", sizeof_fmt(sys.getsizeof(lsh.table)))



    for i in range(30):
        print("Cosine similarity to target:", cosine_similarity(vec.reshape(1,-1), base_vector.reshape(1,-1)))
        start = time.time()        
        result = lsh.query_one(vec)
        print("Query time", time.time() - start)
        print("10 in result",10 in result)
        print("LEN RESULT", len(result))
        noise = np.random.randn(512)/3
        vec += noise
    time.sleep(50)
