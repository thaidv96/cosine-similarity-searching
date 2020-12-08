import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from copy import deepcopy
import time

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)



class CosineLSH(object):
    def __init__(self, base_vectors = None, n_vectors = 16, dim = 512, db_config= None, num_tables = 10):
        if base_vectors == None:
            self.base_vectors = [np.random.randn(n_vectors,dim) for i in range(num_tables)]
            self.n_vectors = n_vectors
        else:
            self.n_vectors = base_vectors.shape[0]
        self.tables = [defaultdict(list) for i in range(num_tables)]
    
    def index_one(self, vector, name):
        for base_vector, table in zip(self.base_vectors, self.tables):
            index = vector.dot(self.base_vectors.T) > 0
            index = (2**(np.array(range(self.n_vectors))) * index).sum()
            table[index].append(name)

    # def insert_one(self, index, name):
    #     current_set = self.table.get(index,[])
    #     current_set.append(name)
    #     self.table[index] = current_set


    def index_batch(self, vectors, names):
        for base_vector,table in zip(self.base_vectors,self.tables):
            indices = vectors.dot(base_vector.T) > 0
            indices = indices.dot(2**(np.array(range(self.n_vectors))))
            for index, name in tqdm(zip(indices, names), total = len(names)):
                table[index].append(name)

    def query_one(self, vector, top_k = 5):
        res = set()
        for base_vector, table in zip(self.base_vectors, self.tables):
            index = vector.dot(base_vector.T) > 0
            index = (2**(np.array(range(self.n_vectors))) * index).sum()
            res |= set(table.get(index,[]))
        return res

if __name__ == '__main__':
    lsh = CosineLSH(n_vectors=16, num_tables = 100)
    base_vectors = np.random.randn(100000,512)                                                                                                                                   
    names = range(100000)                                                                                                                                                        
    lsh.index_batch(base_vectors, names)      
    base_vector = deepcopy(base_vectors[10])
    noise = np.random.randn(512)/3
    vec = base_vector + noise
    print("BASE VECTORS SIZE:", sizeof_fmt(sys.getsizeof(lsh.base_vectors)))
    print("LSH TABLE SIZE:", sizeof_fmt(sys.getsizeof(lsh.tables[0])*100))



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
