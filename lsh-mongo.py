import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from copy import deepcopy
import time
from pymongo import MongoClient
from itertools import combinationss

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)



class CosineLSH(object):
    def __init__(self,client, db_name, n_vectors = 16, dim = 512, num_tables = 100):
        dbnames = client.list_database_names()
        if db_name in dbnames:
            self.base_vectors = client[db_name]['base_vectors'].find_one()
            del self.base_vectors['_id']
            self.base_vectors = {k: np.array(v) for k,v in self.base_vectors.items()}
            self.n_vectors = self.base_vectors['table_0'].shape[0]
            self.num_tables = len(self.base_vectors)
            self.dim = self.base_vectors['table_0'].shape[1]
        else:
            self.base_vectors = {f'table_{i}':np.random.randn(n_vectors,dim) for i in range(num_tables)}
            client[db_name]['base_vectors'].insert_one({k:v.tolist() for k,v in self.base_vectors.items()})
            self.num_tables = num_tables
            self.dim = dim
            self.n_vectors = n_vectors

        self.tables = [client[db_name][f'table_{i}'] for i in range(num_tables)]
        for table in self.tables:
            table.create_index("bucket")
        self.db = client[db_name]

    def index_one(self, vector, name):
        for table_idx in range(self.num_tables):
            index = vector.dot(self.base_vectors[f'table_{i}'].T) > 0
            index = (2**(np.array(range(self.n_vectors))) * index).sum()
            self.insert_by_index(index, [name], self.db[f'table_{table_idx}'])
    

    # def insert_one(self, index, name):
    #     current_set = self.table.get(index,[])
    #     current_set.append(name)
    #     self.table[index] = current_set

    def insert_by_index(self, index, names, table):
        index = int(index)
        current_vals = table.find_one({"bucket":index})
        if current_vals !=None:
            current_vals = current_vals['names']
            current_vals = list(set(current_vals) | set(names))
            table.find_one_and_update({"bucket":index},{'$set':{'names': current_vals}})
        else:
            table.insert_one({"bucket":index, "names":names})


    def index_batch(self, vectors, names,pbar =None):
        for table_idx in range(self.num_tables):

            base_vector = self.base_vectors[f'table_{table_idx}']
            indices = vectors.dot(base_vector.T) > 0
            indices = indices.dot(2**(np.array(range(self.n_vectors))))
            batch_temp_table = defaultdict(list)
            for index, name in zip(indices, names):
                batch_temp_table[index].append(name)
            if type(pbar) == None:
                pbar = tqdm()
            _i = 0
            _total = len(batch_temp_table)
            for index, _names in batch_temp_table.items():
                pbar.set_description(f"Indexing... {_i}/{_total}")
                _i+=1
                self.insert_by_index(index, _names, self.db[f'table_{table_idx}'])
    
    def get_index_with_radius(self,index, radius):
        indices = []
        for r in range(radius):
            new_index = deepcopy(index)
            for _change in combinations(range(len(index)),r):
                _index = list(_change)
                new_index[_index] = np.logical_not(new_index[_index])
                _index = (2**(np.array(range(self.n_vectors))) * new_index).sum()
                indicies.append(_index)
        return indicies


    def query_one(self, vector, top_k = 5,radius=2):
        res = set()
        for table_idx in range(self.num_tables):
            base_vector = self.base_vectors[f'table_{table_idx}']
            table = self.db[f'table_{table_idx}']
            index = vector.dot(base_vector.T) > 0
            index = (2**(np.array(range(self.n_vectors))) * index).sum()
            table_vals = table.find_one({"bucket":int(index)})
            if table_vals != None:
                res |= set(table_vals['names'])
        return res

if __name__ == '__main__':
    client = MongoClient()
    db_name = 'test'
    client.drop_database(db_name)
    num_per_epoch = 50000
    lsh = CosineLSH(db_name=db_name, client = client, num_tables=1)
    pbar = tqdm(range(20))
    for i in pbar:
        pbar.set_description(f"Generating {num_per_epoch} random vectors")
        base_vectors = np.random.randn(num_per_epoch,512)
        names = range(num_per_epoch* i, num_per_epoch*(i+1)) 
        pbar.set_description(f"Indexing...")                                                                                                                                                       
        lsh.index_batch(base_vectors, names,pbar)
        pbar.set_description("Ingesting base data into db")
        client[db_name]['base'].insert_many([{'name':name, 'vector': v.tolist()} for name, v in zip(names, base_vectors)])      
    noise = np.random.randn(512)/3
    base_vector = client.base.find_one()
    vec = np.array(base_vector['vector']) + noise

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
