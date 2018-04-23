"""Test module"""
#%%
import pandas as pd
import Class
import scipy
from DocumentDataFrame import DocumentDataFrame
import testclass

#%%
TRAIN_TABLE = pd.read_csv('train.csv')
D = Class.DocumentDataFrame(TRAIN_TABLE.comment_text[[0, 1, 2, 3, 4, 5, 6]])
B = D.cbtw_vec(TRAIN_TABLE.toxic[[0, 1, 2, 3, 4, 5, 6]])
C = D.normalize_count_matrix()
E = D.count_matrix()
F = D.cbtw_matrix(TRAIN_TABLE.toxic[[0, 1, 2, 3, 4, 5, 6]])
#%%

X = testclass.mango()
X.funct2()
print(C)
print(scipy.sparse.lil_matrix(E))
print(B.shape)
print(C.shape)
print(C.multiply(B))
print(type(B))
print(type(TRAIN_TABLE.toxic[[0, 1, 2, 3, 4, 5, 6]]))
help(DocumentDataFrame)
