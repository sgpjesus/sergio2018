
import numpy as np
import scipy
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab
from spacy.language import Language


def tokenization_process(string):
    nlp = Language(Vocab())
    tokenizer = Tokenizer(nlp.vocab)
    aux = tokenizer(string)
    output = list()
    for index, word in enumerate(aux):
        if re.search('([A-Za-z0-9À-ÿ]+(-|\.)[A-Za-z0-9À-ÿ]+|[A-Za-z0-9À-ÿ]+)', str(word)):
                output.append(str(re.search('([A-Za-z0-9À-ÿ]+(-|\.)[A-Za-z0-9À-ÿ]+|[A-Za-z0-9À-ÿ]+)', str(word)).group(0)))
    return output


class DocumentDataFrame:
    def __init__(self, data_series):
        if type(data_series) == pd.core.series.Series:
            self.data_series = data_series
            self.n_grams = 1
            self.max_df = 1.0
            self.min_df = 1
            self.tokenizer = tokenization_process

    def set_params(self, n_grams=1, max_doc_frequency=1.0, min_doc_frequency=1, tokenizer=tokenization_process):
        self.n_grams = n_grams
        self.max_df = max_doc_frequency
        self.min_df = min_doc_frequency
        self.tokenizer = tokenizer

    def count_matrix(self):
        count_model = CountVectorizer(max_df=self.max_df, ngram_range=(1, self.n_grams), min_df=self.min_df,
                                      tokenizer=self.tokenizer)
        return count_model.fit_transform(self.data_series), count_model

    def tf_idf_matrix(self):
        tf_idf = TfidfVectorizer(max_df=self.max_df, ngram_range=(1, self.n_grams), min_df=self.min_df,
                                 tokenizer=self.tokenizer)
        return tf_idf.fit_transform(self.data_series), tf_idf

    def normalize_count_matrix(self):
        count_matrix = self.count_matrix()[0]
        normalized_count_matrix = scipy.sparse.lil_matrix((count_matrix.shape[0], count_matrix.shape[1]))
        for row_id in range(count_matrix.shape[0]):
            if count_matrix[row_id,:].max()!= 0:
                normalized_count_matrix[row_id,:] = count_matrix[row_id,:]/count_matrix[row_id,:].max()
        return normalized_count_matrix

    def cbtw_vec(self, y):
        term_frequency_matrix = self.count_matrix()[0]
        y = np.array([y]).transpose()
        print(type(term_frequency_matrix))
        matrix_indexes = term_frequency_matrix.nonzero()
        data_indexes = np.where(term_frequency_matrix.data != 0)[0]
        n_data = len(data_indexes)
        unitary_matrix = scipy.sparse.csr_matrix((np.ones(n_data), (matrix_indexes[0][data_indexes],
                                                                    matrix_indexes[1][data_indexes])),
                                                 shape=term_frequency_matrix.shape)
        a_matrix = unitary_matrix.multiply(y)
        a_vector = a_matrix.sum(axis=0).transpose()
        inverse_vec = np.abs(y-1)
        b_matrix = unitary_matrix.multiply(inverse_vec)
        b_vector = b_matrix.sum(axis=0).transpose()
        c_vector = np.ones(a_vector.shape[1])*y.sum()-a_vector
        b_vector[b_vector == 0] = 0.5
        c_vector[c_vector == 0] = 0.5
        return np.squeeze(np.asarray(np.log(1+np.multiply((a_vector/b_vector),(a_vector/c_vector)))))

    def cbtw_matrix(self, y):
        vec = self.cbtw_vec(y)
        matrix = self.normalize_count_matrix()
        return scipy.sparse.csr_matrix(matrix.multiply(vec))
