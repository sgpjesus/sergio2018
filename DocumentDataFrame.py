"""First module"""
import numpy as np
import scipy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab
from spacy.language import Language


def tokenization_process(string):
    """ Standard spacy lib Tokenizer"""
    nlp = Language(Vocab())
    tokenizer = Tokenizer(nlp.vocab)
    aux = tokenizer(string)
    output = [0] * len(aux)
    for index, word in enumerate(aux):
        output[index] = str(word)
    return output


class DocumentDataFrame:
    """Class intended to create inputs for machine learning with text documents"""
    def __init__(self, train_series=None, test_series=None):
        if isinstance(train_series, pd.Series):
            self._train_series = train_series
            self._test_series = test_series
            self.n_grams = 1
            self.max_df = 1.0
            self.min_df = 1
            self.tokenizer = tokenization_process

    def __set_params(self, n_grams=1, max_doc_frequency=1.0, min_doc_frequency=1,
                     tokenizer=tokenization_process):
        """Method to change the TF-IDF and TF matrixes parameters"""
        self.n_grams = n_grams
        self.max_df = max_doc_frequency
        self.min_df = min_doc_frequency
        self.tokenizer = tokenizer

    def count_matrix(self, matrix):
        """Create TF matrix for wanted set set"""

        count_model = CountVectorizer(max_df=self.max_df, ngram_range=(1, self.n_grams),
                                      min_df=self.min_df, tokenizer=self.tokenizer)
        if matrix == 'train':
            return count_model.fit_transform(self._train_series)
        elif matrix == 'test':
            count_model.fit(self._train_series)
            return count_model.transform(self._test_series)

    def tf_idf_matrix(self, matrix):
        """Create TF-IDF matrix"""
        tf_idf = TfidfVectorizer(max_df=self.max_df, ngram_range=(1, self.n_grams),
                                 min_df=self.min_df, tokenizer=self.tokenizer)
        if matrix == 'train':
            return tf_idf.fit_transform(self._train_series)
        elif matrix == 'test':
            tf_idf.fit(self._train_series)
            return tf_idf.transform(self._test_series)

    def normalize_count_matrix(self, matrix):
        """Create a normalized TF matrix"""
        count_matrix = self.count_matrix(matrix)
        normalized_count_matrix = scipy.sparse.lil_matrix((count_matrix.shape[0],
                                                           count_matrix.shape[1]))
        for row_id in range(count_matrix.shape[0]):
            if count_matrix[row_id, :].max() != 0:
                normalized_count_matrix[row_id, :] = (count_matrix[row_id, :]/
                                                      count_matrix[row_id, :].max())
        return normalized_count_matrix

    def cbtw_vec(self, class_values, matrix):
        """Return for a certain class the vector of CBTW values"""
        term_frequency_matrix = self.count_matrix(matrix)
        class_values = np.array([class_values]).transpose()
        matrix_indexes = term_frequency_matrix.nonzero()
        data_indexes = np.where(term_frequency_matrix.data != 0)[0]
        n_data = len(data_indexes)
        unitary_matrix = scipy.sparse.csr_matrix((np.ones(n_data),
                                                  (matrix_indexes[0][data_indexes],
                                                   matrix_indexes[1][data_indexes])),
                                                 shape=term_frequency_matrix.shape)
        a_matrix = unitary_matrix.multiply(class_values)
        a_vector = a_matrix.sum(axis=0).transpose()
        inverse_vec = np.abs(class_values-1)
        b_matrix = unitary_matrix.multiply(inverse_vec)
        b_vector = b_matrix.sum(axis=0).transpose()
        c_vector = np.ones(a_vector.shape[1])*class_values.sum()-a_vector
        b_vector[b_vector == 0] = 0.5
        c_vector[c_vector == 0] = 0.5
        return np.squeeze(np.asarray(np.log(1+np.multiply((a_vector/b_vector),
                                                          (a_vector/c_vector)))))

    def cbtw_matrix(self, yvec, matrix):
        """Return of the CBTW1 matrix for the model"""
        vec = self.cbtw_vec(yvec, matrix)
        matrix = self.normalize_count_matrix(matrix)
        return matrix.multiply(vec)
