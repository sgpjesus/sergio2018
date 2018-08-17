# Parameter calculations for NLP
#
# Author: Sérgio Jesus <sergiogabrielpts@gmail.com>
# git: <https://github.com/sgpjesus/sergio2018.git>

import numpy as np
import scipy
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab
from spacy.language import Language


def tokenization_process(string):
    r""""
    For a given string, the tokenization occurs using SpaCy classes and methods,
    optimized with RegEx for Portuguese words
    :param string: input string (document)
    :return: List of tokens
    """

    nlp = Language(Vocab())
    tokenizer = Tokenizer(nlp.vocab)  # SpaCy classes

    aux = tokenizer(string)  # SpaCy result

    output = list()
    for index, word in enumerate(aux):
        if re.search('([A-Za-z0-9À-ÿ]+(-)[A-Za-z0-9À-ÿ]+|[A-Za-z0-9À-ÿ]+)',
                     str(word)):
                output.append(str(re.search(
                    '([A-Za-z0-9À-ÿ]+(-)[A-Za-z0-9À-ÿ]+|[A-Za-z0-9À-ÿ]+)',
                    str(word)).group(0)))
    return output


class DocumentDataFrame:
    def __init__(self, train_series=None, test_series=None):
        r"""
        Initialize a DataFrame-kind class; Class has methods to calculate
        TF-IDF, TF and CBTW parameters matrix
        :param train_series: Series with the training data
        :param test_series: Series with test data
        """
        if isinstance(train_series, pd.Series):

            self._train_series = train_series
            self._test_series = test_series
            self.n_grams = 1
            self.max_df = 1.0
            self.min_df = 1
            self.tokenizer = tokenization_process
            self.count_model = 0
            self.tf_idf = 0

    def _set_params(self, n_grams=1, max_doc_frequency=1.0, min_doc_frequency=1,
                    tokenizer=tokenization_process):
        r"""
        Sets new parameters to the class
        :param n_grams: number of sequenced tokens to be considered a parameter
        :param max_doc_frequency: Maximum number of documents a token can appear
        :param min_doc_frequency: Minimum number of documents a token can appear
        :param tokenizer: A function to tokenize the string
        """

        self.n_grams = n_grams
        self.max_df = max_doc_frequency
        self.min_df = min_doc_frequency
        self.tokenizer = tokenizer

    def count_matrix(self, matrix):
        r"""
        Creates a count matrix using sklearn feature extraction method
        Note that a training matrix shall always be used before the test matrix
        :param matrix: String, define what matrix should be used (Test or train)
        :return: TF matrix (sparse csr matrix)
        """

        if matrix == 'train':
            # Defining the count class
            self.count_model = CountVectorizer(max_df=self.max_df,
                                               ngram_range=(1, self.n_grams),
                                               min_df=self.min_df,
                                               tokenizer=self.tokenizer)

            return self.count_model.fit_transform(self._train_series)

        elif matrix == 'test':

            return self.count_model.transform(self._test_series)

    def tf_idf_matrix(self, matrix):
        r"""
        Creates a TF-IDF matrix using sklearn feature extraction method
        Note that a training matrix shall always be used before the test matrix
        :param matrix: string, define what matrix should be used (Test or train)
        :return: TF-IDF matrix (sparse csr matrix)
        """

        if matrix == 'train':
            # Defining the TF-IDF class
            self.tf_idf = TfidfVectorizer(max_df=self.max_df,
                                          ngram_range=(1, self.n_grams),
                                          min_df=self.min_df,
                                          tokenizer=self.tokenizer)

            return self.tf_idf.fit_transform(self._train_series)

        elif matrix == 'test':

            return self.tf_idf.transform(self._test_series)

    def normalize_count_matrix(self, matrix):
        r"""
        Creates a normalized count matrix, as defined in the CBTW paper
        Note that a training matrix shall always be used before the test matrix
        :param matrix: String, define what matrix should be used (Test or train)
        :return: Normalized TF matrix by line maximum value (sparse csr matrix)
        """

        # Obtaining the regular count matrix
        count_matrix = self.count_matrix(matrix)
        normalized_count_matrix = scipy.sparse.lil_matrix((count_matrix.shape[0]
                                                           ,
                                                           count_matrix.shape[1]
                                                           ))

        # Normalizing the matrix with the max term in row
        for row_id in range(count_matrix.shape[0]):

            # Avoiding the division by 0
            if count_matrix[row_id, :].max() != 0:
                # Avoiding appends
                normalized_count_matrix[row_id, :] = (count_matrix[row_id, :] /
                                                      count_matrix[row_id, :].
                                                      max())

        return normalized_count_matrix

    def cbtw_vec(self, class_values, matrix):
        r"""
        Creates a vector of weights to multiply the lines of the normalized term
        frequency matrix, as proposed in the CBTW paper
        :param class_values: pd.Series or list of binary classification, Labels
        of the training sample
        :param matrix: String, define what matrix should be used (Test or train)
        :return: vector of CBTW weights (np.array)
        """

        term_frequency_matrix = self.count_matrix(matrix)

        class_values = np.array([class_values]).transpose()

        matrix_indexes = term_frequency_matrix.nonzero()
        data_indexes = np.where(term_frequency_matrix.data != 0)[0]
        n_data = len(data_indexes)

        unitary_matrix = scipy.sparse.csr_matrix((np.ones(n_data),
                                                  (matrix_indexes[0]
                                                   [data_indexes],
                                                   matrix_indexes[1]
                                                   [data_indexes])),
                                                   shape=term_frequency_matrix.
                                                   shape)

        inverse_vec = np.abs(class_values - 1)

        # Vectors created without the iterative method for faster processing
        # times (less verbose)
        a_matrix = unitary_matrix.multiply(class_values)
        a_vector = a_matrix.sum(axis=0).transpose()

        b_matrix = unitary_matrix.multiply(inverse_vec)
        b_vector = b_matrix.sum(axis=0).transpose()

        c_vector = np.ones(a_vector.shape[1])*class_values.sum()-a_vector

        b_vector[b_vector == 0] = 0.5  # Avoid the division by 0
        # (not contemplated in the paper)
        c_vector[c_vector == 0] = 0.5  # Avoid the division by 0
        # (not contemplated in the paper)

        return np.squeeze(np.asarray(np.log(1+np.multiply((a_vector/b_vector),
                                                          (a_vector/c_vector))))
                          )

    def cbtw_matrix(self, class_values, matrix):
        r"""
        Creates the feature matrix of the CBTW method, as proposed in the CBTW
        Paper. Includes vector creation, this function is  the pipeline for
        CBTW method
        :param class_values: pd.Series or list of binary classification, Labels
        of the training sample
        :param matrix: string, define what matrix should be used (Test or train)
        :return: CBTW Feature matrix (sparse csr matrix)
        """

        vec = self.cbtw_vec(class_values, matrix)
        matrix = self.normalize_count_matrix(matrix)

        return scipy.sparse.csr_matrix(matrix.multiply(vec))
