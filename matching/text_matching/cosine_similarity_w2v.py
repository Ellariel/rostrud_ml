
import pandas as pd
import numpy as np
from numpy import asarray
from gensim.models import Word2Vec

"""Модуль, который инициализирует класс
CosineSimilarityW2v предназначен для реализации
процедуры расчета косинусного расстояния векторов"""


class CosineSimilarityW2v:
    """Класс CosineSimilarityW2v предназначен для реализации
    процедуры расчета косинусного расстояния векторов"""
    def __init__(self, w2v_model, scaler):
        self.w2v_model = w2v_model
        self.scaler = scaler

    def cosine_similarity_w2v_words_features_cv(self, cv_text: pd.DataFrame, vac_text: pd.DataFrame, counter: int) -> float:
        """Функция расчета кос векторов вариант вакансии
        для резюме(объединения с фичами)"""
        a = []
        b = []
        for word in cv_text.iloc[0, 2]:
            try:
                words = self.w2v_model[word]
                a.append(words)
            except KeyError:
                continue
            except ValueError:
                continue
        for word in vac_text.iloc[counter, 2]:
            try:
                words = self.w2v_model[word]
                b.append(words)
            except KeyError:
                continue
            except ValueError:
                continue
        vector_a = []
        vector_a.append(cv_text.iloc[0, 3])
        vector_a.append(cv_text.iloc[0, 4])
        vector_a.append(cv_text.iloc[0, 5])
        vector_a.append(cv_text.iloc[0, 7])
        vector_b = []
        vector_b.append(vac_text.iloc[counter, 3])
        vector_b.append(vac_text.iloc[counter, 4])
        vector_b.append(vac_text.iloc[counter, 5])
        vector_b.append(vac_text.iloc[counter, 6])
        return self.scaler_vectors(a, b, vector_a, vector_b)

    def cosine_similarity_w2v_words_features_vac(self, vac_text: pd.DataFrame, cv_text: pd.DataFrame, counter: int) -> float:
        """Функция расчета кос векторов вариант резюме
        для вакансии(объединения с фичами)"""
        a = []
        b = []
        for word in vac_text.iloc[0, 2]:
            try:
                words = self.w2v_model[word]
                b.append(words)
            except KeyError:
                continue
            except ValueError:
                continue
        for word in cv_text.iloc[counter, 2]:
            try:
                words = self.w2v_model[word]
                a.append(words)
            except KeyError:
                continue
            except ValueError:
                continue
        vector_a = []
        vector_a.append(cv_text.iloc[counter, 3])
        vector_a.append(cv_text.iloc[counter, 4])
        vector_a.append(cv_text.iloc[counter, 5])
        vector_a.append(cv_text.iloc[counter, 7])
        vector_b = []
        vector_b.append(vac_text.iloc[0, 3])
        vector_b.append(vac_text.iloc[0, 4])
        vector_b.append(vac_text.iloc[0, 5])
        vector_b.append(vac_text.iloc[0, 6])
        return self.scaler_vectors(a, b, vector_a, vector_b)

    def scaler_vectors(self, a, b, vector_a, vector_b) -> float:
        """Функция расчета векторов c дополнительными фичами резюме и вакансий"""
        a_m = np.mean(np.asarray(a), axis=0)
        b_m = np.mean(np.asarray(b), axis=0)
        vector_a = np.array(vector_a)
        vector_a = self.scaler.transform([vector_a]).reshape(4,)
        vector_b = np.array(vector_b)
        vector_b = self.scaler.transform([vector_b]).reshape(4,)
        vector_a = vector_a/10
        vector_b = vector_b/10
        norm_a_m = np.concatenate((vector_a, a_m))
        norm_b_m = np.concatenate((vector_b, b_m))
        cosine_similarity_ = np.dot(norm_a_m, norm_b_m) / (np.linalg.norm(norm_a_m) * np.linalg.norm(norm_b_m))
        return cosine_similarity_

#
