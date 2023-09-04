
import pandas as pd
from rostrud_ml.process.adding_tables_psycopg_rostrud import *


"""Модуль, который инициализирует класс
MatchingDataCV предназначен для реализации процедуры
подбора вакансий для резюме в одном регионе"""


class MatchingData:
    """Класс MatchingDataCV предназначен для реализации процедуры
    подбора вакансий для резюме в одном регионе"""
    def __init__(self, counter, similarity):
        self.cos_similarity_w2v_combo = {}
        self.counter = counter
        self.similarity = similarity
     
    def trudvsem_profile_w2v_to_df_cv(self, id_cv: str):
        """Функция выводит результат матчинга в df"""
        cv_text = cv_for_search(id_cv=id_cv)
        vac_text = vac_candidates_one_region(cv_text.iloc[0, 8])
        id_list, recommendations  = self.recommendations_list(cv_text, vac_text, self.similarity.cosine_similarity_w2v_words_features_cv)
        df_cv = cv_profile(id_cv)
        df_vac = vac_candidates_profiles(id_list, recommendations, 'vacancies')
        return df_cv, df_vac
    
    def trudvsem_profile_w2v_to_df_vac(self, id_vac: str):
        """Функция выодит результат матчинга в df"""
        vac_text = vac_for_search(id_=id_vac)
        cv_text = cv_candidates_one_region(vac_text.iloc[0, 7])
        id_list, recommendations = self.recommendations_list(vac_text, cv_text, self.similarity.cosine_similarity_w2v_words_features_vac)
        df_vac = vac_profile(id_vac)
        df_cv = cv_candidates_profiles(id_list, recommendations, 'curricula_vitae')
        return df_vac, df_cv
    
    def recommendations_list(self, text_in: str, text_out: str, cosine_similarity_w2v):
        for i in range(text_out.shape[0]):
            try:
                cosine_similarity_ma = cosine_similarity_w2v(text_in, text_out, i)
                if cosine_similarity_ma > 0.5:
                    self.cos_similarity_w2v_combo[text_out.iloc[i, 1]] = cosine_similarity_ma
            except KeyError:
                continue
            except ValueError:
                continue
        recommendations = dict(sorted(self.cos_similarity_w2v_combo.items(), key=lambda item: item[1], reverse=True)[:self.counter])
        id_list = ','.join([("'" + str(id) + "'")for id in list(recommendations.keys())])
        return id_list, recommendations

    def trudvsem_profile_w2v_from_dict_features_(self, cv_text: pd.DataFrame):
        """Функция выводит результат матчинга
        (для росруда )"""
        if cv_text.iloc[0, 7] == 0:
            vac_text = vac_candidates_one_region(cv_text.iloc[0, 8])
        else: vac_text = vac_candidates_regions()
        for i in range(vac_text.shape[0]):
            try:
                cosine_similarity_ma = self.similarity.cosine_similarity_w2v_words_features_cv(cv_text, vac_text, i)
                if cosine_similarity_ma > 0.5:
                    self.cos_similarity_w2v_combo[vac_text.iloc[i, 1]] = cosine_similarity_ma
            except KeyError:
                continue
            except ValueError:
                continue
        recommendations = dict(sorted(self.cos_similarity_w2v_combo.items(), key=lambda item: item[1], reverse=True)[:self.counter])
        if len(recommendations) == 0:
            print('Подходящих вакансий не найдено, попробуйте выбрать другой регион, или более детально описать Ваш опыт работы')
        else: 
            id_list = ','.join([("'" + str(id) + "'")for id in list(recommendations.keys())])
            df_vac = vac_candidates_profiles(id_list, recommendations, 'vacancies')
            return df_vac

#
