
import pickle
from gensim.models import Word2Vec
from rostrud_ml.matching.text_matching.cosine_similarity_w2v import CosineSimilarityW2v
from rostrud_ml.matching.text_matching.matching_data import MatchingData

"""Модуль, который предназначен для реализации процедуры обработки
нового набора данных для матчинга резюме и вакансий и
дообучения модели"""

"""Загружаем предобученную модель и  стандар скеллер для фич"""
w2v_model = Word2Vec.load('w2v_model_retrain.model')
w2v_model.init_sims(replace = True)
scaler = pickle.load(open('rostrud_ml/matching/text_matching/scaler_short.pkl','rb'))
similarity = CosineSimilarityW2v(w2v_model, scaler)

"""Подберем вакансии для резюме, указываем id резюме
и количество вакансий для вывода
ВАРИАНТ чистовой  с выводом строчек резюме и вакансий"""
matching = MatchingData(10, similarity)
df_cv, df_vac = matching.trudvsem_profile_w2v_to_df_cv('00624890-fc26-11eb-81cd-6db06c9eaf56')
print(df_cv)
print(df_vac) 

"""Подберем резюме для вакансии, указываем id вакансии
и количество резюме для вывода
ВАРИАНТ чистовой с выводом строчек резюме и вакансий"""
df_vac, df_cv = matching.trudvsem_profile_w2v_to_df_vac('00118432-3152-11ec-a785-bf2cfe8c828d')
print(df_vac)
print(df_cv) 

#
