import pandas as pd
import numpy as np
import pickle
import yaml
from gensim.models import Word2Vec
from rostrud_ml.matching.text_matching.cosine_similarity_w2v import CosineSimilarityW2v
from rostrud_ml.matching.text_matching.matching_data import MatchingData

"""Загружаем предобученную модель и  стандар скеллер для фич"""
with open('rostrud_ml/matching/preparation/russian_stopwords_and_demands_50.yml') as file:
    russian_stopwords_and_demands = yaml.full_load(file)
w2v_model = Word2Vec.load("w2v_model_retrain.model")
w2v_model.init_sims(replace = True)
scaler = pickle.load(open('rostrud_ml/matching/text_matching/scaler_short.pkl','rb'))

"""Подберем вакансии для резюме"""

dict_new = {"experience":30, "salary":320000, 
 "demands":"Встреча посетителей| поддержание комфорта и приятной атмосферы для гостей в зале| управление персоналом(обучение, составление графика работы, контроль рабочего процесса)| работа с документацией(отчёты инвентаризации)| разрешение конфликтных ситуаций",
 "relocation":0,
 "retraining_capability":0,
 "business_trips":0,
 "all_regions":0,
 "region_code":"7700000000000",
 "industry_code":"Finances",
 "profession_name":"Старший администратор",
 "position_name": None}

prepare_cv = PrepareData(russian_stopwords_and_demands)
df_cv = prepare_cv.final_prepare_cv_from_dict(dict_new)
similarity = CosineSimilarityW2v(w2v_model, scaler)
matching_data_cv = MatchingData(10, similarity)
matching_data_cv.trudvsem_profile_w2v_from_dict_features_(df_cv)
#
