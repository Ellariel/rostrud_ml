
import pandas as pd
import pickle
import yaml
from gensim.models import Word2Vec
from rostrud_ml.process.adding_tables_psycopg import AddingDataPsycopg
from rostrud_ml.matching.preparation.prepare_data import PrepareData
with open('rostrud_ml/matching/preparation/russian_stopwords_and_demands_50.yml') as file:
    russian_stopwords_and_demands = yaml.full_load(file)
with open('rostrud_ml/matching/preparation/russian_stopwords_and_respons.yml') as file:
    russian_stopwords_and_respons = yaml.full_load(file)
w2v_model = Word2Vec.load("w2v_model_retrain.model")

"""Модуль, который предназначен для реализации процедуры обработки
нового набора данных для матчинга резюме и вакансий и
дообучения модели"""


"""Создаем таблицы лемм вакансий и резюме для матчинга
(Для первой загрузки записей в таблицы для матчинга и обучения модели)"""
db = AddingDataPsycopg()
db.create_table('for_matching_cv', 'project_trudvsem')
db.create_table('for_matching_vac', 'project_trudvsem')
id_cv_list = []
id_vac_list = []

"""Получаем новые записи, которых нет в таблицах для матчинга(шаг для первичной или вторичной загрузки)"""
db = AddingDataPsycopg()
df_cv = db.get_table_as_df_join('curricula_vitae', 'get_df_join')
df_vac = db.get_table_as_df_join('vacancies', 'get_df_join')

"""Шаг для повторной загрузки лемм записей в таблицы для матчинга(если загрузка первичная пропускаем)"""
df = db.get_table_as_df('identifier', 'for_matching_cv', 'project_trudvsem')
id_cv_list = df['identifier'].tolist()
df = db.get_table_as_df('identifier', 'for_matching_vac', 'project_trudvsem')
id_vac_list = df['identifier'].tolist()

"""Подготавливаем записи для загрузки в таблицы для матчинга(шаг для первичной или вторичной загрузки)"""
prepare_cv = PrepareData(russian_stopwords_and_demands, w2v_model)
prepare_vac = PrepareData(russian_stopwords_and_respons, w2v_model)
df_cv = prepare_cv.prepare_df(df_cv, id_cv_list, 'curricula_vitae')
df_vac = prepare_vac.prepare_df(df_vac, id_vac_list, 'vacancies')

"""Загружаем новые записи в таблицы для матчинга(шаг для первичной или вторичной загрузки)"""
db.write_to_sql(df_cv, 'for_matching_cv', 'project_trudvsem')
db.write_to_sql(df_vac, 'for_matching_vac', 'project_trudvsem')

"""Удаляем не актуальные записи, которых нет в исходных таблицах резюме и вакансий(если загрузка первичная пропускаем)"""
db.delete_strings('identifier', 'id_cv', 'for_matching_cv', 'curricula_vitae', 'project_trudvsem')
db.delete_strings('identifier', 'identifier', 'for_matching_vac', 'vacancies', 'project_trudvsem')
db.conn.close()

"""Обучение модели (для первичной загрузки)"""
text_lemmas = prepare_vac.lemmas_to_list(df_cv, df_vac)
prepare_vac.train_model(text_lemmas)

"""Проверяем количество обновленных записей и дообучаем модель при условии большого объема новых записей(если загрузка первичная пропускаем)"""
print('Строк в таблице резюме ' + str(len(id_cv_list)) + ' добавлено строк ' + str(df_cv.shape[0]))
print('Строк в таблице вакансий ' + str(len(id_vac_list)) + ' добавлено строк ' + str(df_vac.shape[0]))
if (len(id_cv_list) + len(id_vac_list)) == 0:
    print('Дообучение модели не требуется')
else:
    difference = ((df_cv.shape[0] + df_vac.shape[0]) / (len(id_cv_list) + len(id_vac_list)))
    if difference <= 0.4:
        print('Дообучение модели не требуется:', difference)
    else:
        text_lemmas = prepare_vac.lemmas_to_list(df_cv, df_vac)
        prepare_vac.retrain_model(text_lemmas)
#
