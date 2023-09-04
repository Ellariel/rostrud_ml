
import os
import pandas as pd
import datetime
import itertools
from pyaspeller import YandexSpeller
from pymystem3 import Mystem
from rostrud_ml.utils.config import Config
from gensim.models import Word2Vec

"""Модуль, который инициализирует класс PrepareData
предназначен для реализации процедуры обработки
нового набора данных для матчинга резюме и вакансий и для реализации процедуры обучения и дообучения
языковой модели для матчинга резюме и вакансий"""

class PrepareData:
    """Класс PrepareData предназначен для реализации процедуры
    обработки нового набора данных для матчинга резюме и вакансий"""
    
    def __init__(self, stopwords_on_demands, w2v_model):
        self.stopwords_on_demands = stopwords_on_demands
        self.w2v_model = w2v_model
        
    def prepare_df(self, df_c: pd.DataFrame, id_list: list, query_name: str) -> pd.DataFrame:
        """Функция, которая собирает датасет для загрузки в БД"""
        df_new = df_c.loc[~df_c.identifier.isin(id_list), :]
        self.variable_normalization(df_new, query_name)
        text_cols = ['demands', 'position_name', 'profession_name']
        for col in text_cols:
            self.lemmatization(df_new, col, self.stopwords_on_demands)
        text_cols = ['industry_code', 'profession_name', 'position_name', 'demands']
        df_new['text_w2v'] = df_new[text_cols].apply(
            lambda row: ','.join(row.values.astype(str)), axis=1)
        if query_name == 'curricula_vitae':
            df_new = df_new[['identifier', 'text_w2v', 'salary', 'experience_salary',
                             'btrips_salary_cv', 'btrips_relocation_cv', 'retraining_capability']].astype(str)
        elif query_name == 'vacancies':
            df_new = df_new[['identifier', 'text_w2v', 'salary', 'experience_salary', 
                             'accommodation_salary_vac', 'retraining_capability']].astype(str)
        return df_new
    
    def variable_normalization(self, df: pd.DataFrame, query_name: str) -> pd.DataFrame:
        """Функция, которая нормализует данные в таблице"""
        df.reset_index(drop=True, inplace=True)
        cols_list = Config(os.path.join('.', 'rostrud_ml/utils/all_tables_names.yml')).get_config('variable_normalization')
        int_cols = cols_list[query_name]
        df[int_cols] = df[int_cols].fillna(0).astype(int)
        df.loc[df.experience < 0, 'experience'] = 0
        df.loc[df.experience > 10, 'experience'] = cols_list['maximum_experience']
        df.loc[df.salary < 0, 'salary'] = 0
        df.loc[df.salary > 100000, 'salary'] = cols_list['maximum_salary']
        df['experience_salary'] = df['experience'] * df['salary']
        if query_name == 'curricula_vitae':
            df['btrips_salary_cv'] = df['business_trips'] * df['salary']
            df['btrips_relocation_cv'] = df['business_trips'] * df['relocation']
        elif query_name == 'vacancies':
            df['accommodation_salary_vac'] = df['accommodation_capability'] * df['salary']
        text_cols = ['industry_code', 'profession_name', 'position_name', 'demands']
        df[text_cols] = df[text_cols].fillna("")
        return df
    
    def chunks(self, iterable: list, n: int = 1000):
        """ Разбиение на чанки"""
        it = iter(iterable)
        while item := list(itertools.islice(it, n)):
            yield item
    
    def process_chunk(self, chunk: list = [], delimiter='brrrr'):
        """Функция, которая лемматизирует текстовые записи
        (ВЛОЖЕННАЯ ФУНКЦИЯ В lemmatization)"""
        self.mystem = Mystem()
        self.speller = YandexSpeller()
        alltexts = f' {delimiter} '.join([str(i) for i in chunk])
        tagged = []
        processed = self.mystem.analyze(alltexts)
        doc = []
        for w in processed:
            if w['text'] == delimiter:
                tagged.append(','.join(doc))
                doc = []
            else:
                lemma = None
                try:
                    lemma = w["analysis"][0]["lex"].lower().strip()
                    if w["analysis"][0]["qual"] == 'bastard':
                        fixed = lemma
                        fixed = self.speller.spelled(lemma)
                        fixed = fixed.split()
                        for z in fixed:
                            v = self.mystem.analyze(z)
                            lemma = v[0]["analysis"][0]["lex"]
                except IndexError:
                    pass
                except KeyError:
                    pass
                lemma = self.check_stopwords(lemma)
                if lemma is not None:
                    doc.append(lemma)
        tagged.append(','.join(doc))
        return tagged
    
    def check_stopwords(self, lemma):
        """Функция, которая проверяет леммы
        на наличие в стоп-словаре"""
        if lemma not in self.stopwords_on_demands:
            return lemma
    
    def lemmatization(self, df_new: pd.DataFrame, col: str, stopwords_list: list) -> pd.DataFrame:
        """Функция, которая лемматизирует список и
        чистит леммы от стоп-слов (ВЛОЖЕННАЯ ФУНКЦИЯ В prepare_df)"""
        df_new[col] = self.test_spell(df_new[col])
        return df_new
    
    def test_spell(self, df: pd.Series) -> list:
        """Функция, которая лемматизирует текстовые записи
        (ВЛОЖЕННАЯ ФУНКЦИЯ В lemmatization)"""
        chunk_list = list(itertools.chain(*[self.process_chunk(chunk) for chunk in self.chunks(df.tolist())]))
        return chunk_list
    
    def lemmas_to_list(self, df_cv: pd.DataFrame, df_vac: pd.DataFrame) -> list:
        """Функция, которая подготавливает набор лемм для обучения модели"""
        df_cv['text_w2v'] = df_cv['text_w2v'].apply(lambda x: x.split(','))
        df_cv['text_w2v'] = [[val.strip() for val in sublist] for sublist in df_cv['text_w2v'].values]
        df_vac['text_w2v'] = df_vac['text_w2v'].apply(lambda x: x.split(','))
        df_vac['text_w2v'] = [[val.strip() for val in sublist] for sublist in df_vac['text_w2v'].values]
        text_lemmas = pd.concat([df_cv['text_w2v'], df_vac['text_w2v']], ignore_index=True).tolist()
        return text_lemmas
    
    def train_model(self, text_lemmas):
        """Функция, которая обучает новую модель на наборе данных
        (можно подавать колонку df или список списков)"""
        w2v_model = Word2Vec(
            sentences=text_lemmas, min_count=2, window=3, size=300,
            negative=3, alpha=0.03, min_alpha=0.0007, sample=6e-5, sg=0)
        #w2v_model.wv.vocab
        w2v_model.save("w2v_model.model")
        w2v_model.init_sims(replace=True)
        return w2v_model
    
    def retrain_model(self, text_lemmas):
        """Функция, которая дообучает  модель на наборе новых данных
        (можно подавать колонку df или список списков)
        перезаписывает модель в файл"""
        w2v_model_retrain = self.w2v_model
        #w2v_model_retrain.build_vocab(text_lemmas, update=True)
        w2v_model_retrain.train(text_lemmas, total_examples=len(text_lemmas), epochs=10)
        now = datetime.datetime.now()
        w2v_model_retrain.save(f'w2v_model_retrain_{now.year}_{now.month}_{now.day}.model')
        w2v_model_retrain.init_sims(replace=True)
        w2v_model_retrain.save(f'w2v_model_retrain_l2_{now.year}_{now.month}_{now.day}.model')
        print('Модель успешно дообучена')
        
    def final_prepare_cv_from_dict(self, dict_new: dict) -> pd.DataFrame:
        """Функция, которая собирает датасет одного резюме для подбора вакансий(для роструда)"""
        df_new = pd.DataFrame([dict_new], columns=dict_new.keys())
        self.variable_normalization(df_new, 'curricula_vitae')
        text_cols = ['demands', 'position_name', 'profession_name']
        for col in text_cols:
            self.lemmatization(df_new, col, self.stopwords_on_demands)
        text_cols = ['industry_code', 'profession_name', 'position_name', 'demands']
        df_new['text_w2v'] = df_new[text_cols].apply(
            lambda row: ','.join(row.values.astype(str)), axis=1)
        df_new['text_w2v'] = df_new['text_w2v'].apply(lambda x: x.split(','))
        df_new = df_new[['text_w2v', 'salary', 'experience_salary',
        'btrips_salary_cv', 'btrips_relocation_cv', 'retraining_capability', 'region_code', 'all_regions']]
        return df_new

#
