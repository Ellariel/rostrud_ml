
import os
import pandas as pd
import psycopg2
import tempfile
from rostrud_ml.utils.config import Config
from rostrud_ml.process.adding_tables_psycopg import AddingDataPsycopg

"""Модуль, который инициализирует класс AddingDataPsycopg,
предназначен для реализации процедуры выгрузки и добавления
и удаления для нового набора данных"""

db = AddingDataPsycopg()
            
def delete_duplicates(table_name: str, schema: str):
    """Функция, которая удаляет старые версии записи из таблицы"""
    id_list = Config(os.path.join('.', 'rostrud_ml/utils/all_tables_names.yml')).get_config('delete_duplicates')
    id_ = id_list[table_name]
    with db.conn.cursor() as cursor:
        cursor.execute(f"""CREATE TEMPORARY TABLE t_temp_data
        AS (SELECT {id_}, MAX(date_last_updated) AS date_last_updated
        FROM {schema}.{table_name}
        GROUP BY {id_});""")
        cursor.execute("COMMIT")
        cursor.execute(f"""DELETE FROM {schema}.{table_name}
        WHERE NOT EXISTS (SELECT t_temp_data.{id_}, t_temp_data.date_last_updated
        FROM t_temp_data
        WHERE t_temp_data.{id_} = {schema}.{table_name}.{id_} 
        AND t_temp_data.date_last_updated = {schema}.{table_name}.date_last_updated)""")
        cursor.execute("COMMIT")
        print('Строки удалены')
            
def update_inactivation(schema: str, table_name: str, date: str, hashes: str):
    """Функция, которая присваивает положительный неактивный статус записи и новую дату инактивации"""
    with db.conn.cursor() as cursor:
        cursor.execute(f"""UPDATE {schema}.{table_name}
        SET date_inactivation = '{date}', inactive = 1
        WHERE inactive = 0 AND md5_hash IN ({hashes});""")
        cursor.execute("COMMIT")
            
def update_inactivation_new(schema: str, table_name: str, date: str, hashes: str):
    """Функция, которая присваивает положительный неактивный статус записи и новую дату инактивации"""
    with db.conn.cursor() as cursor:
        cursor.execute(f"""CREATE TEMPORARY TABLE temp_data
        AS (SELECT md5_hash
        FROM {schema}.{table_name}
        WHERE inactive = 0 AND md5_hash IN ({hashes})
        GROUP BY md5_hash);""")
        cursor.execute("COMMIT")
        cursor.execute(f"""UPDATE {schema}.{table_name}
        SET date_inactivation = '{date}', inactive = 1
        WHERE EXISTS (SELECT temp_data.md5_hash
        FROM temp_data
        WHERE {schema}.{table_name}.md5_hash = temp_data.md5_hash);""")
        cursor.execute("COMMIT")

def fix_error_inactivation(schema: str, table_name: str, active_hashes: str):
    """Функция, которая по результатам проверки (хеш есть в последней выгрузке, но ранее был присвоен статус 
    "неактивный") возвращает записи нулевой неактивный статус и удаляет дату инактивации"""
    with db.conn.cursor() as cursor:
        cursor.execute(f"""UPDATE {schema}.{table_name}
        SET date_inactivation = NULL, inactive = 0
        WHERE inactive = 1 AND md5_hash IN ({active_hashes});""")
        cursor.execute("COMMIT")
            
def fix_error_inactivation_new(schema: str, table_name: str, active_hashes: str):
    """Функция, которая по результатам проверки (хеш есть в последней выгрузке, но ранее был присвоен статус 
    "неактивный") возвращает записи нулевой неактивный статус и удаляет дату инактивации"""
    with db.conn.cursor() as cursor:
        cursor.execute(f"""CREATE TEMPORARY TABLE temp_data_fix
        AS (SELECT md5_hash
        FROM {schema}.{table_name}
        WHERE inactive = 1 AND md5_hash IN ({active_hashes})
        GROUP BY md5_hash);""")
        cursor.execute("COMMIT")
        cursor.execute(f"""UPDATE {schema}.{table_name}
        SET date_inactivation = NULL, inactive = 0
        WHERE EXISTS (SELECT temp_data_fix.md5_hash
        FROM temp_data_fix
        WHERE {schema}.{table_name}.md5_hash = temp_data_fix.md5_hash);""")
        cursor.execute("COMMIT")

def get_inactive_hash_list(table_name: str, schema: str) -> list:
    """Функция, которая собирает хеш-суммы"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT md5_hash FROM {schema}.{table_name} 
                            WHERE inactive = 1) TO STDOUT CSV"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        #тк последняя строка заканчивается на \n последний по сплиту элемент - пустая строка, её не берём
        results = tmpfile.read().decode().split('\n')[:-1] 
    return results

def update_inact(df_hashes: pd.DataFrame, schema: str, table_name: str, date: str):
    """Функция, которая присваивает положительный неактивный статус записи и новую дату инактивации
    предположительно быстрее"""
    with db.conn.cursor() as cursor:
        cursor.execute(f"""CREATE TEMPORARY TABLE temp_hashes
                        (md5_hash TEXT PRIMARY KEY);""")
        cursor.execute("COMMIT")
    copy_sql = f"""
              COPY temp_hashes (md5_hash) FROM STDIN WITH CSV HEADER
              DELIMITER as ','
              """
    with tempfile.TemporaryFile() as temp:
        temp.write(df_hashes.to_csv(index=False).encode())
        temp.seek(0)
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, temp)
    db.conn.commit()     
    with db.conn.cursor() as cursor:
        cursor.execute(f"""UPDATE {schema}.{table_name}
        SET date_inactivation = '{date}', inactive = 1
        FROM temp_hashes
        WHERE inactive = 0 AND {schema}.{table_name}.md5_hash = temp_hashes.md5_hash;""")
        cursor.execute("COMMIT")
    print(datetime.datetime.now().time(), 'Добавили неактивные')
            
def fix_error_inact(df_hashes: pd.DataFrame, schema: str, table_name: str):
    """Функция, которая убирает неактивный статус записи и дату инактивации
    предположительно быстрее"""
    with db.conn.cursor() as cursor:
        cursor.execute(f"""CREATE TEMPORARY TABLE temp_check
                        (md5_hash TEXT PRIMARY KEY);""")
        cursor.execute("COMMIT")
    copy_sql = f"""
              COPY temp_check (md5_hash) FROM STDIN WITH CSV HEADER
              DELIMITER as ','
              """
    with tempfile.TemporaryFile() as temp:
        temp.write(df_hashes.to_csv(index=False).encode())
        temp.seek(0)
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, temp)
    db.conn.commit()
    with db.conn.cursor() as cursor:
        cursor.execute(f"""UPDATE {schema}.{table_name}
        SET date_inactivation = NULL, inactive = 0
        FROM temp_check
        WHERE {schema}.{table_name}.md5_hash = temp_check.md5_hash;""")
        cursor.execute("COMMIT")

"""Функции для подбора списка вакансий для резюме"""

def cv_for_search(id_cv: str) -> pd.DataFrame:
    """Функция, которая выгружает леммы и фичи из таблицы резюме
    (в функцию подаем ид резюме)"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.for_matching_cv.*, region_code
    FROM project_trudvsem.for_matching_cv
    LEFT JOIN project_trudvsem.curricula_vitae
    ON project_trudvsem.curricula_vitae.id_cv = project_trudvsem.for_matching_cv.identifier
    WHERE identifier = '{id_cv}' and project_trudvsem.curricula_vitae.inactive = 0) TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return text_for_matching(df)
    
def vac_candidates_one_region(region_code: str) -> pd.DataFrame:
    """Функция, которая выгружает леммы и фичи из таблицы вакансий,
    подбор в ОДНОМ РЕГИОНЕ (в функцию подаем код региона)"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.for_matching_vac.*, region
    FROM project_trudvsem.for_matching_vac
    LEFT JOIN project_trudvsem.vacancies
    ON project_trudvsem.vacancies.identifier = project_trudvsem.for_matching_vac.identifier
    WHERE region = '{str(region_code)}' and project_trudvsem.vacancies.inactive = 0) TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return text_for_matching(df)

def vac_candidates_regions(region_code: str) -> pd.DataFrame:
    """Функция, которая выгружает леммы и фичи из таблицы вакансий,
    подбор в СПИСКЕ РЕГИОНОВ (в функцию подаем список регионов:
    "'6100000000000', '3600000000000'")"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.for_matching_vac.*, region
    FROM project_trudvsem.for_matching_vac
    LEFT JOIN project_trudvsem.vacancies
    ON project_trudvsem.vacancies.identifier = project_trudvsem.for_matching_vac.identifier
    WHERE region IN ({regions_code}) and project_trudvsem.vacancies.inactive = 0) TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return text_for_matching(df)
    
def cv_profile(id_: str) -> pd.DataFrame:
    """Функция, которая выгружает профайл из таблицы резюме
    (в функцию подаем ид резюме)"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.curricula_vitae.id_cv,  position_name,
    STRING_AGG(demands, ',') AS demands, industry_code, salary, experience, region_code
    FROM project_trudvsem.curricula_vitae
    LEFT JOIN project_trudvsem.workexp
    ON project_trudvsem.curricula_vitae.id_cv = project_trudvsem.workexp.id_cv
    WHERE project_trudvsem.curricula_vitae.id_cv = '{id_}' and project_trudvsem.curricula_vitae.inactive = 0
    GROUP BY project_trudvsem.curricula_vitae.id_cv, position_name, industry_code, salary, experience, region_code)
    TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return df
    
def vac_candidates_profiles(id_list: str, recommendations: dict, table_name: str) -> pd.DataFrame:
    """Функция, которая выгружает профайлы из таблицы вакансий"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT identifier,  title, responsibilities,
    industry, base_salary_min, base_salary_max,
    experience_requirements, region
    FROM project_trudvsem.vacancies
    WHERE identifier IN ({id_list}) and project_trudvsem.vacancies.inactive = 0)
    TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return candidates(df, recommendations, table_name)

"""Функции для подбора списка резюме для вакансии"""

def vac_for_search(id_: str) -> pd.DataFrame:
    """Функция, которая выгружает леммы и фичи из таблицы вакансий
    (в функцию подаем ид вакансии)"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.for_matching_vac.*, region
    FROM project_trudvsem.for_matching_vac
    LEFT JOIN project_trudvsem.vacancies
    ON project_trudvsem.vacancies.identifier = project_trudvsem.for_matching_vac.identifier
    WHERE project_trudvsem.vacancies.identifier = '{id_}' and project_trudvsem.vacancies.inactive = 0)
    TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return text_for_matching(df)
    
def cv_candidates_one_region(region_code: str) -> pd.DataFrame:
    """Функция, которая выгружает леммы и фичи из таблицы резюме,
    подбор в ОДНОМ РЕГИОНЕ (в функцию подаем код региона)"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.for_matching_cv.*, region_code
    FROM project_trudvsem.for_matching_cv
    LEFT JOIN project_trudvsem.curricula_vitae
    ON project_trudvsem.curricula_vitae.id_cv = project_trudvsem.for_matching_cv.identifier
    WHERE region_code = '{str(region_code)}' and project_trudvsem.curricula_vitae.inactive = 0)
    TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return text_for_matching(df)

def cv_candidates_regions(region_code: str) -> pd.DataFrame:
    """Функция, которая выгружает леммы и фичи из таблицы резюме,
    подбор в СПИСКЕ РЕГИОНОВ (в функцию подаем список регионов:
    "'6100000000000', '3600000000000'")"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.for_matching_cv.*, region_code
    FROM project_trudvsem.for_matching_cv
    LEFT JOIN project_trudvsem.curricula_vitae
    ON project_trudvsem.curricula_vitae.id_cv = project_trudvsem.for_matching_cv.identifier
    WHERE region_code IN ({regions_code}) and project_trudvsem.curricula_vitae.inactive = 0)
    TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return text_for_matching(df)

def vac_profile(id_: str) -> pd.DataFrame:
    """Функция, которая выгружает профайл из таблицы вакансий
    (в функцию подаем ид вакансии)"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT identifier,  title, responsibilities,
    industry, base_salary_min, base_salary_max,
    experience_requirements, region
    FROM project_trudvsem.vacancies
    WHERE project_trudvsem.vacancies.identifier = '{id_}' and project_trudvsem.vacancies.inactive = 0)
    TO STDOUT WITH CSV HEADER"""
        with  db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return df

def cv_candidates_profiles(id_list: str, recommendations: dict, table_name: str) -> pd.DataFrame:
    """Функция, которая выгружает профайлы из таблицы резюме"""
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = f"""COPY (SELECT project_trudvsem.curricula_vitae.id_cv AS identifier,  position_name,
    STRING_AGG(demands, ',') AS demands, industry_code, salary,
    experience, region_code
    FROM project_trudvsem.curricula_vitae
    LEFT JOIN project_trudvsem.workexp
    ON project_trudvsem.curricula_vitae.id_cv = project_trudvsem.workexp.id_cv
    WHERE project_trudvsem.curricula_vitae.id_cv IN ({id_list}) and project_trudvsem.curricula_vitae.inactive = 0
    GROUP BY project_trudvsem.curricula_vitae.id_cv, position_name,industry_code, salary,
    experience, region_code) TO STDOUT WITH CSV HEADER"""
        with db.conn.cursor() as cursor:
            cursor.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
    return candidates(df, recommendations, table_name)

def text_for_matching(df: pd.DataFrame):
    df['text_w2v'] = df['text_w2v'].apply(lambda x: x.split(','))
    return df

def candidates(df: pd.DataFrame, recommendations: dict, table_name: str)-> pd.DataFrame:
    subset_list = Config(os.path.join('.', 'rostrud_ml/utils/all_tables_names.yml')).get_config('subset_list')
    subset = subset_list[table_name]
    df['score'] = df.apply(lambda row: recommendations[row.identifier], axis=1)
    df = df.sort_values('score', ascending=False).drop_duplicates(subset=subset).reset_index(drop=True)
    return df

#
