import gzip
import shutil
from os import listdir
from rostrud_ml.process.parsing_xmls import *


# Получаем дату из имени файла
def get_date(url):
    return url.split('data-')[-1].split('T')[0] #use regexp to get the date of the file

# def get_date_year_ago(date_ymd):
#     year_ago = int(date_ymd.rsplit('-')[0]) - 1
#     return str(year_ago) + date_ymd[4:]

# Разархивируем файл в определённый путь
def unzip_cv(path):
    with gzip.open(path, 'rb') as f_in:
        with open(path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Выбрать парсер по переданному имени обновляемой таблицы
def get_parser(table_name, path=None):
    if table_name == 'cvs':
        return ParseCvs()
    if table_name == 'invitations':
        return ParseInvitations()
    if table_name == 'vacancies':
        return ParseVacancies()
    if table_name == 'responses':
        return ParseResponses()
    if table_name == 'organizations':
        return ParseOrganizations()
    if table_name == 'regions':
        return ParseRegions()
    if table_name == 'industries':
        return ParseIndustries()
    if table_name == 'professions':
        return ParseProfessions()
    if table_name == 'stat_companies':
        return ParseStatCompany()
    if table_name == 'stat_citizens':
        return ParseStatCitizens()
    return None
    
    
# Создать датафрейм из csv с типом данных 'object'
def read_csv_object(filepath):
    return pd.read_csv(filepath, dtype = 'object')

# Создать датафрейм из коллекции .csv файлов, тип данных 'object', сортировка по алфавиту, обнуляя индекс
def make_df(dirpath):
    filepaths = [dirpath + "/" + f for f in listdir(dirpath) if f.endswith('.csv')]
    return pd.concat(map(read_csv_object, filepaths), sort=True).reset_index(drop=True)

# Добавить кавычки при создании строки из списка
def to_str_wquotes(lst):
    return ', '.join(f"'{w}'" for w in lst)
