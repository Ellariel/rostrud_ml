# -*- coding: utf-8 -*-
"""pipline_clone.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1XvrKYxexqwHqBsGbCy8jzROTluMNFZ0V

# Работа модели для матчинга резюме и вакансий портала "Работа в России":
"""
#Установим библиотеку, разработанную ЦПУР
#jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=9999 --NotebookApp.port_retries=0
#!git clone https://github.com/ellariel/rostrud_ml.git

"""После скачивания корневая папка должна быть переименована в rostrud_ml
## Предварительные действия
У этого скрипта есть несколько зависимостей, их можно поставить через pip. При запуске блока кода ниже всё поставится само.
Если в какой-то момент выполнения кода вы увидите ошибку, что вам не хватает некого питоновского пакета, вы можете дописать его название в новую строчку файла requirements.txt и снова запустить этот блок.
"""
#Установка окружения
#!pip install -r rostrud_ml/requirements.txt
"""В файле rostrud_ml/utils/all_tables_names.yml указать путь к директории, куда будут скачиваться и парситься файлы с резюме, вакансиями (сейчас там "workdir").
В файле rostrud-ml/utils/config_to_bd_example.yml указать ваши настройки для подключения к БД и поменять название файла на config_to_bd.yml
"""
import requests, os
from tqdm import tqdm
import pandas as pd

file_name = 'filelist_daily.csv'
monthly=False
remove_gz=True

base_url = 'https://opendata.trudvsem.ru/oda2Hialoephidohyie1oR6chaem1oN0quiephooleiWei1aiD/'
file_url = ['7710538364-cv/', '7710538364-professions/', '7710538364-regions/', '7710538364-industries/', '7710538364-vacancy/', '7710538364-invitation/', '7710538364-response/']

def retreive_filelist(monthly=True):
    # last file in each month
    def _getfileslist(url):
      r = requests.get(url)
      return ['data' + i.split('">data')[0] for i in r.text.split('<a href="data')[1:]]
    
    s = pd.DataFrame(_getfileslist(base_url + file_url[0])).rename(columns={0: 'cv'})
    s = s[s['cv'].apply(lambda x: pd.notna(x) and '.gz' in x)]
    s = s[s['cv'].apply(lambda x: int(x.split('data-')[1][:4]) >= 2019)]

    if monthly:
      s['month'] = s['cv'].apply(lambda x: x.split('data-')[1][:6])
      s = s.groupby('month').agg('last').reset_index()
      for l in file_url[1:]:
          c = l.split('-')[1]
          c = c[:len(c)-1]
          v = pd.DataFrame(pd.Series(_getfileslist(base_url + l)).rename(c))
          v = v[v[c].apply(lambda x: pd.notna(x) and '.gz' in x)]
          v = v[v[c].apply(lambda x: int(x.split('data-')[1][:4]) >= 2019)]
          v['month'] = v[c].apply(lambda x: x.split('data-')[1][:6])
          v = v.groupby('month').agg('last').reset_index()
          s = s.set_index('month').join(v.set_index('month')).reset_index()
    else:
    # all files
      s['date'] = s['cv'].apply(lambda x: x.split('data-')[1][:8])
      s = s.groupby('date').agg('last').reset_index()
      for l in file_url[1:]:
          c = l.split('-')[1]
          c = c[:len(c)-1]
          v = pd.DataFrame(pd.Series(_getfileslist(base_url + l)).rename(c))
          v = v[v[c].apply(lambda x: pd.notna(x) and '.gz' in x)]
          v = v[v[c].apply(lambda x: int(x.split('data-')[1][:4]) >= 2019)]
          v['date'] = v[c].apply(lambda x: x.split('data-')[1][:8])
          v = v.groupby('date').agg('last').reset_index()
          s = s.set_index('date').join(v.set_index('date')).reset_index()
    #s.to_csv('filtered_all.csv', index=False)
    s['added'] = False
    return s

"""## Загрузка в БД необходимых данных"""
filelist = None
if not os.path.exists(file_name):
    filelist = retreive_filelist(monthly=monthly)
    filelist.sort_values(by='month' if monthly else 'date', ascending=False, inplace=True)
    filelist.to_csv(file_name, index=False)
else:    
    filelist = pd.read_csv(file_name) #список ссылок полученный на прошлом этапе
print(f"file list of {len(filelist)} files")

#Установка модулей
from rostrud_ml.process.adding_tables_psycopg import AddingDataPsycopg
from rostrud_ml.process.renewal import Renewal

#Cоздание таблиц в 
print(f"open database and create tables...")
db = AddingDataPsycopg()
for table in ['curricula_vitae', 'workexp', 'vacancies', 'professions', 'invitations', 'responses']: # 'edu', 'addedu', 'industries', 'regions', 
    db.create_table(table, 'project_trudvsem') #'project_trudvsem' - название схемы в вашей БД, создать, если нету
db.conn.close()

"""Взять ссылки на файлы с данными с ftp поратала https://opendata.trudvsem.ru/oda2Hialoephidohyie1oR6chaem1oN0quiephooleiWei1aiD/:

Из директорий
https://opendata.trudvsem.ru/oda2Hialoephidohyie1oR6chaem1oN0quiephooleiWei1aiD/7710538364-cv/
https://opendata.trudvsem.ru/oda2Hialoephidohyie1oR6chaem1oN0quiephooleiWei1aiD/7710538364-vacancy/ нужно выбрать подходящие файлы (имеющие в наименовании дату) - один файл резюме и один вакансий.

Подобным образом выбрать ссылки для данных о профессиях, сферах занятости, регионах из соответствующих директорий:
https://opendata.trudvsem.ru/oda2Hialoephidohyie1oR6chaem1oN0quiephooleiWei1aiD/7710538364-professions/
https://opendata.trudvsem.ru/oda2Hialoephidohyie1oR6chaem1oN0quiephooleiWei1aiD/7710538364-industries/
https://opendata.trudvsem.ru/oda2Hialoephidohyie1oR6chaem1oN0quiephooleiWei1aiD/7710538364-regions/

Для каждого типа, вставив нужную ссылку, запустить следующие ячейки:
В примере: тип "резюме" и самый первый файл резюме 2018г. Необходимо поменять ссылку на наиболее актуальный файл.
"""
print(f"starting process...")
for idx, link in tqdm(filelist.iterrows(), total=len(filelist)):
  if link['added']:
      continue
    
  if idx == filelist.index[0]:
      try:
        print('\nprofessions')
        ren = Renewal('professions', base_url + file_url[1] + link['professions'])
        ren.download()
        ren.extract()
        ren.parse_update()
      except Exception as e:
        print(f'professions error: {e}')
      ren.delete(remove_gz=False)
      
#      try:
#        print('\nindustries')
#        ren = Renewal('industries', base_url + file_url[2] + link['regions'])
#        ren.download()
#        ren.extract()
#        ren.parse_update()
#      except Exception as e:
#        print(f'industries error: {e}')
#      ren.delete(remove_gz=False)

#      try:
#        print('\nregions')
#        ren = Renewal('regions', base_url + file_url[3] + link['industries'])
#        ren.download()
#        ren.extract()
#        ren.parse_update()
#      except Exception as e:
#        print(f'regions error: {e}')
#      ren.delete(remove_gz=False)

  if pd.notna(link['cv']):
    try:
      print('\ncurricula_vitae')
      ren = Renewal('curricula_vitae', base_url + file_url[0] + link['cv'])
      ren.download()
      ren.extract()
      ren.parse_update(default_tables=['curricula_vitae', 'workexp'])  #default_tables=['curricula_vitae', 'workexp', 'edu', 'addedu']
    except Exception as e:
      print(f'curricula_vitae error: {e}')
    ren.delete(remove_gz=remove_gz)

  if pd.notna(link['invitation']):
    try:
      print('\ninvitations')
      ren = Renewal('invitations', base_url + file_url[5] + link['invitation'])
      ren.download()
      ren.extract()
      ren.parse_update()
    except Exception as e:
      print(f'invitations error: {e}')
    ren.delete(remove_gz=remove_gz)
   
  if pd.notna(link['response']):
    try:
      print('\nresponses')
      ren = Renewal('responses', base_url + file_url[6] + link['response'])
      ren.download()
      ren.extract()
      ren.parse_update()
    except Exception as e:
      print(f'responses error: {e}')
    ren.delete(remove_gz=remove_gz)
  
  if pd.notna(link['vacancy']):
    try:
      print('\nvacancies')
      ren = Renewal('vacancies', base_url + file_url[4] + link['vacancy'])
      ren.download()
      ren.extract()
      ren.parse_update()
    except Exception as e:
      print(f'vacancies error: {e}')
    ren.delete(remove_gz=remove_gz)

  filelist.loc[idx, 'added'] = True
  filelist.to_csv(file_name, index=False)



