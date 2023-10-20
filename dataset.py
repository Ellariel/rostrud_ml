from tqdm import tqdm
from ast import literal_eval
from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import pickle, os
import hashlib
from scipy import stats
from cycler import cycler
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from rostrud_ml.process.adding_tables_psycopg import AddingDataPsycopg
#import warnings
#warnings.filterwarnings('ignore')

base_dir = './'
cached_dir = os.path.join(base_dir, 'cached')
os.makedirs(cached_dir, exist_ok=True)

region_types = pd.read_csv(os.path.join(base_dir, 'regions_types.csv'))[['region_code', 'forest', 'auto', 'vpk', 'border']]
_ren = {}
for c in region_types.columns:
  if 'region_' not in c:
    _ren.update({c : 'region_'+c})
region_types.rename(columns=_ren, inplace=True)

def save_pickle(df, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(df, f)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def localize(x):
  try:
    return pd.to_datetime(x).tz_localize(None)
  except:
    return np.nan

def _query(q, ignore_cache=False): # table_name='curricula_vitae', shema_name='project_trudvsem', +table_name+shema_name)
    file_name = os.path.join(cached_dir, hashlib.md5(q.encode()).hexdigest()+'.pickle')
    print(file_name)
    df = None
    if os.path.exists(file_name) and not ignore_cache:
        df = load_pickle(file_name)
        print(f'cached query loaded: {len(df)}')
    else:
        db = AddingDataPsycopg()
        #df = db.get_table_as_df(q, table_name, shema_name)
        df = db.query_exec_as_df(q)
        db.conn.close()
        del db
        if 'date_publish' in df:
          df['date_publish'] = df['date_publish'].apply(localize)
        if 'date_creation' in df:
            df['date_creation'] = df['date_creation'].apply(localize)
        save_pickle(df, file_name)
        print(f'query executed: {len(df)}')
    return df

def _rank(x):
  _r = (x.value_counts(dropna=False) / len(x)).to_dict()
  return x.apply(lambda x: _r[x] if x in _r else np.nan)

def _replace(x):
  return str(x).replace('\\', '').replace('/', '')

_check_length = {
    #'add_certificates': 3,
    'add_certificates_modified': 3,
    'skills': 10,
    'additional_skills': 10,
    #'other_info': 10,
    'other_info_modified': 10,
    #'achievements': 10,
    #'achievements_modified': 10,
    #'company_name': 2,
    #'job_title': 3,
    'workexp': 150,
}

allowed_cols = ['id_cv', 'industry_code', 'profession_name',
       'add_certificates_modified', 'additional_skills',
       'birthday', 'business_trips', 'busy_type',
       'date_publish', 'date_modify_inner_info', 'drive_licences',
       'education_type', 'experience', 'gender',
       'inner_info_status', 'locality', 'other_info_modified',
       'position_name', 'relocation',
       'retraining_capability', 'salary', 'schedule_type', 'skills',

       'achievements_modified', 'company_name', 'date_from', 'date_to', 'job_title', 'demands',

       'region_code', 'region_name', 'accomodation_accessibility', 'attraction_region', 'economic_growth',
       'kindergarten_accessibility', 'medium_salary_difference', 'price_level', 'unemployment_level',

       'region_forest', 'region_auto', 'region_vpk', 'region_border',

       'completeness_score', 'completeness_rank', 'responses',]

curricula_vitae_cols = "id_cv, add_certificates, add_certificates_modified, additional_skills, birthday, business_trips, busy_type, date_creation, date_modify_inner_info, date_publish, drive_licences, education_type, experience, gender, industry_code, inner_info_status, locality, md5_hash, nark_certificate, nark_inspection_status, other_info, other_info_modified, position_name, profession_code, region_code, relocation, retraining_capability, salary, schedule_type, skills"
curricula_vitae_cols = ['cv.'+i for i in curricula_vitae_cols.split(', ') if i in allowed_cols]
curricula_vitae_cols = ', '.join(curricula_vitae_cols)

regions_cols = "region_code, region_name, accomodation_accessibility, attraction_region, economic_growth, kindergarten_accessibility, medium_salary_difference, price_level, unemployment_level"
regions_cols = ['project_trudvsem.regions.'+i for i in regions_cols.split(', ') if i in allowed_cols]
regions_cols = ', '.join(regions_cols)

workexp_cols = "achievements, achievements_modified, company_name, date_from, date_to, demands, job_title, md5_hash, type"
workexp_cols = ['exp.'+i for i in workexp_cols.split(', ') if i in allowed_cols]
workexp_cols_agg = 'CONCAT(\'{\', ' + ', \',\',  '.join(['\'\"'+i.split('.')[-1]+'\":\"\', COALESCE(REPLACE('+i+'::text, \'\"\', \'\'), \'\'), \'\"\'' for i in workexp_cols]) + ', \'}\') AS workexp'

conditions = [['01-01-2022', '01-03-2022'],
              ['01-05-2022', '01-07-2022'],
              ['01-10-2022', '01-11-2022'],
              ['01-01-2023', '01-03-2023'],
              ['01-04-2020', '01-06-2020']]
# conditions = ' OR '.join([f"(date_publish >= \'{i[0]}\' AND date_publish < \'{i[1]}\')" for i in conditions])
#LIMIT 3

queries = [f"""WITH cv AS (SELECT * FROM project_trudvsem.curricula_vitae
                                     WHERE (date_publish >= \'{q[0]}\' AND date_publish < \'{q[1]}\')
                                     )
                SELECT {curricula_vitae_cols}, {regions_cols}, responses, project_trudvsem.professions.profession_name AS profession_name, CONCAT('[', STRING_AGG(workexp, ','), ']') AS workexp
                FROM cv
                  LEFT JOIN (SELECT id_cv, {workexp_cols_agg}
                             FROM project_trudvsem.workexp AS exp
                             WHERE id_cv IN (SELECT id_cv FROM cv)) AS exp
                       ON cv.id_cv = exp.id_cv
    LEFT JOIN project_trudvsem.professions ON cv.profession_code = project_trudvsem.professions.profession_code
      LEFT JOIN project_trudvsem.regions ON cv.region_code = project_trudvsem.regions.region_code
                LEFT JOIN (SELECT id_candidate, COUNT(id_response) as responses
                           FROM project_trudvsem.responses
                           GROUP BY id_candidate) AS rsp
                     ON cv.id_candidate = rsp.id_candidate
GROUP BY {curricula_vitae_cols}, {regions_cols}, responses, project_trudvsem.professions.profession_name""" for q in conditions]

chunk_size = 10000
for i, q in enumerate(queries):
  sample = _query(q)
  print(f'full sample: {len(sample)}')

  for idx in tqdm(range(0, sample.shape[0], chunk_size)):
      sample.loc[sample.iloc[idx:idx + chunk_size].index, 'workexp'] = sample.iloc[idx:idx + chunk_size]['workexp'].apply(lambda x: literal_eval(_replace(x)))
      sample.loc[sample.iloc[idx:idx + chunk_size].index, 'completeness_score'] = sample.iloc[idx:idx + chunk_size].apply(lambda x: len(''.join([str(x[i]) for i in _check_length.keys()])), axis=1)
      sample.loc[sample.iloc[idx:idx + chunk_size].index, 'completeness_full'] = sample.iloc[idx:idx + chunk_size].apply(lambda x: sum([len(str(x[i])) > _check_length[i] for i in _check_length.keys()]) >= len(_check_length), axis=1)

# sample['workexp'] = sample['workexp'].apply(lambda x: literal_eval(_replace(x)))
# sample['completeness_score'] = sample.apply(lambda x: len(''.join([str(x[i]) for i in _check_length.keys()])), axis=1)
# sample['completeness_full'] = sample.apply(lambda x: sum([len(str(x[i])) > _check_length[i] for i in _check_length.keys()]) >= len(_check_length), axis=1)

  sample['completeness_rank'] = _rank(sample['completeness_score']) #stats.percentileofscore(samples[i]['completeness_score'], samples[i]['completeness_score'])
  #save_pickle(sample, os.path.join(base_dir, f'sample{i}_full.pickle'))
  sample = sample[sample['completeness_full']]
  print(f'complete sample: {len(sample)}')
  sample = sample.set_index('region_code').join(region_types.set_index('region_code'))
  if 'date_modify_inner_info' in sample:
        sample['date_modify_inner_info'] = sample['date_modify_inner_info'].apply(localize)
  save_pickle(sample, os.path.join(base_dir, f'sample{i}_complete.pickle'))
  sample = sample.sort_values('date_modify_inner_info', ascending=False).drop_duplicates('id_cv')
  print(f'noduplicates sample: {len(sample)}')
  save_pickle(sample, os.path.join(base_dir, f'sample{i}_noduplicates.pickle'))
  del sample