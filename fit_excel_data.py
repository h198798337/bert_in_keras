#!/usr/bin/env python3
# Author: fanqiang
# create date: 2021/2/8
# Content: 
# desc:
import re
import pandas as pd

v_excel_path = '/data/home/fanqiang/bert_in_keras/resource/7、monki-衣服500条.xlsx'
v_fit_data_file_path = '/data/home/fanqiang/bert_in_keras/resource/尺寸打分差异.xlsx'

v_fit_excel = pd.read_excel(v_fit_data_file_path, sheet_name='Sheet3')

excel_data = pd.read_excel(v_excel_path, dtype={'id': str, 'cid': int})

for i, content in enumerate(v_fit_excel.content):
    hit_flag = False
    for j, ex_content in enumerate(excel_data.content):
        ex_content = re.sub('[\n|]+', ' ', ex_content)
        if ex_content.strip() == content.strip():
            hit_flag = True
            excel_data.尺寸[j] = int(v_fit_excel.dim_2[i])
    if not hit_flag:
        print('not hit', v_fit_excel.dim_2[i], content)

excel_data.to_excel('/data/home/fanqiang/bert_in_keras/resource/7、monki-衣服500条-fix.xlsx', index=False)