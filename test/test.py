import os
import pandas as pd
import tensorflow as tf

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
print(zip_path)
print(csv_path)
print(_)
# path = r'C:\Users\pc\.keras\datasets\jena_climate_2009_2016_extracted\jena_climate_2009_2016.csv'
# print(path)
# # 读取 CSV 文件
# df = pd.read_csv(path)
# df = df[5::6]
# date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
# # 显示 DataFrame 的前几行
# print(df.head())
# print(date_time.head())
