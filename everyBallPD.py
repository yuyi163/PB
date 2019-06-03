import pandas as pd
import numpy as np

# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv("powerballData/sampleData.csv")
df_copy = df.copy()  # 留住原始的df
# print(df)
current_index = 1
next_index = current_index - 1
# print(df.columns.values)

current_data = df.ix[df.index == current_index, 0:]

print("==========================<<<<<<<<<<<<<<<<<<<<上一期>>>>>>>>>>>>>>>>>>>>=============================")
print(current_data)

if next_index >= 0:
    # next_data = df.iloc[next_index, 0: 6]
    next_data = df.ix[df.index == next_index, 0:]
    print("==========================<<<<<<<<<<<<<<<<<<<<下一期>>>>>>>>>>>>>>>>>>>>=============================")
    print(next_data)


for column in df.columns.values[1:]:
    print("<<<<<<<<<<<<<<<<<<<<"+column+">>>>>>>>>>>>>>>>>>>>")
    current_b = df.loc[current_index, column]
    next_b = df.loc[current_index - 1, column]
    # print("<<<<<<<<<<<<<<<<<<<<某期某球的值、下一期该球的值>>>>>>>>>>>>>>>>>>>>")
    # print(type(current_b), current_b, type(next_b), next_b)

    df_same = df.loc[df[column] == current_b].copy()
    # print("==========================<<<<<<<<<<<<<<<<<<<<与某期B1球的值相同的期数>>>>>>>>>>>>>>>>>>>>=============================")
    # print(df_same)
    # print("==========================<<<<<<<<<<<<<<<<<<<<下一期统计>>>>>>>>>>>>>>>>>>>>=============================")
    df_next = df.loc[df.index.isin(df_same.index.values - 1)]
    # print(df_next)
    df_count = df_next.groupby(by=[column], as_index=False)['time'].count()
    df_count_sorted = df_count.sort_values(by=['time'], ascending=False)

    print(df_count_sorted[:5])
