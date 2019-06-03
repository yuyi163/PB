import pandas as pd
import numpy as np

# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv("powerballData/sampleData.csv")
df_copy = df.copy()
print(df)
current_index = 1
next_index = current_index - 1
current_data = df.iloc[current_index, 0: 6]
current_data1 = df.ix[df.index == current_index, 0:6]

print("==========================<<<<<<<<<<<<<<<<<<<<上一期>>>>>>>>>>>>>>>>>>>>=============================")
print(current_data1)

if next_index >= 0:
    # next_data = df.iloc[next_index, 0: 6]
    next_data = df.ix[df.index == next_index, 0:6]
    print("==========================<<<<<<<<<<<<<<<<<<<<下一期>>>>>>>>>>>>>>>>>>>>=============================")
    print(next_data)
df['cost1'] = df['b1'].map(lambda x: np.abs(x - current_data['b1']))
df['cost2'] = df['b2'].map(lambda x: np.abs(x - current_data['b2']))
df['cost3'] = df['b3'].map(lambda x: np.abs(x - current_data['b3']))
df['cost4'] = df['b4'].map(lambda x: np.abs(x - current_data['b4']))
df['cost5'] = df['b5'].map(lambda x: np.abs(x - current_data['b5']))
# df['cost6'] = df['powerBall'].map(lambda x: np.abs(x - data_latest['powerBall']))
df['cost'] = df['cost1'] + df['cost2'] + df['cost3'] + df['cost4'] + df['cost5']

df['same1'] = df['b1'].map(lambda x: 1 if x in current_data.values else 0)
df['same2'] = df['b2'].map(lambda x: 1 if x in current_data.values else 0)
df['same3'] = df['b3'].map(lambda x: 1 if x in current_data.values else 0)
df['same4'] = df['b4'].map(lambda x: 1 if x in current_data.values else 0)
df['same5'] = df['b5'].map(lambda x: 1 if x in current_data.values else 0)
# df['same6'] = df['powerBall'].map(lambda x: 1 if x in data_latest.values else 0)
df['same'] = df['same1'] + df['same2'] + df['same3'] + df['same4'] + df['same5']
# print(df)
df_cost = df.loc[df['cost'] < 10].copy()
df_cost = df_cost[['time', 'b1', 'b2', 'b3', 'b4', 'b5', 'powerBall', 'cost', 'same']]

df_same = df.loc[df['same'] > 2].copy()
df_same = df_same[['time', 'b1', 'b2', 'b3', 'b4', 'b5', 'powerBall', 'cost', 'same']]

print("==========================<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>=============================")
print(df_cost)
print(df_same)
print("==========================<<<<<<<<<<<<<<<<<<<<预测>>>>>>>>>>>>>>>>>>>>=============================")
print(df.loc[df.index.isin(df_cost.index.values - 1)])
print(df.loc[df.index.isin(df_same.index.values - 1)])

