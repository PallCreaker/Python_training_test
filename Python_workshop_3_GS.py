import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import output_notebook,show
import matplotlib.pyplot as plt


#データフレームにファイルを読み込む
df = pd.read_csv('C:/Users/180256/Documents/Python_program/NHANES_sample_data.csv')

#不要な列の削除
drop_col = ['RIDRETH3', 'WTINT2YR', 'WTMEC2YR', 'DR1TCARB', 'DR1TTFAT', 'LBXGLU', 'BMXHT']
df=df.drop(drop_col, axis=1)

#列名の変更
df_new = df.rename(columns={'RIDAGEYR': 'Age', 'RIAGENDR': 'Gender', 'BMXWT': 'Body_weight', 'BMXBMI': 'BMI', 'LBXGH': 'HbA1c', 'LBDGLUSI':'Glucose'})

#Nullのある行を抜きだす
df_nun=df_new[df.isnull().any(1)]
#print(df_nun)

#新しい列（身長：Height）を追加
#BMIを算出する式「体重÷（身長(m)^2」から身長を算出
df_new['Height'] = np.sqrt(df_new['Body_weight']/df_new['BMI']) 
#print(df_new)

#中央値と分散の計算
#descrive関数を用いることで基本統計量を表示可能
result=df_new[['Age','Gender','Body_weight','BMI','HbA1c','Glucose']].describe()
#print(result)

#指定した条件で抽出
df_lim = df_new[(df_new['Age'] >= 20) 
                & (df_new['BMI'] <= 50) 
                & (df_new['BMI'] >= 18) 
                & (df_new['Glucose'] <= 16) 
                & (df_new['Glucose'] >= 4) 
                & (df_new['HbA1c'] <= 10) 
                & (df_new['HbA1c'] >= 4)]
#print(df_lim)

#指定した条件でカウント
df_man = (df_new['Gender'] == 1)
#print(df_man.sum())

df_temp = df_lim.rename(columns={'BMI': 'A', 'Body_weight': 'B', 'Glucose': 'C'})
df_temp_r = df_temp.reset_index(drop=True)
#print(df_temp_r)

#特定の行列を表示
#print(df_temp_r.iloc[34,:])
#print(df_temp_r.iloc[:,3])
#print(df_temp_r.iloc[57,4])
#print(df_temp_r.iloc[[50,100,250],[1,4]])
#print(df_temp_r.iloc[57,[2,3,5]])
#print(df_temp_r.iloc[128,2])

#3行ごとに抽出
#print(df_temp_r[['B','C']].iloc[df_temp_r.index%3==0])

#サブグループの作成
#男女別に新しいデータフレームを作成
df_male=df_lim[(df_lim['Gender']==1)]
df_female=df_lim[(df_lim['Gender']==2)]

age_1_M=df_male[(df_male['Age'] >= 20)
             & (df_male['Age'] < 40)]
age_1_F=df_female[(df_female['Age'] >= 20)
             & (df_female['Age'] < 40)]
age_2_M=df_male[(df_male['Age'] >= 40)
             & (df_male['Age'] < 60)]
age_2_F=df_female[(df_female['Age'] >= 40)
             & (df_female['Age'] < 60)]
age_3_M=df_male[(df_male['Age'] >= 60)]
age_3_F=df_female[(df_female['Age'] >= 60)]

age_1_M['subgroup_dict']="M1"
age_1_F['subgroup_dict']="F1"
age_2_M['subgroup_dict']="M2"
age_2_F['subgroup_dict']="F2"
age_3_M['subgroup_dict']="M3"
age_3_F['subgroup_dict']="F3"

df_sub = pd.concat([age_1_M, age_1_F], axis=0)
df_sub = pd.concat([df_sub, age_2_M], axis=0)
df_sub = pd.concat([df_sub, age_2_F], axis=0)
df_sub = pd.concat([df_sub, age_3_M], axis=0)
df_sub = pd.concat([df_sub, age_3_F], axis=0)
df_sub = df_sub.sort_index()

subgroup_count = pd.DataFrame({'sub_name':['M1', 'F1', 'M2', 'F2', 'M3', 'F3',], 'length':[1,1,1,1,1,1]})

subgroup_count.iat[0,0]=len(age_1_M)
subgroup_count.iat[1,0]=len(age_1_F)
subgroup_count.iat[2,0]=len(age_2_M)
subgroup_count.iat[3,0]=len(age_2_F)
subgroup_count.iat[4,0]=len(age_3_M)
subgroup_count.iat[5,0]=len(age_3_F)

print(df_sub)
print(subgroup_count)

#データフレームから配列に変換
bw = np.array(df_lim['Body_weight'])
gen = np.array(df_lim['Gender'])
age = np.array(df_lim['Age'])

demo = np.c_[bw,gen]
demo = np.c_[demo,age]
#print(demo)

#配列からデータフレームに変換して指定した条件で抽出
df_demo = pd.DataFrame(demo,columns=['bw','gen','age'])
df_demo_lim = df_demo[(df_demo['gen'] == 1) 
                      & (df_demo['bw'] <= 50)]
#print(df_demo_lim)

#散布図作成(Age-HbA1c)
"""
p = figure(plot_width = 600, plot_height = 600,
           title = 'HbA1c - Male',
           x_axis_label='Age', y_axis_label='HbA1c')

p.circle(df_male.loc[:,'Age'], df_male.loc[:,'HbA1c'], size=12, color='blue')
output_notebook()
show(p)

#散布図作成(Age-BMI)
p = figure(plot_width = 600, plot_height = 600,
           title = 'HbA1c - female',
           x_axis_label='Age', y_axis_label='BMI')

p.circle(df_female.loc[:,'Age'], df_female.loc[:,'BMI'], size=12, color='pink')
output_notebook()
show(p)
"""

#相関係数
df_lim_corr = df_lim.corr(method='pearson')
print(df_lim_corr)

#ヒストグラム作成
"""
plt.figure(figsize=(15, 20))
 
plt.subplot(3,2,1)
plt.hist(df_lim['Age'])
plt.title('Python Workshop 2018')
plt.xlabel('Age')
plt.ylabel('freq')

plt.subplot(3,2,2)
plt.hist(df_lim['Gender'])
plt.title('Python Workshop 2018')
plt.xlabel('gender')
plt.ylabel('freq')

plt.subplot(3,2,3)
plt.hist(df_lim['Body_weight'])
plt.title('Python Workshop 2018')
plt.xlabel('Body_weight')
plt.ylabel('freq')

plt.subplot(3,2,4)
plt.hist(df_lim['BMI'])
plt.title('Python Workshop 2018')
plt.xlabel('BMI')
plt.ylabel('freq')

plt.subplot(3,2,5)
plt.hist(df_lim['HbA1c'])
plt.title('Python Workshop 2018')
plt.xlabel('HbA1c')
plt.ylabel('freq')

plt.subplot(3,2,6)
plt.hist(df_lim['Glucose'])
plt.title('Python Workshop 2018')
plt.xlabel('Glucose')
plt.ylabel('freq')

plt.savefig("hist.png")
"""