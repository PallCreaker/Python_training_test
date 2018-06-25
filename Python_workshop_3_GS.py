import pandas as pd
import numpy as np

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
#df_result = pd.dataframe({
 #                   'colum' : df_new.T[:,1],
  #                  'median' : df(np.median(df_new,axis=0),
   #                 'variance' : np.var(df_new,axis=0),
    #                'mean' : np.mean(df_new,axis=0),
     #               'min' : np.min(df_new,axis=0),
      #              'max' : np.max(df_new,axis=0),
       #             'quantiles(1st)' : np.percentile(df_new,25,axis=0),
        #            'quantiles(3rd)' : np.percentile(df_new,75,axis=0),
         #           'standard deviation' : np.std(df_new,axis=0),
          #          })

result=df_new[['Age','Gender','Body_weight','BMI','HbA1c','Glucose']].describe()
#print(result)

#指定した条件で抽出
df_lim = df_new[(df_new['Age'] >= 20) & (df_new['BMI'] <= 50) & (df_new['BMI'] >= 18) & (df_new['Glucose'] <= 16) & (df_new['Glucose'] >= 4) & (df_new['HbA1c'] <= 10) & (df_new['HbA1c'] >= 4)]

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

#データフレームから配列に変換
bw = np.array(df_temp_r['B'])
gen = np.array(df_temp_r['Gender'])
age = np.array(df_temp_r['Age'])

demo = np.c_[bw,gen]
demo = np.c_[demo,age]
#print(demo)

#配列からデータフレームに変換して指定した条件で抽出
df_demo = pd.DataFrame(demo,columns=['bw','gen','age'])
df_demo_lim = df_demo[(df_demo['gen'] == 1) & (df_demo['bw'] <= 50)]
#print(df_demo_lim)

#相関係数
def_lim_corr = df_lim.corr(method='pearson')
#print(def_lim_corr)

#ヒストグラム作成
import matplotlib.pyplot as plt

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
