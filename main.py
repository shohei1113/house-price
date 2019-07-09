# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{: .3f}'.format(x))

from subprocess import check_output

# +
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.head(5)
# -

test.head(5)

# +
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

print('\nThe train data size after dropping Id feature is : {}')
print('\nThe test data size after dropping Id feature is : {}')
# -

# SalePrice  - 施設の売却価格（ドル）。これは予測しようとしているターゲット変数です。    
# MSSubClass：建物クラス    
# MSZoning：一般的なゾーニング分類      
# LotFrontage：不動産に接続されている通りの直線フィート    
# LotArea：ロットサイズ（平方フィート）    
# Street：道路アクセスの種類    
# Alley：路地アクセスの種類    
# LotShape：資産の一般的な形状    
# LandContour：物件の平坦度    
# Utilities：利用可能なユーティリティの種類    
# LotConfig：ロット構成    
# LandSlope：土地の傾斜    
# Neighborhood：Ames市域内の物理的な場所    
# Condition1：幹線道路または鉄道への近接    
# Condition2：幹線道路または鉄道への近さ（秒がある場合）    
# BldgType：住居の種類    
# HouseStyle：住まいのスタイル    
# OverallQual：全体的な素材と仕上げの品質    
# OverallCond：全体的なコンディション評価    
# YearBuilt：元の建設日    
# YearRemodAdd：改造日    
# RoofStyle：屋根の種類    
# RoofMatl：屋根材    
# Exterior1st：家の外装    
# Exterior2nd：家の外装材（複数の素材がある場合）    
# MasVnrType：石積みのベニヤタイプ    
# MasVnrArea：メーソンリーベニアの面積（平方フィート）    
# ExterQual：外装材の品質    
# ExterCond：外装上の材料の現状    
# MasVnrType：財団の種類    
# BsmtQual：地下室の高さ    
# BsmtCond：地下室の概況    
# Foundation：ストライキまたは庭レベルの地下壁    
# BsmtFinType1：地下室仕上がり面積    
# BsmtFinSF1：タイプ1仕上げ平方フィート    
# BsmtFinType2：2番目に完成した領域の品質（存在する場合）    
# BsmtFinSF2：タイプ2仕上げ平方フィート    
# BsmtUnfSF：地下1平方フィート    
# TotalBsmtSF：地下室の総平方フィート    
# Heating：暖房の種類    
# HeatingQC：暖房の品質と状態    
# CentralAir：セントラルエアコン    
# Electrical：電気システム    
# 1stFlrSF：1階平方フィート    
# 2ndFlrSF：2階平方フィート    
# LowQualFinSF：低品質仕上げ平方フィート（全フロア）    
# GrLivArea：上階（地面）のリビングエリア平方フィート    
# BsmtFullBath：地下フルバスルーム    
# BsmtHalfBath：地下ハーフバスルーム    
# FullBath：グレード以上のフルバスルーム    
# HalfBath：グレード以上のハーフバス    
# Bedroom：地下室より上の寝室の数    
# Kitchen：キッチンの数    
# KitchenQual：キッチンの質    
# TotRmsAbvGrd：グレード以上の総客室数（バスルームは含まれません）    
# Functional：家庭用機能性評価    
# Fireplaces：暖炉の数    
# FireplaceQu：暖炉の品質    
# GarageType：ガレージの場所    
# GarageYrBlt：年式ガレージが建てられました    
# GarageFinish：ガレージの内部仕上げ    
# GarageCars：自動車容量のガレージサイズ    
# GarageArea：ガレージサイズ（平方フィート）    
# GarageQual：ガレージ品質    
# GarageCond：ガレージ状態    
# PavedDrive：舗装された私道    
# WoodDeckSF：ウッドデッキ面積（平方フィート）    
# OpenPorchSF：オープンポーチ面積（平方フィート）    
# EnclosedPorch：囲まれたポーチの面積（平方フィート）    
# 3SsnPorch：3平方フィートの3シーズンポーチ面積    
# ScreenPorch：スクリーンポーチの面積（平方フィート）    
# PoolArea：プール面積（平方フィート）    
# PoolQC：プール品質    
# Fence：フェンスの品質    
# MiscFeature：他のカテゴリに含まれていないその他の機能    
# MiscVal：$その他の機能の価値    
# MoSold：月売れ    
# YrSold：販売年    
# SaleType：販売の種類    
# SaleCondition：販売条件    

# ## データ処理

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# +
# 右下に外れ値となるデータがあるため削除

# 外れ値の削除は必ずするべき
# トレーニングデータにはおそらく他の外れ値はある。テストデータに異常値がある場合、
# それらを全て削除するとモデルに悪影響が出るためそれらの全てを削除するわけではない。
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.ylabel('GrLivArea', fontsize=13)
plt.show()

# +
# 頻度を表すグラフと正規確率プロットを描画
# 正規確率プロットは一直線の場合正規分布
sns.distplot(train['SalePrice'], fit=norm)

(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {: .2f} and sigma = {: .2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {: .2f} and $sigma=$ {: .2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
# -

# ターゲット変数は右に傾いている。    
# モデルは正規分布データを好むので、この変数を変換してより正規分布にする必要があります

# +
train['SalePrice'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'], fit=norm)

(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {: .2f} and sigma = {: .2f}\n' .format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
# -

ntrain = train.shape[0]
print(ntrain)
ntest = test.shape[0]
print(ntest)
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print('all_data size is : {}' .format(all_data.shape))

# ## 欠損値

all_data_na = (all_data.isnull().sum() / len(all_data) * 100)
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

# 欠損値を可視化
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.xlabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

corrmat = train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)

# ヒートマップから大多数の家にはプールがないことがわかる
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

# 大多数の家には路地のアクセスがない
all_data['Alley'] = all_data['Alley'].fillna('None')

# 大多数の家にはフェンスがない
all_data['Fence'] = all_data['Fence'].fillna('None')

# 大多数の家には暖炉がない
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')

# 不動産に接続されている通りの直線フィートは、
# その近隣にある他の住宅とほぼ同じ面積を持つ可能性が高いため、近隣のLotFontageの中央値で欠損地を埋めることができる
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data['Functional'] = all_data['Functional'].fillna('Typ')

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

# +
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
# -

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# +
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
# -

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# +
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

# +
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
# -

all_data = pd.get_dummies(all_data)
print(all_data.shape)


train = all_data[:ntrain]
test = all_data[ntrain:]

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# +
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# -

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



