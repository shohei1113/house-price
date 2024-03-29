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
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.head(5)
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

test.head(5)

x_train = train.drop('SalePrice', axis=1)
x_train.head()
y_train = train['SalePrice']
y_train.head()

x_train.corr()

df_cor = train.corr()
df_cor['SalePrice'].sort_values(ascending=False)

x_train = train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]

x_train.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# +
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)
model.score(x_train, y_train)
# -
model.score(x_test, y_test)







