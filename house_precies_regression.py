
# HOUSE PREDICTON MODEL

### İŞ PROBLEMİ
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak, farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.

### VERİ SETİ HİKAYESİ
# Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki
# linkten ulaşabilirsiniz. Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları boş bırakılmış olup, bu
# değerleri sizin tahmin etmeniz beklenmektedir.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# VERİ SETİ OKUTMA
train_df = pd.read_csv("/Users/ahmetbozkurt/Desktop/House_Precies_Regression/house_dataset/train.csv")
test_df = pd.read_csv("/Users/ahmetbozkurt/Desktop/House_Precies_Regression/house_dataset/test.csv")

# İki dataframe'i tek bir dataframe olarak birleştiriyorum.# İki dataframe'i tek bir dataframe olarak birleştiriyorum.
df = pd.concat([train_df, test_df], ignore_index=True)

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)

# Numerik ve kategorik değişkenleri yakalayalım.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik,numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir
    cat_th: int, float
        Numerik fakat kategorik değişkenler için sınıf eşik değeri
    car_th: int, float
        Kategorik fakat kardinal değişkenlerin sınıf eşik değeri

      Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyelim.

# Kategorik Değişkenler için:
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

# Numerik değişkenler için

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    print(f"############## {col} ############")
    num_summary(df, col)


#  Kategorik değişkenler ile hedef değişken inceleyelim.
def target_category(dataframe,  target, col_category):
    print(dataframe.groupby(col_category).agg({target: "mean"}))
    print("#" * 40)

for col in cat_cols:
    print(f"######### {col.upper()} #########")
    target_category(df, "SalePrice", col)

df["Utilities"].value_counts()

# Aykırı Gözlem İçeren değişkenleri bulalım.

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe.loc[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit

# Aykırı Değerler VARDIR!!!


# Korelasyon İnceleyelim.
def corr_map(df, width=14, height=6, annot_kws=15, corr_th=0.7):
    corr = df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))  # np.bool yerine bool
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    mtx = np.triu(df.corr())
    f, ax = plt.subplots(figsize = (width,height))
    sns.heatmap(df.corr(),
                annot= True,
                fmt = ".2f",
                ax=ax,
                vmin = -1,
                vmax = 1,
                cmap = "RdBu",
                mask = mtx,
                linewidth = 0.4,
                linecolor = "black",
                annot_kws={"size": annot_kws})
    plt.yticks(rotation=0,size=15)
    plt.xticks(rotation=75,size=15)
    plt.title('\nCorrelation Map\n', size = 40)
    plt.show();
    return drop_list

corr_map(df[num_cols])


#  Eksik gözlem var mı inceleyelim.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns

missing_values_table(df)


###################################
# Feature Engineering
###################################

# Aykırı değerler için

for col in num_cols:
    if col != "SalesPrice":
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


# GarageYrBlt değişkeninin son değeri baskılamama rağmen 2150 oldu bunu en son hangi değer varsa onunla değiştiriyorum.
df["GarageYrBlt"].describe().T
# Bu kod ile en son 2010 olduğunu gördüm onunla değiştiriyorum
df.loc[df["GarageYrBlt"] == 2150, "GarageYrBlt"] = 2010.0


# Eksik Değerler
# Veri seti içerisinde bazı eksik değerler birbiri ile ilişkili ve bunlar arasındaki ilşikiden değişkenlerin aslında olmadığı için NaN gözüküyor.Bu yüzden değişkenin dtype değerine bakarak 
# uygun olan "No" veya 0 ile doldurucağız.

no_list = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType", "FireplaceQu",
           "MiscFeature", "Fence", "PoolQC", "GarageFinish", "GarageQual", "GarageCond"]
zero_list = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea", "BsmtFullBath", "BsmtHalfBath",
             "GarageYrBlt", ]

df[no_list] = df[no_list].fillna("No")
df[zero_list] = df[zero_list].fillna(0)

missing_values_table(df)


# Şimdi de geriye kalan boş değerleri bu fonksiyon ile değişken türüne göre median ve mode ile dolduruyoruz.

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

df = quick_missing_imp(df)

# Rare Encoder uygulayalım..

# Bazı kategorik değişkenlerin içindeki unique değerlerin sayısı çok az.Bunlara müdahele etmeden encode ederek veri setimin değişken sayısını arttırmış olacağım ve model aşamasında train sürelerim 
# uzamış olacak.

def rare_analyser(dataframe, target, col_names):
    print(col_names, ":", len(dataframe[col_names].value_counts()))
    print(pd.DataFrame({"COUNT": dataframe[col_names].value_counts(),
                        "RATIO": dataframe[col_names].value_counts() / len(dataframe),
                        "TARGET_MEAN": dataframe.groupby(col_names)[target].mean()}))
    print("#" * 50, end="\n\n")

for col in cat_cols:
    rare_analyser(df, "SalePrice", col)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp.loc[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

# Şimdi de veri seti içerisinden yeni feature'lar üretelim.
df.columns = [col.title() for col in df.columns]

# Total floor değişkeni
df["New_Total_Flr"] = df["1Stflrsf"] + df["2Ndflrsf"]

# Satıldığı ay bilgisini alma
df["New_Month_Name"] = pd.to_datetime(df["Mosold"], format="%m")
df["New_Month_Name"] = pd.to_datetime(df["New_Month_Name"]).dt.strftime("%B")

# Binanın yaşını hesaplayalım
df["New_Build_Age"] = np.abs(df["Yrsold"] - df["Yearbuilt"])

# Satılmadan ne kadar önce tadilat yapılmış
df["Yearremodadd"] = df["Yearremodadd"].astype(int)
df["New_Remode_End"] = np.abs(df["Yrsold"] - df["Yearremodadd"])

# Garajın yaşı
df["Garageyrblt"] = df["Garageyrblt"].astype(int)
df["New_Garage_Age"] = np.abs(df["Yrsold"] - df["Garageyrblt"])

# Garaj var mı yok mu
df["New_Garage_IsThere"] = df["Garagearea"].apply(lambda x: 1 if x > 0 else 0)

# Eski ve yeni binaları belirleyelim
df["New_Build_Age"].describe().T
df.loc[(df["New_Build_Age"] >= 0) & (df["New_Build_Age"] < 10), "New_Build_Cond"] = "New_Build"
df.loc[(df["New_Build_Age"] >= 10) & (df["New_Build_Age"] < 30), "New_Build_Cond"] = "Middle_Age_Build"
df.loc[(df["New_Build_Age"] >= 30) & (df["New_Build_Age"] < 60), "New_Build_Cond"] = "Old_Build"
df.loc[df["New_Build_Age"] >= 60, "New_Build_Cond"] = "Very_Old_Build"

# Bodrum dahil total alan
df["New_Total_Area"] = df["New_Total_Flr"] + df["Totalbsmtsf"]

# Toplam banyo sayıları
df["New_Total_HalfBath"] = df["Bsmthalfbath"] + df["Halfbath"]
df["New_Total_FullBath"] = df["Bsmtfullbath"] + df["Fullbath"]
df["New_Total_Bath"] = df["New_Total_FullBath"] + df["New_Total_HalfBath"]

# Aylardan Mevsim oluşturma
seasons = {
    "Winter": ["December", "January", "February"],
    "Spring": ["March", "April", "May"],
    "Summer": ["June", "July", "August"],
    "Autumn": ["September", "October", "November"]
}

def season_find(month):
    for season, months in seasons.items():
        if month in months:
            return season

df["New_Season"] = df["New_Month_Name"].apply(lambda x: season_find(x))

# Duvar kaplaması var mı yok mu
df["New_Masvnr_IsThere"] = df["Masvnrarea"].apply(lambda x: 1 if x > 0 else 0)

# Farklı özellikleri var mı yok mu
df["New_Miscval_IsThere"] = df["Miscval"].apply(lambda x: 1 if x > 0 else 0)

# Tekrar değişkenleri sınıflandırıyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Bazı değişkenler aynı şeyleri veya veri setinde bir şey ifade etmediği için siliyorum.
drop_list = ["Mosold", "Poolqc", "Poolarea", "Miscfeature", "Utilities", "Street", "Alley", "Heating", "Roofmatl",
             "Condition2"]
df = df.drop(drop_list, axis=1)


# Encoding işlemlerini gerçekleştirelim.

# Label encoder
def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_col = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_col:
    label_encoder(df, col)

# One Hot Encoder
def one_hot_encoder(dataframe, categorical_col, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_col, drop_first=drop_first, dtype=int)
    return dataframe

ohe_cols = [col for col in df.columns if col not in binary_col and df[col].dtypes == "O"]

df = one_hot_encoder(df, ohe_cols)
df.shape

# Scale işlemlerini halledelim.
scale_not_list = ["Id", "Saleprice"]
numeric_cols = df.select_dtypes(include="number").columns
scale_list = [col for col in numeric_cols if col not in scale_not_list]

# RobustScaler kullanalım.
rs_scale = RobustScaler()
df[scale_list] = rs_scale.fit_transform(df[scale_list])



###############################
# MODEL KURMA
###############################

# Train ve Test verisini ayırınız.
train_df = df.loc[df["Saleprice"].notnull()]
test_df = df.loc[df["Saleprice"].isnull()]
train_df.shape
test_df.shape
train_df.tail()

X = train_df.drop(["Saleprice", "Id"], axis=1)
y = train_df["Saleprice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor(verbosity=-1)),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 3349581045375.2065 (LR) 
# RMSE: 49761.0689 (KNN) 
# RMSE: 40938.1852 (CART) 
# RMSE: 29225.5866 (RF) 
# RMSE: 25618.296 (GBM) 
# RMSE: 28657.7033 (XGBoost) 
# RMSE: 27746.7145 (LightGBM) 
# RMSE: 25612.3248 (CatBoost) 

# Seçtiğim modeller
# RMSE: 27746.7145 (LightGBM) 
# RMSE: 25612.3248 (CatBoost) 

# Bonus olarak MAE skorlarına bir fonksiyonla bakalım.
min_mae = np.inf
for name, regressor in models:
    model = regressor.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    if mae < min_mae:
        min_mae = mae
        final_model_name = name
        final_model = model
    print(f"MAE: {mae} ({name}) ")
print(f"Minumum MAE: {min_mae} with {final_model}")



# Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyelim.

plt.hist(np.log1p(train_df['Saleprice']), bins=100)  # LOG DÖNÜŞÜMÜ GRAFİĞİ
X_log = train_df.drop(["Saleprice", "Id"], axis=1)
y_log = np.log1p(train_df["Saleprice"])

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

model_lgbm = LGBMRegressor()
model_log = model_lgbm.fit(X_train_log, y_train_log)
y_pred_log = model_log.predict(X_test_log)

# Yapılan LOG dönüşümünün tersini alarak model sonucuma bakıyorum.Fakat daha da yükseltti.
new_y_pred = np.expm1(y_pred_log)
new_y_test = np.expm1(y_test_log)
np.sqrt(mean_squared_error(new_y_test, new_y_pred))

# Hiperparametre Optimizasyonu
# İki modelle devam ediyorum.LightGBM ve CatBosst
lgbm = LGBMRegressor(verbosity=-1)
catboost = CatBoostRegressor(verbose=False)

rmse_lgbm = np.mean(np.sqrt(-cross_val_score(lgbm, X, y, cv=5, scoring="neg_mean_squared_error")))
# 27746.714489035832
rmse_cat = np.mean(np.sqrt(-cross_val_score(catboost, X, y, cv=5, scoring="neg_mean_squared_error")))
# 25612.3247761981

# LightGBM ile optimizasyonu gerçekleştireceğim.
lgbm = LGBMRegressor(verbosity=-1)
lightgbm_params = {"learning_rate": [0.1, 0.01],
                   "max_depth": [-1, 1, 3],
                   "n_estimators": [200, 300, 500],
                   "colsample_bytree": [0.5, 0.7, 1]}
lgbm_bs_params = GridSearchCV(lgbm, lightgbm_params, cv=5, n_jobs=-1).fit(X, y)
lgbm_bs_params.best_params_

final_model_lgbm = lgbm.set_params(**lgbm_bs_params.best_params_).fit(X, y)
new_rmse_lgbm = np.mean(np.sqrt(-cross_val_score(final_model_lgbm, X, y, cv=5, scoring="neg_mean_squared_error")))


################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdirelim..
################################################################
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(final_model_lgbm, X, num=15)

# Grafikte modelimizin tahminlerini inceleyelim.
final_model_lgbm.fit(X_train, y_train)
train_predictions = final_model_lgbm.predict(X_test)

plt.figure(figsize=(34, 8))
plt.subplot(1, 2, 1)
plt.plot(y_test.values, label='Gerçek Değerler')
plt.plot(train_predictions, label='Tahmin Edilen Değerler', linestyle='--')
plt.xlabel('Örnekler')
plt.ylabel('Değerler')
plt.title('Eğitim Seti: Gerçek vs Tahmin')
plt.legend()

plt.tight_layout()
plt.show()


########################################
# test dataframeindeki boş olan salePrice değişkenlerini tahminleyiniz ve
# Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturunuz. (Id, SalePrice)
########################################

# test dataframeindeki boş olan salePrice değişkenlerini tahminleyelim ve Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturalım. (Id, SalePrice)
base = test_df.drop(["Id", "Saleprice"], axis=1)
predictions = final_model_lgbm.predict(base)
submissions_dict = {"Id": test_df["Id"], "SalePrice": predictions}
submissions_df = pd.DataFrame(submissions_dict)
submissions_df["Id"] = submissions_df["Id"].astype(int)
submissions_df.to_csv("house_prediction.csv", index=False)
