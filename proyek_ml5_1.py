"""
# Analisis Prediktif: Prediksi Harga Rumah di Jakarta Selatan
<hr>

### *Oleh: [Panji Arlin Saputra](https://www.dicoding.com/users/panjiarlins)*
### *Proyek Submission 1: Machine Learning Terapan Dicoding*
<hr>

## **Pendahuluan**
Pada proyek ini, topik yang dibahas adalah mengenai perniagaan yang dibuat untuk memprediksi data harga jual rumah di Jakarta Selatan. Proyek ini dibuat untuk proyek Submission 1 - Machine Learning Terapan Dicoding.

# **1. Mengimpor pustaka/modul python yang dibutuhkan**
"""

# Memasang modul scikit-learn terbaru
!pip install -U scikit-learn

# Commented out IPython magic to ensure Python compatibility.
# Untuk pengolahan data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Untuk visualisasi data
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# Untuk pembuatan model
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# Untuk evaluasi model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

"""# **2. Mempersiapkan Dataset**

## **2.1 Menyiapkan kredensial akun Kaggle**
"""

# Membuat folder .kaggle di dalam folder root
!rm -rf ~/.kaggle && mkdir ~/.kaggle/

# Menyalin berkas kaggle.json pada direktori aktif saat ini ke folder .kaggle
!mv kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

"""## **2.2 Mengunduh dan Menyiapkan Dataset**

Informasi Dataset:

Jenis | Keterangan
----- | -----
Sumber | [Kaggle Dataset: Daftar Harga Rumah](https://www.kaggle.com/wisnuanggara/daftar-harga-rumah)
Jenis dan Ukuran Berkas | XLSX (118 kB)

"""

# Mengunduh dataset menggunakan Kaggle CLI
!kaggle datasets download wisnuanggara/daftar-harga-rumah

# Mengekstrak berkas zip ke direktori aktif saat ini
!unzip /content/daftar-harga-rumah.zip

"""# **3. Pemahaman Data (Data Understanding)**

## **3.1 Memuat Data pada sebuah Dataframe menggunakan pandas**
"""

# Untuk memuat himpunan data
path = '/content/HARGA RUMAH JAKSEL.xlsx'
rumah = pd.read_excel(path, header=1)

"""## **3.2 Keterangan kolom pada dataset**"""

# Menampilkan sample pada dataset
rumah.head(10)

# Memuat informasi dataframe
rumah.info()

# Menghitung jumlah data yang kosong pada setiap kolom
rumah.isna().sum()

# Memuat deskripsi setiap kolom dataframe
rumah.describe().round(2)

"""# **4. Persiapan Data (Data Preparation)**

## **4.1 Mengatasi masalah data yang semua nilainya sama pada satu kolom dengan menghapus kolom tersebut**
"""

# Menghapus kolom 'KOTA'
rumah = rumah.drop(['KOTA'], axis=1)

# Mengecek total baris dan kolom dari dataset
rumah.shape

"""## **4.2 Mengatasi masalah data harga yang nilainya terlalu besar dengan membaginya dengan 10<sup>9</sup> agar satuannya nilainya menjadi miliar rupiah**"""

rumah['HARGA'] = rumah['HARGA'] / 1e9

"""## **4.3 Menangani _data outliers_**"""

# Memvisualisasikan data Luas Tanah dengan boxplot untuk mendeteksi outliers
sns.boxplot(x=rumah['LT'])

# Menghapus data ouliers dengan metode IQR
Q1 = rumah.quantile(0.25)
Q3 = rumah.quantile(0.75)
IQR = Q3 - Q1
rumah = rumah[~((rumah<(Q1-1.5*IQR))|(rumah>(Q3+1.5*IQR))).any(axis=1)]

# Mengecek ukuran dataset setelah menghapus data outliers
rumah.shape

"""## **4.4 Analisis Univariat**"""

# Membagi fitur pada dataset menjadi dua bagian
numerical_features = ['HARGA', 'LT', 'LB', 'JKT', 'JKM']
categorical_features = 'GRS'

# Fitur kategoris
count = rumah[categorical_features].value_counts()
percent = 100 * rumah[categorical_features].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=categorical_features);

# Fitur numerik
rumah.hist(bins=15, figsize=(10,10))
plt.show()

"""## **4.5 Analisis Multivariat**"""

# Fitur kategoris
sns.catplot(x=categorical_features, y='HARGA', kind='bar', dodge=False, height=4, aspect=3,  data=rumah, palette='Set3')
plt.title('Rata-rata "HARGA" Relatif terhadap - {}'.format(categorical_features))

# Mengamati hubungan antar fitur numerik
sns.pairplot(rumah, diag_kind = 'kde')

# Mengevaluasi skor korelasi
plt.figure(figsize=(10, 8))
correlation_matrix = rumah.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title('Metrik Korelasi Untuk Fitur Numerik', size=20)

"""## **4.6 _Encoding_ fitur kategoris**"""

# Melakukan encoding pada fitur kategoris
rumah = pd.concat([rumah, pd.get_dummies(rumah['GRS'], prefix='GRS', drop_first=True)],axis=1)
rumah.drop(['GRS'], axis=1, inplace=True)
rumah.rename({'GRS_TIDAK ADA':'GRS'}, axis=1, inplace=True)

# Menampilkan dataset setelah dilakukan proses encoding
rumah.head(10)

"""## **4.7 Reduksi Dimensi dengan PCA**"""

# Mengamati korelasi antara fitur 'LT' dan 'LB'
sns.pairplot(rumah[['LT','LB']], plot_kws={'s': 3});

# Mengamati proporsi informasi pada fitur 'LT' dan 'LB'
pca = PCA(n_components=2, random_state=24)
pca.fit(rumah[['LT','LB']])
princ_comp = pca.transform(rumah[['LT','LB']])
pca.explained_variance_ratio_.round(3)

# Mengaplikasikan PCA pada fitur 'LT' dan 'LB'
pca = PCA(n_components=1, random_state=24)
pca.fit(rumah[['LT','LB']])
rumah['LUAS'] = pca.transform(rumah.loc[:, ('LT','LB')]).flatten()
rumah.drop(['LT','LB'], axis=1, inplace=True)
rumah.head()

"""## **4.8 Melakukan pembagian data pada dataset dengan train_test_split**"""

X = rumah.drop(['HARGA'], axis=1)
y = rumah['HARGA']

# Melakukan pembagian data dengan train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=24)

# Mengecek jumlah baris pada data latih dan data tes
print(X_train.shape)
print(X_test.shape)

"""## **4.9 Standarisasi nilai data pada fitur numerik dengan StandardScaler**"""

# Inisialisasi fungsi MinMaxScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Melihat sampel data hasil standarisasi data
print(f'{X_train[0:5]}\n')
print(X_test[0:5])

# Mengecek jumlah baris pada data latih dan data tes
print(X_train.shape)
print(X_test.shape)

"""# **5. Pembuatan Model**

## **5.1 Model Baseline dengan Algoritma K-Nearest Neighbors**
"""

# Pembuatan model baseline
baseline_knn = KNeighborsRegressor()
baseline_knn.fit(X_train, y_train)

# Mengukur metrik evaluasi pada model baseline terhadap data tes
baseline_knn_mse = mean_squared_error(y_true=y_test, y_pred=baseline_knn.predict(X_test))
baseline_knn_r2 = r2_score(y_true=y_test, y_pred=baseline_knn.predict(X_test))
baseline_knn_mae = mean_absolute_error(y_true=y_test, y_pred=baseline_knn.predict(X_test))

# Menampilkan hasil pengukuran metrik evaluasi
print('MODEL BASELINE KNN\n')
print(f'MSE: {baseline_knn_mse}')
print(f'R-Squared: {baseline_knn_r2}')
print(f'MAE: {baseline_knn_mae}')

"""## **5.2 Pengembangan Model K-Nearest Neighbors dengan Hyper Parameter Tuning menggunakan HalvingGridSearchCV**"""

# Hyperparameter yang akan di tuning
param_grid = {'n_neighbors': range(1,24),
              'p': [1, 2],
              'weights': ["uniform", "distance"],
              'algorithm': ["ball_tree", "kd_tree", "brute"],
              }

# Pencarian parameter terbaik dengan HalvingGridSearchCV
knn_new_param = HalvingGridSearchCV(baseline_knn, param_grid, aggressive_elimination=True).fit(X_train, y_train)

# Hasil hyperparameter tuning dengan skor terbaik yang di dapatkan
print(f"Best parameter: {knn_new_param.best_estimator_}")

# Penerapan hyperparameter pada model
knn = KNeighborsRegressor(**knn_new_param.best_params_)
knn.fit(X_train, y_train)

"""## **5.3 Model Baseline dengan Algoritma Random Forest**"""

# Pembuatan model baseline
baseline_RF = RandomForestRegressor()
baseline_RF.fit(X_train, y_train)

# Mengukur metrik evaluasi pada model baseline terhadap data tes
baseline_RF_mse = mean_squared_error(y_true=y_test, y_pred=baseline_RF.predict(X_test))
baseline_RF_r2 = r2_score(y_true=y_test, y_pred=baseline_RF.predict(X_test))
baseline_RF_mae = mean_absolute_error(y_true=y_test, y_pred=baseline_RF.predict(X_test))

# Menampilkan hasil pengukuran metrik evaluasi
print('MODEL BASELINE RF\n')
print(f'MSE: {baseline_RF_mse}')
print(f'R-Squared: {baseline_RF_r2}')
print(f'MAE: {baseline_RF_mae}')

"""## **5.4 Pengembangan Model Random Forest dengan Hyper Parameter Tuning menggunakan HalvingGridSearchCV**"""

# Hyperparameter yang akan di tuning
param_grid = {'n_estimators': range(10,110,5)}

# Pencarian parameter terbaik dengan HalvingGridSearchCV
RF_new_param = HalvingGridSearchCV(baseline_RF, param_grid, aggressive_elimination=True).fit(X_train, y_train)

# Hasil hyperparameter tuning dengan skor terbaik yang di dapatkan
print(f"Best parameter: {RF_new_param.best_estimator_}")

# Penerapan hyperparameter pada model
RF = RandomForestRegressor(**RF_new_param.best_params_, n_jobs=-1, random_state=24)
RF.fit(X_train, y_train)

"""## **5.5 Model Baseline dengan Algoritma Adaptive Boosting**"""

# Pembuatan model baseline
baseline_boosting = AdaBoostRegressor()
baseline_boosting.fit(X_train, y_train)

# Mengukur metrik evaluasi pada model baseline terhadap data tes
baseline_boosting_mse = mean_squared_error(y_true=y_test, y_pred=baseline_boosting.predict(X_test))
baseline_boosting_r2 = r2_score(y_true=y_test, y_pred=baseline_boosting.predict(X_test))
baseline_boosting_mae = mean_absolute_error(y_true=y_test, y_pred=baseline_boosting.predict(X_test))

# Menampilkan hasil pengukuran metrik evaluasi
print('MODEL BASELINE boosting\n')
print(f'MSE: {baseline_boosting_mse}')
print(f'R-Squared: {baseline_boosting_r2}')
print(f'MAE: {baseline_boosting_mae}')

"""## **5.6 Pengembangan Model Adaptive Boosting dengan Hyper Parameter Tuning menggunakan HalvingGridSearchCV**"""

# Hyperparameter yang akan di tuning
param_grid = {'n_estimators': range(10,101,5), 'learning_rate': [0.01, 0.05, 0.1, 0.5]}

# Pencarian parameter terbaik dengan HalvingGridSearchCV
boosting_new_param = HalvingGridSearchCV(baseline_boosting, param_grid, aggressive_elimination=True).fit(X_train, y_train)

# Hasil hyperparameter tuning dengan skor terbaik yang di dapatkan
print(f"Best parameter: {boosting_new_param.best_estimator_}")

# Penerapan hyperparameter pada model
boosting = AdaBoostRegressor(**boosting_new_param.best_params_, random_state=24)
boosting.fit(X_train, y_train)

"""# **6. Evaluasi Model**

## **6.1 Mengukur metrik evaluasi model pada setiap algoritma yang digunakan**
"""

# Menyiapkan dataframe untuk analisis model
mse = pd.DataFrame(columns=['train', 'test'])
r2 = pd.DataFrame(columns=['train', 'test'])
mae = pd.DataFrame(columns=['train', 'test'])

# Melakukan pengukuran metrik evaluasi
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))

    r2.loc[name, 'train'] = r2_score(y_true=y_train, y_pred=model.predict(X_train))
    r2.loc[name, 'test'] = r2_score(y_true=y_test, y_pred=model.predict(X_test))

    mae.loc[name, 'train'] = mean_absolute_error(y_true=y_train, y_pred=model.predict(X_train))
    mae.loc[name, 'test'] = mean_absolute_error(y_true=y_test, y_pred=model.predict(X_test))

"""## **6.2 Memvisualisasikan hasil pengukuran metrik evaluasi**"""

# Metrik evaluasi MSE
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
ax.set_title('MSE')
print(mse)

# Metrik evaluasi R2-Squared
fig, ax = plt.subplots()
r2.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
ax.set_title('R2-Squared')
print(r2)

# Metrik Evaluasi MAE
fig, ax = plt.subplots()
mae.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
ax.set_title('MAE')
print(mae)

"""## **6.3 Melakukan prediksi dengan data uji**"""

prediksi = X_test[:2].copy()
pred_dict = {'y_true': y_test[:2]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(2)
 
pd.DataFrame(pred_dict)

"""# **Penutupan**

Model untuk memprediksi harga rumah di Jakarta Selatan telah selesai dibuat dan model ini dapat digunakan untuk memprediksi data yang sebenarnya. Berdasarkan hasil prediksi yang dilakukan terhadap data uji pada proyek ini, dapat disimpulkan bahwa model yang menggunakan algoritma Random Forest menghasilkan hasil prediksi yang lebih baik dibandingkan model yang menggunakan algoritma K-Nearest Neighbors dan Adaptive Boosting.


### **_Referensi_**

* Dokumentasi Scikit-learn: https://scikit-learn.org/stable/modules/classes.html
* Kaggle Dataset Daftar Harga Rumah: https://www.kaggle.com/wisnuanggara/daftar-harga-rumah
"""