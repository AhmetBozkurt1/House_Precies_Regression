<img width="1072" alt="Ekran Resmi 2024-06-14 14 00 55" src="https://github.com/AhmetBozkurt1/House_Precies_Regression/assets/120393650/567c9d90-116f-4d39-aa92-976ac621abea">

# House Precies Predict Regression

### İŞ PROBLEMİ
☞ Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak, farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.

### VERİ SETİ HİKAYESİ
☞ Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri sizin tahmin etmeniz beklenmektedir.

### PROJE YARIŞMA LİNKİ
☞ Proje yarışma linkine [buradan](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation) ulaşabilirsiniz.

### MODEL OLUŞTURMA
- Veri seti keşfedilir ve özelliklerin analizi yapılır.
- Eksik veriler ve aykırı değerler işlenir.
- Özellik mühendisliği adımlarıyla yeni özellikler türetilir.
- Kategorik değişkenler sayısal formata dönüştürülür.
- Model seçimi yapılır ve hiperparametre optimizasyonu gerçekleştirilir.
- En iyi modelin performansı değerlendirilir.


### Gereksinimler
☞ Bu proje çalıştırılmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- xgboost
- catboost

### Kurulum
☞ Projeyi yerel makinenizde çalıştırmak için şu adımları izleyebilirsiniz:

- GitHub'dan projeyi klonlayın.
- Projeyi içeren dizine gidin ve terminalde `conda env create -f environment.yaml` komutunu çalıştırarak gerekli bağımlılıkları yükleyin.
- Derleyicinizi `conda` ortamına göre ayarlayın.
- Projeyi bir Python IDE'sinde veya Jupyter Notebook'ta açın.
