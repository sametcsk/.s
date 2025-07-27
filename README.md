Proje, aşağıdaki ana adımlar izlenerek gerçekleştirilmiştir:

1. Veri Yükleme ve Genel Bakış
insurance.csv dosyası Pandas kütüphanesi kullanılarak yüklendi.

df.head(), df.info(), df.describe() gibi komutlarla veri setinin yapısı, veri tipleri, eksik değerler ve istatistiksel özetleri incelendi.

Veri setinde eksik değer bulunmadığı teyit edildi.

2. Veri Ön İşleme ve Özellik Mühendisliği
Özellik ve Hedef Ayırma: charges (ücretler) sütunu hedef değişken (y) olarak ayrılırken, diğer tüm sütunlar özellikler (X) olarak belirlendi.

Kategorik ve Sayısal Sütunların Ayrımı: Veri setindeki kategorik (object tipi) ve sayısal (float, int tipi) sütunlar otomatik olarak tespit edildi.

sex, smoker, region sütunları kategorik olarak belirlendi.

age, bmi, children sütunları sayısal olarak belirlendi.

Sayısal Kolonların Korelasyon Analizi: seaborn.heatmap kullanılarak sayısal kolonlar arasındaki korelasyonlar görselleştirildi. Bu adım, özellikler arası ilişkileri ve hedef değişkenle olası bağlantıları anlamak için önemlidir.

Veri Dönüştürme (ColumnTransformer):

Sayısal Sütunlar: StandardScaler kullanılarak sayısal özellikler ölçeklendirildi. Bu, farklı ölçeklerdeki özelliklerin modellere eşit katkıda bulunmasını sağlar ve özellikle mesafe tabanlı algoritmalar (KNN, SVR) için kritiktir.

Kategorik Sütunlar: OneHotEncoder kullanılarak kategorik özellikler sayısal (ikili) formata dönüştürüldü. Bu, makine öğrenimi modellerinin bu özellikleri anlayabilmesini sağlar. handle_unknown='ignore' parametresi, eğitim setinde görülmeyen yeni kategorilerin test setinde oluşması durumunda hatayı önler.

Eğitim ve Test Setlerine Ayırma: Veri seti, %80 eğitim ve %20 test oranıyla train_test_split fonksiyonu kullanılarak bölündü. random_state=42 kullanılarak sonuçların tekrarlanabilirliği sağlandı.

Dönüşüm Uygulaması: Oluşturulan preprocesser objesi, fit_transform metodu ile eğitim setine uygulandı ve test setine sadece transform metodu ile uygulandı.

3. Regresyon Modellerinin Eğitimi ve Kıyaslaması
Bu bölümde, her model için hem varsayılan ayarlarla hem de hiperparametre optimizasyonu sonrası performans değerlendirmesi yapıldı.

3.1. Varsayılan (Default) Ayarlı Modeller
Her bir regresyon modeli (Linear Regression, KNN Regressor, SVR Regressor, Decision Tree Regressor), varsayılan parametreleriyle eğitildi ve test seti üzerinde aşağıdaki performans metrikleri hesaplandı:

R2 Skoru (Coefficient of Determination): Modelin bağımlı değişkendeki varyansı ne kadar iyi açıkladığını gösterir (0 ile 1 arası, 1 en iyidir).

MAE (Mean Absolute Error): Tahminlerin gerçek değerlerden ortalama mutlak sapmasını gösterir (daha düşük, daha iyidir).

MSE (Mean Squared Error): Hataların karelerinin ortalamasını gösterir ve büyük hataları daha fazla cezalandırır (daha düşük, daha iyidir).

Elde edilen sonuçlar default_model_results sözlüğünde saklandı.

3.2. Hiperparametre Ayarı Yapılmış (Optimize Edilmiş) Modeller
Her model için (Linear Regression hariç, çünkü ana hiperparametreleri yoktur) GridSearchCV kullanılarak en uygun hiperparametre kombinasyonları arandı. GridSearchCV, belirtilen hiperparametre aralıklarında tüm olası kombinasyonları dener ve çapraz doğrulama (cross-validation) ile en iyi performans veren kombinasyonu bulur.

Linear Regression: Hiperparametre ayarlamasına ihtiyaç duymadığı için, varsayılan performansı optimize edilmiş sonuçlarına dahil edildi.

K-Nearest Neighbors (KNN) Regressor: n_neighbors, weights, p gibi parametreler için optimizasyon yapıldı.

Support Vector Regressor (SVR): C, epsilon, gamma, kernel gibi kritik parametreler için geniş bir arama yapıldı.

Decision Tree Regressor: max_depth, min_samples_leaf, min_samples_split, criterion gibi parametreler için optimizasyon yapıldı.

Optimizasyon sonrası en iyi modeller (best_estimator_) test seti üzerinde değerlendirildi ve sonuçlar optimized_model_results sözlüğünde saklandı. Ayrıca, her modelin en iyi hiperparametreleri (best_params_) de kaydedildi.

4. Performans Tabloları ve Görselleştirmeler
Hesaplanan tüm performans metrikleri (R2, MAE, MSE) Pandas DataFrame'lerine dönüştürüldü ve okunabilir tablolar halinde sunuldu.

Ayrıca, aşağıdaki kıyaslama grafikleri matplotlib ve seaborn kullanılarak oluşturuldu:

Varsayılan Ayarlı Modellerin R2 Skoru Karşılaştırması: Modellerin başlangıçtaki R2 performansını gösterir.

Optimize Edilmiş Modellerin R2 Skoru Karşılaştırması: Modellerin optimizasyon sonrası R2 performansını gösterir.

Modellerin R2 Skorlarındaki Değişim (Varsayılan vs Optimize Edilmiş): Hiperparametre ayarlamasının her bir modelin R2 skorunu nasıl iyileştirdiğini gösteren kritik bir grafik.

Modellerin MAE Skorlarındaki Değişim (Varsayılan vs Optimize Edilmiş): Ortalama mutlak hata açısından optimizasyonun etkisini gösterir.

Modellerin MSE Skorlarındaki Değişim (Varsayılan vs Optimize Edilmiş): Hata karelerinin ortalaması açısından optimizasyonun etkisini gösterir.

Ana Bulgularım ve Hangi Model Daha İyiydi?
Bu projenin en önemli çıkarımı, hiperparametre optimizasyonunun makine öğrenimi modellerinin tahmin kalitesi üzerinde devasa bir etki yarattığıdır.

Linear Regression: Basit yapısı nedeniyle hiperparametre ayarlamasına ihtiyaç duymamış ve performansı sabit kalmıştır. Diğer optimize edilmiş modellerin gerisinde kalmıştır.

Support Vector Regressor (SVR): Projenin en şaşırtıcı modeli SVR oldu. Varsayılan ayarlarla performansı düşüktü ve başlangıçta umut vermiyordu. Ancak, kapsamlı hiperparametre ayarlaması sonrası performansında muazzam bir artış gösterdi ve KNN ile birlikte en iyi modeller arasına girdi. Bu, SVR'nin potansiyelini ortaya çıkarmak için doğru ayarlamanın ne kadar hayati olduğunu gösterdi.

K-Nearest Neighbors (KNN) Regressor ve Decision Tree Regressor: Her iki model de varsayılan ayarlarla iyi bir başlangıç yapmış olsalar da, hiperparametre optimizasyonu ile performansları daha da iyileşerek en yüksek R2 skorlarına ve en düşük hata metriklerine ulaşan modeller arasına girdiler. Özellikle düşük MAE ve MSE değerleri ile tahminlerinin gerçek değerlere oldukça yakın olduğunu gösterdiler.

Sonuç olarak, bu sağlık sigortası ücreti tahmin probleminde, doğru hiperparametre optimizasyonu yapıldığında hem K-Nearest Neighbors (KNN) Regressor hem de Support Vector Regressor (SVR) modelleri, en yüksek tahmin performansını sergilemiştir. Bu proje, veri biliminde sadece doğru algoritmayı seçmenin değil, aynı zamanda o algoritmanın hiperparametrelerini veri setine en uygun şekilde ayarlamanın ne kadar kritik olduğunu bir kez daha güçlü bir şekilde kanıtlamıştır.
