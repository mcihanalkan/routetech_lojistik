import pulp

# 1. Modeli Başlatma
# Problemi maliyet minimize etme odaklı tanımlıyoruz
model = pulp.LpProblem("Lojistik_Maliyet_Optimizasyonu", pulp.LpMinimize)

# Örnek İndeks Setleri (Eray veya Ebubekir'den gelecek veriler için)
hatlar = ["TM1-TM2", "TM2-TM3", "TM1-TM4"]
gunler = ["Pazartesi", "Sali", "Carsamba"]
arac_turleri = ["Tir", "Kamyon"]

# 2. Karar Değişkenlerinin Tanımlanması (LpVariable.dicts)
# kiralik_x: Belirli bir hatta, günde ve araç türünde kaç kiralık araç kullanılacağı
kiralik_x = pulp.LpVariable.dicts(
    "kiralik_x", 
    (hatlar, gunler, arac_turleri), 
    lowBound=0, # eksi değer alamaz
    cat='Integer' # tam sayı olmalı
)

# spot_y: Belirli bir hatta, günde ve araç türünde kaç spot araç kullanılacağı
spot_y = pulp.LpVariable.dicts(
    "spot_y", 
    (hatlar, gunler, arac_turleri), 
    lowBound=0, 
    cat='Integer'
)

# Örnek Kullanım:
# print(kiralik_x["TM1-TM2"]["Pazartesi"]["Tir"])