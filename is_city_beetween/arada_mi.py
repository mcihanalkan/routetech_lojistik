def is_city_between(gidis_tm, varis_tm, aday_tm, mesafe_matrisi, tolerans=0.15):
    """
    Bir aday şehrin, gidiş ve varış şehirleri arasında mantıklı bir 
    rota üzerinde olup olmadığını kontrol eder.
    
    Parametreler:
    - gidis_tm (str): Başlangıç Transfer Merkezi
    - varis_tm (str): Hedef Transfer Merkezi
    - aday_tm (str): Uğranması planlanan aday Transfer Merkezi
    - mesafe_matrisi (DataFrame): Haversine ile hesaplanmış şehirler arası mesafe tablosu
    - tolerans (float): Kabul edilebilir maksimum yol uzama oranı (Örn: 0.15 -> %15)
    
    Dönüş:
    - bool: Eğer şehir rotanın üzerindeyse True, aksi halde False
    """
    
    # 1. Aday şehir gidiş veya varış şehri ile aynıysa direkt False döndür
    if aday_tm == gidis_tm or aday_tm == varis_tm:
        return False
        
    # 2. Mesafeleri matristen çek
    direkt_mesafe = mesafe_matrisi.loc[gidis_tm, varis_tm]
    sapma_mesafesi = mesafe_matrisi.loc[gidis_tm, aday_tm] + mesafe_matrisi.loc[aday_tm, varis_tm]
    
    # 3. Maksimum kabul edilebilir mesafeyi hesapla
    maksimum_kabul_edilebilir_mesafe = direkt_mesafe * (1 + tolerans)
    
    # 4. Kontrolü yap ve sonucu döndür
    if sapma_mesafesi <= maksimum_kabul_edilebilir_mesafe:
        return True
    else:
        return False