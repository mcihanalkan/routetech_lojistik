import pandas as pd
from pathlib import Path
from config import PAYLOAD_CSV, RENTED_STOKS_CSV, CAR_PARAMS_CSV
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "haversine"))
from haversine import GetDistanceMatrixAsList, GetCenters, is_city_between

class ModelData: 
    def __init__(self, centers, distances, center_matrix_df, tm_index):
        pass

class DataLoader:
    def __init__(self):
        self.centers = GetCenters()
        self.distances = GetDistanceMatrixAsList()
        self.center_matrix_df = pd.DataFrame(
            self.distances, 
            index=self.centers, 
            columns=self.centers
        )
        self.tm_index = {tm: idx for idx, tm in enumerate(self.centers)}
        
    def load_demand(self):
        """Talep verilerini yükle"""
        if not PAYLOAD_CSV.exists():
            return self._create_sample_demand()
        
        df = pd.read_csv(PAYLOAD_CSV)
        demand = {}
        hatlar = []
        gunler = set()
        
        for _, row in df.iterrows():
            source = row.get('source_tm') or row.get('kaynak_tm') or row.get('source')
            dest = row.get('destination_tm') or row.get('varis_tm') or row.get('destination')
            
            if pd.isna(source): source = row.iloc[1]
            if pd.isna(dest): dest = row.iloc[2]
            
            hat = f"{source}-{dest}"
            hatlar.append(hat)
            
            recommended = float(row.get('recommended_demand', row.get('q50', 0)))
            date_str = str(row['date']) if 'date' in row else str(row.iloc[0])
            tarih_obj = pd.to_datetime(date_str)
            gun = f"{tarih_obj.day:02d}_Mayis"
            gunler.add(gun)
            
            demand[(hat, gun)] = int(recommended)
        
        return {
            'demand': demand,
            'hatlar': list(dict.fromkeys(hatlar)),
            'gunler': sorted(list(gunler))
        }
    
    def load_rental_stocks(self):
        """Kiralık stok verilerini yükle"""
        if not RENTED_STOKS_CSV.exists():
            return {}
        
        df = pd.read_csv(RENTED_STOKS_CSV)
        stocks = {}
        for _, row in df.iterrows():
            stocks[(row['route'], row['vehicle_type'])] = int(row['quantity'])
        return stocks
    
    def load_vehicle_params(self):
        """Araç parametrelerini yükle"""
        if not CAR_PARAMS_CSV.exists():
            return {}
        
        df = pd.read_csv(CAR_PARAMS_CSV)
        params = {}
        for _, row in df.iterrows():
            params[row['vehicle_type']] = {
                "sabit_kira": int(row['sabit_kira']),
                "kiralik_km_maliyet": int(row['kiralik_km_maliyet']),
                "spot_sabit_maliyet": int(row['spot_sabit_maliyet']),
                "spot_km_maliyet": int(row['spot_km_maliyet']),
                "kapasite_desi": int(row['kapasite_desi']),
            }
        return params
    
    def _create_sample_demand(self):
        """Örnek talep verisi oluştur (dosya yoksa)"""
        hatlar = [f"{tm1}-{tm2}" for tm1 in self.centers for tm2 in self.centers if tm1 != tm2]
        gunler = [f"{i:02d}_Mayis" for i in range(11, 18)]
        demand = {(h, g): 15000 for h in hatlar for g in gunler}
        return {'demand': demand, 'hatlar': hatlar, 'gunler': gunler}
