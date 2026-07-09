import pandas as pd
import numpy as np
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ============================================================================
# 1. VERİ HAZIRLAMA (DATA MODEL)
# ============================================================================
def create_data_model():
    """
    OR-Tools Routing API'si için gereken veri sözlüğünü (data model) oluşturur.
    Gerçek senaryoda bu kısımlar Excel dosyalarından pd.read_excel ile doldurulur.
    """
    data = {}
    
    # -------------------------------------------------------------------------
    # ÖRNEK VERİ SETİ (Excel'den çekilecek formatta)
    # Node (Düğüm) Mantığı:
    # 0: Hayali (Virtual) Başlangıç/Bitiş Noktası (Depo)
    # 1, 2, 3...: Şehirler/Transfer Merkezleri (İstanbul, Yalova vb.)
    # -------------------------------------------------------------------------
    num_locations = 4 # 0: Depo, 1: İstanbul, 2: Yalova, 3: Eskişehir
    
    # Mesafe Matrisi (Örn: sehirler_arasi_lojistik.xlsx'ten türetilir)
    # data['distance_matrix'][from_node][to_node] -> Kilometre
    data['distance_matrix'] = [
        [0, 0, 0, 0],       # 0 (Sanal Depo) her yere 0 km
        [0, 0, 60, 300],    # 1: İst -> İst=0, İst->Yalova=60, İst->Eskişehir=300
        [0, 60, 0, 240],    # 2: Yalova
        [0, 300, 240, 0],   # 3: Eskişehir
    ]
    
    # Zaman Matrisi (Örn: Araç hızlarına göre hesaplanır, Saat cinsinden * 100 ile int yapılabilir)
    data['time_matrix'] = [
        [0, 0, 0, 0],
        [0, 0, 92, 400],    # 92 = 0.92 saat (TIR için)
        [0, 92, 0, 350],
        [0, 400, 350, 0],
    ]

    # Alım-Teslimat Çiftleri (Pickup & Delivery) - TALEP TAHMİNİ.xlsx
    # Her bir satır bir 'Talep ID'yi temsil eder. [Alınacak_Yer_Node, Teslim_Edilecek_Yer_Node]
    data['pickups_deliveries'] = [
        [1, 2], # D00001: İstanbul'dan al (1), Yalova'ya teslim et (2)
        [1, 3], # D00002: İstanbul'dan al (1), Eskişehir'e teslim et (3)
    ]
    
    # Taleplerin Desi Miktarı
    # Node indexine karşılık gelen desi. Alım noktaları pozitif (+), Teslim noktaları negatif (-)
    data['demands'] = [0, 4, -4, 0] # Örnek: Node 1'den 4 desi alınır, Node 2'ye 4 desi bırakılır.
    
    # Araç Bilgileri - Araç_Kapasite_Maliyet_Saat.xlsx
    data['num_vehicles'] = 3 # Optimizasyonun kullanabileceği maksimum araç filosu
    data['vehicle_capacities'] = [22400, 12000, 5600] # Tır, Kamyon, Kamyonet
    
    # Zaman Pencereleri (Time Windows) - Talep Tamamlama Saati'nden türetilir
    # (Min_Zaman, Max_Zaman). Geç kalırsa SLA cezası kesilir (Şimdilik katı sınır)
    data['time_windows'] = [
        (0, 2400),  # 0: Depo (Her zaman açık)
        (0, 2400),  # 1: İstanbul
        (0, 900),   # 2: Yalova (D00001 için 09:00'a kadar)
        (0, 1700),  # 3: Eskişehir (D00002 için 17:00'a kadar)
    ]

    data['depot'] = 0 # Araçların hayali kalkış noktası
    return data

# ============================================================================
# 2. OPTİMİZASYON MODELİNİ KURMA
# ============================================================================
def main():
    data = create_data_model()

    # Yönlendirme İndeks Yöneticisi (Routing Index Manager)
    # Şehir sayısını, araç sayısını ve başlangıç noktasını tanımlar.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot']
    )

    # Routing Modeli (Asıl Çözücü)
    routing = pywrapcp.RoutingModel(manager)

    # ------------------------------------------------------------------------
    # A. MESAFE BOYUTU (Maliyet hesaplaması için)
    # ------------------------------------------------------------------------
    def distance_callback(from_index, to_index):
        """İki node arasındaki mesafeyi döndürür."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index) # Amacımız bu mesafeyi minimize etmek

    # ------------------------------------------------------------------------
    # B. KAPASİTE BOYUTU (Fiziksel desi kısıtı)
    # ------------------------------------------------------------------------
    def demand_callback(from_index):
        """Node'daki desi talebini döndürür. (Alım pozitif, teslimat negatif)"""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    # Boyut Ekleme: Araçlar kapasitelerini aşamaz
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'], # Her aracın kendi kapasitesi
        True,  # Yük sıfırdan başlar
        'Capacity'
    )

    # ------------------------------------------------------------------------
    # C. ZAMAN PENCERESİ BOYUTU (Teslimat saatleri)
    # ------------------------------------------------------------------------
    def time_callback(from_index, to_index):
        """İki node arasındaki seyahat süresini döndürür."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Boyut Ekleme: Araçların yolda geçirdiği süre
    routing.AddDimension(
        time_callback_index,
        300,  # bekleme süresi toleransı (araç erken varırsa bekleyebilir)
        2400, # bir aracın yapabileceği maksimum süre (Örn: 24 saat)
        False, # Süre kümülatif artar, sıfırlanmaz
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Zaman pencerelerini node'lara uygulama
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # ------------------------------------------------------------------------
    # D. ALIM VE TESLİMAT KISITLARI (Pickup & Delivery)
    # ------------------------------------------------------------------------
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        
        # 1. Kural: Aynı kargoyu alan araç ile teslim eden araç AYNI olmalıdır.
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
        )
        
        # 2. Kural: Alım noktası (Pickup), zaman olarak Teslimat noktasından (Delivery) ÖNCE ziyaret edilmelidir.
        routing.solver().Add(
            time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index)
        )

    # ============================================================================
    # 3. ÇÖZÜCÜ PARAMETRELERİ (META-HEURISTIC ALGORTİMALAR)
    # ============================================================================
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # İlk Çözüm Stratejisi (Initial Solution): Problemin kökünü hızlıca bul
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    
    # Arama Stratejisi (Meta-Heuristic): İlk çözümü al ve daha iyi hale getir
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    search_parameters.time_limit.seconds = 300 # Arama için verilecek süre (gerçekte 300 saniye olacak)
    search_parameters.log_search = True # Çözüm adımlarını konsola yazdır

    # Modeli Çöz
    solution = routing.SolveWithParameters(search_parameters)

    # ============================================================================
    # 4. ÇIKTI YAZDIRMA (Basit Versiyon)
    # ============================================================================
    if solution:
        print("\n✅ Çözüm Bulundu!\n")
        total_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = f'Araç {vehicle_id} için Rota:\n'
            route_distance = 0
            route_load = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                time_var = time_dimension.CumulVar(index)
                
                plan_output += f' Node {node_index} Yük({route_load}) Saat({solution.Min(time_var)}) -> '
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                
            plan_output += f' Node {manager.IndexToNode(index)}\n'
            plan_output += f'Mesafe: {route_distance}km\n'
            print(plan_output)
            total_distance += route_distance
        print(f'Toplam Operasyon Mesafesi: {total_distance}km')
    else:
        print("❌ Çözüm bulunamadı.")

if __name__ == '__main__':
    main()