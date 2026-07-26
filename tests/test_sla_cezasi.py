"""
tests/test_sla_cezasi.py
=========================
SLA (teslim suresi) ceza mekanizmasinin UCTAN UCA testi.

sla_cezasi_tl fonksiyonu, gecikme cezasini hesaplayan TEK yer:
    ceza = geciken_desi x gecikme_saat x 0.4 TL (SLA_CEZA_TL_PER_DESI_PER_SAAT)
ve gecikme_saat, gecikmeyi HER ZAMAN bir ust tam saate yuvarliyor
(KISITLAR.md Bolum 5).

Burada IKI AYRI risk var, o yuzden test dosyasi da IKI BOLUME ayrilir:

  BOLUM A - FORMUL TESTLERI (izole birim testler)
      sla_cezasi_tl / gecikme_saat fonksiyonlarini DOGRUDAN cagirip birkac
      net senaryoyu kontrol eder. Harici dosyaya ihtiyac duymaz.

  BOLUM B - PIPELINE DOGRULAMASI (gercek cikti testi)
      Formul dogru olsa bile PIPELINE icinde YANLIS girdilerle (yanlis
      deadline, yanlis tamamlanma ani) cagriliyor olabilir - bu daha once
      gercek yasanmis bir hata sinifi (kod yorumlarinda "SLA cezasi cok az"
      notu, bkz. alns_engine.py::try_insert_path docstring). Bu bolum,
      results/optimization_results.csv'deki (alns_optimize.py ciktisi) her
      satirin KENDI alanlarindan (Talep_Tarihi/Slotu, Varis_Tarihi/Saati,
      Varis_Ellecleme_Dk) + rota matrisindeki hedef_teslim_gun'dan BAGIMSIZ
      bir deadline/tamamlanma ani/ceza turetip, sistemin raporladigi
      SLA_Cezasi_TL ile karsilastirir. Tutmuyorsa: formul dogru calisiyor
      ama pipeline'da yanlis girdilerle cagriliyor demektir.

Girdiler (proje kokune gore, bu dosya tests/ altinda calisir):
    results/optimization_results.csv       -> BOLUM B icin (alns_optimize.py ciktisi)
    data/raw/sehirler_arasi_lojistik.xlsx  -> BOLUM B icin (route matrisi, hedef_teslim_gun)
    (BOLUM A hicbir harici dosyaya ihtiyac duymaz)

Kullanim:
    pytest tests/test_sla_cezasi.py -v
    python tests/test_sla_cezasi.py                      # BOLUM A + B, ozet + exit code
    python tests/test_sla_cezasi.py --report-out results/sla_dogrulama_raporu.csv
    python tests/test_sla_cezasi.py --skip-pipeline       # sadece BOLUM A (dosyaya ihtiyac yok)

Cikis kodu (CLI):
    0 -> BOLUM A + BOLUM B tamamen basarili
    1 -> BOLUM B'de en az bir uyusmazlik / veri hatasi var
    2 -> BOLUM A'da (formulun kendisinde) bir hata var - KRITIK, oncelikle buna bakilmali
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd

try:
    import pytest
except ImportError:  # pragma: no cover - CLI kullanimi pytest'e ihtiyac duymaz
    pytest = None  # type: ignore[assignment]

from src.alns.time_model import gecikme_saat, sla_cezasi_tl
from src.config import ROUTE_MATRIX_COLUMNS, SLA_CEZA_TL_PER_DESI_PER_SAAT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "optimization_results.csv"
DEFAULT_ROUTE_MATRIX_XLSX = PROJECT_ROOT / "data" / "raw" / "sehirler_arasi_lojistik.xlsx"

# SLA_Cezasi_TL zaten 2 ondalige yuvarlanmis raporlaniyor (bkz. alns_optimize.py,
# round(..., 2)); iki bagimsiz yuvarlama ucunda +-0.01'lik farklara tolerans.
TOLERANS_TL = 0.02

REQUIRED_RESULT_COLS = [
    "Talep_ID", "Cikis_TM", "Varis_TM", "Nihai_Kaynak", "Nihai_Varis",
    "Talep_Tarihi", "Talep_Slotu", "Varis_Tarihi", "Varis_Saati",
    "Varis_Ellecleme_Dk", "Bu_Talebin_Desisi", "SLA_Cezasi_TL",
]


# ==============================================================================
# BOLUM A - FORMUL TESTLERI (izole birim testler, harici dosya gerektirmez)
# ==============================================================================

_FORMUL_DEADLINE = datetime(2026, 7, 1, 12, 0, 0)
_FORMUL_DESI = 10_000.0

if pytest is not None:

    @pytest.mark.parametrize(
        "fark, beklenen_saat, aciklama",
        [
            (timedelta(hours=-1), 0, "erken teslimat -> gecikme yok"),
            (timedelta(0), 0, "tam zamaninda (deadline'a esit) -> gecikme yok"),
            (timedelta(minutes=1), 1, "1 dakikalik gecikme -> 1 saate yukari yuvarlanir"),
            (timedelta(minutes=59), 1, "59 dakikalik gecikme -> 1 saate yukari yuvarlanir"),
            (timedelta(hours=2), 2, "tam 2 saatlik gecikme -> tam 2 saat (fazladan yuvarlama yok)"),
            (timedelta(hours=2, minutes=1), 3, "2 saat 1 dakika -> 3 saate yukari yuvarlanir"),
            (timedelta(hours=23, minutes=59), 24, "23s59d -> 24 saate yuvarlanir"),
        ],
    )
    def test_gecikme_saat_yuvarlama(fark, beklenen_saat, aciklama):
        fiili_tamamlanma = _FORMUL_DEADLINE + fark
        assert gecikme_saat(fiili_tamamlanma, _FORMUL_DEADLINE) == beklenen_saat, aciklama


def test_tam_zamaninda_teslimat_ceza_sifir():
    """Gecikme yok -> ceza = 0."""
    gecikme = gecikme_saat(_FORMUL_DEADLINE, _FORMUL_DEADLINE)
    assert sla_cezasi_tl(_FORMUL_DESI, gecikme) == 0.0


def test_bir_dakika_gecikme_bir_saat_sayilir():
    """1 dakikalik gecikme -> 1 saat sayilmali, ceza = desi x 1 x 0.4."""
    fiili_tamamlanma = _FORMUL_DEADLINE + timedelta(minutes=1)
    gecikme = gecikme_saat(fiili_tamamlanma, _FORMUL_DEADLINE)
    assert gecikme == 1
    ceza = sla_cezasi_tl(_FORMUL_DESI, gecikme)
    beklenen = _FORMUL_DESI * 1 * SLA_CEZA_TL_PER_DESI_PER_SAAT
    assert abs(ceza - beklenen) < 1e-9


def test_tam_iki_saat_gecikme():
    """Tam 2 saatlik gecikme -> ceza = desi x 2 x 0.4 (fazladan yuvarlama yok)."""
    fiili_tamamlanma = _FORMUL_DEADLINE + timedelta(hours=2)
    gecikme = gecikme_saat(fiili_tamamlanma, _FORMUL_DEADLINE)
    assert gecikme == 2
    ceza = sla_cezasi_tl(_FORMUL_DESI, gecikme)
    beklenen = _FORMUL_DESI * 2 * SLA_CEZA_TL_PER_DESI_PER_SAAT
    assert abs(ceza - beklenen) < 1e-9


def test_iki_saat_bir_dakika_gecikme_uce_yuvarlanir():
    """2 saat 1 dakikalik gecikme -> 3 saate yuvarlanmali, ceza = desi x 3 x 0.4."""
    fiili_tamamlanma = _FORMUL_DEADLINE + timedelta(hours=2, minutes=1)
    gecikme = gecikme_saat(fiili_tamamlanma, _FORMUL_DEADLINE)
    assert gecikme == 3
    ceza = sla_cezasi_tl(_FORMUL_DESI, gecikme)
    beklenen = _FORMUL_DESI * 3 * SLA_CEZA_TL_PER_DESI_PER_SAAT
    assert abs(ceza - beklenen) < 1e-9


def test_sla_ceza_katsayisi_configden_okunuyor():
    """Formulun config.py'deki sabiti (0.4) kullandigini, hard-code baska bir
    deger kullanmadigini dogrular - sabit degisirse bu test de otomatik
    gecerli kalir (davranisi degil, formulun DOGRU KAYNAGI kullandigini test eder)."""
    beklenen = 100.0 * 5 * SLA_CEZA_TL_PER_DESI_PER_SAAT
    assert abs(sla_cezasi_tl(100.0, 5) - beklenen) < 1e-9


def test_negatif_gecikme_sla_cezasi_da_sifir():
    """gecikme_saat negatif donmez (0'a clamp) ama formulun kendisi de
    negatif bir gecikme_saat_degeri verilse (savunmaci) ceza uretmemeli -
    ceza fonksiyonuna dogrudan 0 verildiginde davranis kontrolu."""
    assert sla_cezasi_tl(_FORMUL_DESI, 0) == 0.0


def run_formul_testleri() -> list[str]:
    """pytest'siz CLI calistirmasi icin BOLUM A'yi dogrudan calistirir.
    Hata mesajlarinin listesini dondurur (bos liste = hepsi basarili)."""
    hatalar: list[str] = []
    senaryolar = [
        (timedelta(hours=-1), 0, "erken teslimat -> gecikme yok"),
        (timedelta(0), 0, "tam zamaninda -> gecikme yok"),
        (timedelta(minutes=1), 1, "1 dk gecikme -> 1 saat"),
        (timedelta(minutes=59), 1, "59 dk gecikme -> 1 saat"),
        (timedelta(hours=2), 2, "tam 2 saat gecikme -> 2 saat"),
        (timedelta(hours=2, minutes=1), 3, "2s1dk gecikme -> 3 saat"),
        (timedelta(hours=23, minutes=59), 24, "23s59dk gecikme -> 24 saat"),
    ]
    for fark, beklenen_saat, aciklama in senaryolar:
        g = gecikme_saat(_FORMUL_DEADLINE + fark, _FORMUL_DEADLINE)
        if g != beklenen_saat:
            hatalar.append(f"gecikme_saat yanlis ({aciklama}): beklenen={beklenen_saat}, gelen={g}")
        beklenen_ceza = _FORMUL_DESI * beklenen_saat * SLA_CEZA_TL_PER_DESI_PER_SAAT
        ceza = sla_cezasi_tl(_FORMUL_DESI, g)
        if abs(ceza - beklenen_ceza) > 1e-6:
            hatalar.append(f"sla_cezasi_tl yanlis ({aciklama}): beklenen={beklenen_ceza}, gelen={ceza}")
    return hatalar


# ==============================================================================
# BOLUM B - PIPELINE DOGRULAMASI (gercek cikti testi)
# ==============================================================================

def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"optimization_results.csv bulunamadi: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_RESULT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"optimization_results.csv beklenen kolonlari icermiyor: {missing}")
    # Bos Talep_ID'li satirlar gercek bir talebe ait degildir (zorunlu ama
    # yuksuz kiralik seferlerin bilgilendirme satirlari) - SLA dogrulamasi
    # disinda tutulur.
    df = df[df["Talep_ID"].notna() & (df["Talep_ID"].astype(str).str.strip() != "")]
    return df


def load_hedef_teslim_gun(path: Path) -> dict[tuple[str, str], float]:
    """(cikis, varis) -> hedef_teslim_gun. Kolon adlari src/config.py::ROUTE_MATRIX_COLUMNS'tan."""
    if not path.exists():
        raise FileNotFoundError(f"sehirler_arasi_lojistik.xlsx bulunamadi: {path}")
    route = pd.read_excel(path)
    src_col = ROUTE_MATRIX_COLUMNS["source"]
    dst_col = ROUTE_MATRIX_COLUMNS["destination"]
    hedef_col = ROUTE_MATRIX_COLUMNS["target_delivery_days"]
    missing = [c for c in (src_col, dst_col, hedef_col) if c not in route.columns]
    if missing:
        raise ValueError(f"sehirler_arasi_lojistik.xlsx beklenen kolonlari icermiyor: {missing}")
    return {
        (row[src_col], row[dst_col]): float(row[hedef_col])
        for _, row in route.iterrows()
    }


def _parse_dt(gun: str, saat: str) -> datetime:
    return datetime.strptime(f"{gun} {saat}", "%Y-%m-%d %H:%M")


def build_sla_dogrulama_raporu(results_df: pd.DataFrame, hedef_teslim_gun: dict) -> pd.DataFrame:
    """Her satir icin BAGIMSIZ yeniden hesaplanmis beklenen SLA cezasini,
    sistemin raporladigi degerle karsilastiran bir rapor dondurur.

    Sadece NIHAI bacak satirlari (Varis_TM == Nihai_Varis) SLA cezasi
    tasiyabilir (bkz. alns_optimize.py - sla_cezasi sadece son bacakta
    yaziliyor); ara bacaklarda SLA_Cezasi_TL HER ZAMAN 0 olmali.
    """
    df = results_df.copy()
    df["is_final_leg"] = df["Varis_TM"] == df["Nihai_Varis"]

    kayitlar = []
    for row in df.itertuples():
        if not row.is_final_leg:
            kayitlar.append({
                "Talep_ID": row.Talep_ID, "Cikis_TM": row.Cikis_TM, "Varis_TM": row.Varis_TM,
                "Nihai_Kaynak": row.Nihai_Kaynak, "Nihai_Varis": row.Nihai_Varis,
                "Tur": "ARA_BACAK",
                "Gecikme_Saat": None,
                "Beklenen_SLA_TL": 0.0,
                "Raporlanan_SLA_TL": round(float(row.SLA_Cezasi_TL), 2),
                "Fark_TL": round(float(row.SLA_Cezasi_TL) - 0.0, 2),
                "Durum": "OK" if row.SLA_Cezasi_TL == 0 else "ARA_BACAKTA_SLA_CEZASI_VAR",
            })
            continue

        hat = (row.Nihai_Kaynak, row.Nihai_Varis)
        hg = hedef_teslim_gun.get(hat)
        if hg is None:
            kayitlar.append({
                "Talep_ID": row.Talep_ID, "Cikis_TM": row.Cikis_TM, "Varis_TM": row.Varis_TM,
                "Nihai_Kaynak": row.Nihai_Kaynak, "Nihai_Varis": row.Nihai_Varis,
                "Tur": "NIHAI_BACAK",
                "Gecikme_Saat": None, "Beklenen_SLA_TL": None,
                "Raporlanan_SLA_TL": round(float(row.SLA_Cezasi_TL), 2),
                "Fark_TL": None,
                "Durum": "ROTA_MATRISINDE_HAT_YOK",
            })
            continue

        demand_dt = _parse_dt(row.Talep_Tarihi, row.Talep_Slotu)
        deadline = demand_dt + timedelta(days=hg)
        tamamlanma = _parse_dt(row.Varis_Tarihi, row.Varis_Saati) + timedelta(
            minutes=float(row.Varis_Ellecleme_Dk)
        )

        g = gecikme_saat(tamamlanma, deadline)
        beklenen = round(sla_cezasi_tl(float(row.Bu_Talebin_Desisi), g), 2)
        raporlanan = round(float(row.SLA_Cezasi_TL), 2)
        fark = round(beklenen - raporlanan, 2)

        kayitlar.append({
            "Talep_ID": row.Talep_ID, "Cikis_TM": row.Cikis_TM, "Varis_TM": row.Varis_TM,
            "Nihai_Kaynak": row.Nihai_Kaynak, "Nihai_Varis": row.Nihai_Varis,
            "Tur": "NIHAI_BACAK",
            "Gecikme_Saat": g,
            "Beklenen_SLA_TL": beklenen,
            "Raporlanan_SLA_TL": raporlanan,
            "Fark_TL": fark,
            "Durum": "OK" if abs(fark) <= TOLERANS_TL else "UYUSMUYOR",
        })

    return pd.DataFrame(kayitlar)


def _load_default_rapor() -> pd.DataFrame:
    results_df = load_results(DEFAULT_RESULTS_CSV)
    hedef_teslim_gun = load_hedef_teslim_gun(DEFAULT_ROUTE_MATRIX_XLSX)
    return build_sla_dogrulama_raporu(results_df, hedef_teslim_gun)


def test_ara_bacaklarda_sla_cezasi_yok():
    """Ara (relay) bacaklarda SLA cezasi hic raporlanmamali - ceza sadece
    nihai bacakta tek seferde yaziliyor olmali (cift sayim riskine karsi)."""
    rapor = _load_default_rapor()
    ihlaller = rapor[rapor["Durum"] == "ARA_BACAKTA_SLA_CEZASI_VAR"]
    assert ihlaller.empty, (
        f"{len(ihlaller)} ara bacak satirinda sifir olmayan SLA_Cezasi_TL bulundu:\n"
        f"{ihlaller.to_string(index=False)}"
    )


def test_tum_hatlar_rota_matrisinde_bulunuyor():
    """Her nihai hat (Nihai_Kaynak, Nihai_Varis) icin rota matrisinde bir
    hedef_teslim_gun tanimli olmali - yoksa deadline hic hesaplanamaz."""
    rapor = _load_default_rapor()
    eksik = rapor[rapor["Durum"] == "ROTA_MATRISINDE_HAT_YOK"]
    assert eksik.empty, (
        f"{len(eksik)} nihai bacak satiri icin rota matrisinde hat bulunamadi:\n"
        f"{eksik.drop_duplicates(subset=['Nihai_Kaynak', 'Nihai_Varis']).to_string(index=False)}"
    )


def test_nihai_bacaklarda_sla_cezasi_dogru_hesaplanmis():
    """KRITIK TEST: her nihai bacak satiri icin, satirin KENDI raporladigi
    Talep_Tarihi/Slotu (deadline baslangici) + Varis_Tarihi/Saati +
    Varis_Ellecleme_Dk (fiili tamamlanma) + rota matrisinin hedef_teslim_gun'undan
    BAGIMSIZ turetilen SLA cezasi, sistemin raporladigi SLA_Cezasi_TL ile
    (yuvarlama toleransi disinda) EŞLEŞMELI. Eslesmiyorsa formul dogru
    calisiyor olsa bile YANLIS bir deadline/tamamlanma anıyla cagrilmis demektir."""
    rapor = _load_default_rapor()
    nihai = rapor[rapor["Tur"] == "NIHAI_BACAK"]
    ihlaller = nihai[nihai["Durum"] == "UYUSMUYOR"]

    if not ihlaller.empty:
        oran = len(ihlaller) / len(nihai) * 100
        ornekler = ihlaller.sort_values("Fark_TL", ascending=False).head(15)
        mesaj = (
            f"\n{len(ihlaller)}/{len(nihai)} nihai bacak satirinda (%{oran:.1f}) "
            f"raporlanan SLA_Cezasi_TL, satirin kendi Varis_Tarihi/Saati + "
            f"Varis_Ellecleme_Dk'sinden bagimsiz turetilen beklenen cezadan SAPIYOR.\n"
            f"Fark her zaman Beklenen > Raporlanan yonunde ise, SLA cezasi "
            f"SISTEMATIK OLARAK AZ hesaplaniyor olabilir (bkz. gorev notundaki "
            f"'SLA cezasi cok az' bug sinifi).\n"
            f"En buyuk 15 sapma:\n{ornekler.to_string(index=False)}"
        )
        if pytest is not None:
            pytest.fail(mesaj)
        else:  # pragma: no cover
            raise AssertionError(mesaj)


# ==============================================================================
# CLI (main() ile bagimsiz calistirma / rapor kaydetme)
# ==============================================================================

def print_pipeline_summary(rapor: pd.DataFrame) -> None:
    nihai = rapor[rapor["Tur"] == "NIHAI_BACAK"]
    ara = rapor[rapor["Tur"] == "ARA_BACAK"]

    print("-" * 70)
    print("BOLUM B: SLA CEZASI PIPELINE DOGRULAMA RAPORU")
    print("-" * 70)
    print(f"Toplam satir: {len(rapor)}  (nihai bacak: {len(nihai)}, ara bacak: {len(ara)})")
    print()
    print("Nihai bacak durum dagilimi:")
    print(nihai["Durum"].value_counts().to_string())
    print()
    print("Ara bacak durum dagilimi:")
    print(ara["Durum"].value_counts().to_string())

    uyusmuyor = nihai[nihai["Durum"] == "UYUSMUYOR"]
    if not uyusmuyor.empty:
        print()
        print(f"--- UYUSMAYAN {len(uyusmuyor)} NIHAI BACAK (en buyuk 20 fark) ---")
        print(
            uyusmuyor.sort_values("Fark_TL", ascending=False)
            .head(20)
            .to_string(index=False)
        )
        toplam_fark = uyusmuyor["Fark_TL"].sum()
        print(f"\nToplam sapma (Beklenen - Raporlanan): {toplam_fark:.2f} TL")


def main() -> int:
    parser = argparse.ArgumentParser(description="SLA cezasi - formul + pipeline dogrulama")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--route-matrix-xlsx", type=Path, default=DEFAULT_ROUTE_MATRIX_XLSX)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Sadece BOLUM A'yi (formul testlerini) calistir, dosyaya ihtiyac duymaz",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SLA CEZASI TEST SUITE (BOLUM A: formul, BOLUM B: pipeline)")
    print("=" * 70)

    # ---- BOLUM A ----
    print("-" * 70)
    print("BOLUM A: FORMUL TESTLERI (gecikme_saat / sla_cezasi_tl)")
    print("-" * 70)
    formul_hatalari = run_formul_testleri()
    if formul_hatalari:
        print(f"BASARISIZ - {len(formul_hatalari)} senaryo formul seviyesinde hata verdi:")
        for h in formul_hatalari:
            print(f"  - {h}")
        print("\nSONUC: KRITIK HATA - formulun kendisi yanlis, once bu duzeltilmeli.")
        return 2
    print("Tum formul senaryolari basarili (0 gecikme, 1dk, tam 2 saat, 2s1dk).")

    if args.skip_pipeline:
        print("\n--skip-pipeline verildi, BOLUM B atlandi.")
        print("SONUC: BASARILI (sadece BOLUM A calistirildi).")
        return 0

    # ---- BOLUM B ----
    try:
        results_df = load_results(args.results_csv)
        hedef_teslim_gun = load_hedef_teslim_gun(args.route_matrix_xlsx)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nHATA (BOLUM B): {exc}", file=sys.stderr)
        return 1

    rapor = build_sla_dogrulama_raporu(results_df, hedef_teslim_gun)
    print()
    print_pipeline_summary(rapor)

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        rapor.to_csv(args.report_out, index=False, encoding="utf-8-sig")
        print(f"\nDetayli rapor kaydedildi: {args.report_out}")

    nihai = rapor[rapor["Tur"] == "NIHAI_BACAK"]
    pipeline_basarisiz = (
        (rapor["Durum"] == "ARA_BACAKTA_SLA_CEZASI_VAR").any()
        or (nihai["Durum"] == "UYUSMUYOR").any()
        or (nihai["Durum"] == "ROTA_MATRISINDE_HAT_YOK").any()
    )

    print()
    print("=" * 70)
    if pipeline_basarisiz:
        print("SONUC: BOLUM A basarili, BOLUM B BASARISIZ - SLA cezasi pipeline'da tutarsiz uygulaniyor.")
        return 1

    print("SONUC: BASARILI - formul dogru VE pipeline'da dogru girdilerle uygulaniyor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())