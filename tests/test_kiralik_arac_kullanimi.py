"""
Kullanim:
    python tests/test_kiralik_arac_kullanimi.py
    python tests/test_kiralik_arac_kullanimi.py --results-csv results/optimization_results.csv \
        --contract-xlsx data/raw/Kiralik_Araclar.xlsx --sheet-name Sayfa1
    python tests/test_kiralik_arac_kullanimi.py --report-out results/kiralik_dogrulama_raporu.csv
    pytest tests/test_kiralik_arac_kullanimi.py

Cikis kodu (script olarak calistirildiginda):
    0 -> tum kontroller basarili (limit asimi yok; atil kapasite yok)
    1 -> en az bir LIMIT_ASILDI ihlali var (kritik hata)
    2 -> limit asimi yok ama ATIL_KAPASITE var (uyari / verimlilik kaybi)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------

# Bu dosya <proje_kok>/tests/ altinda oldugu icin proje koku bir ust dizin.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "optimization_results.csv"
DEFAULT_CONTRACT_XLSX = PROJECT_ROOT / "data" / "raw" / "Kiralık_Araclar.xlsx"
DEFAULT_SHEET_NAME = "Sayfa1"

KIRALIK_TIPI = "Kiralik"

REQUIRED_RESULT_COLS = ["Tarih", "Arac_Tipi", "Arac_Turu", "Arac_ID", "Cikis_TM", "Varis_TM"]


# ----------------------------------------------------------------------------
# Veri okuma
# ----------------------------------------------------------------------------

def load_optimization_results(path: Path) -> pd.DataFrame:
    """optimization_results.csv dosyasini okur ve gerekli kolonlarin
    var oldugunu dogrular."""
    if not path.exists():
        raise FileNotFoundError(f"optimization_results.csv bulunamadi: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_RESULT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"optimization_results.csv beklenen kolonlari icermiyor: {missing}. "
            f"Bulunan kolonlar: {list(df.columns)}"
        )

    df["Tarih"] = df["Tarih"].astype(str)
    return df


def load_contract_limits(path: Path, sheet_name: str = DEFAULT_SHEET_NAME) -> pd.DataFrame:
    """Kiralik_Araclar.xlsx dosyasini okur ve kolon adlarini
    (Cikis_TM, Varis_TM, Limit, Arac_Turu) olarak standardize eder.

    Once verilen `sheet_name` (varsayilan "Sayfa1") denenir; o sheet
    bulunamazsa dosyadaki ILK sheet'e (index 0) otomatik dusulur, boylece
    sheet adi farkli bir projede de script kirilmaz.
    """
    if not path.exists():
        raise FileNotFoundError(f"Kiralik_Araclar.xlsx bulunamadi: {path}")

    try:
        contract = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError:
        # Istenen sheet adi yoksa ilk sheet'e dus (ornegin sheet adi
        # projede "Sayfa1" degil de baska bir isimse).
        contract = pd.read_excel(path, sheet_name=0, engine="openpyxl")

    if contract.shape[1] != 4:
        raise ValueError(
            f"Kiralik_Araclar.xlsx 4 kolon bekleniyor "
            f"(Cikis TM, Varis TM, Arac sayisi, Arac Turu), bulunan: {contract.shape[1]}"
        )

    contract.columns = ["Cikis_TM", "Varis_TM", "Limit", "Arac_Turu"]
    contract["Limit"] = contract["Limit"].astype(int)
    return contract


# ----------------------------------------------------------------------------
# Dogrulama mantigi
# ----------------------------------------------------------------------------

def build_usage_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Her (Tarih, Cikis_TM, Varis_TM, Arac_Turu) icin FIILEN kullanilan
    (distinct) kiralik arac sayisini hesaplar.

    Not: Ayni Arac_ID ayni gunde birden fazla talebi konsolide edebildigi
    icin satir sayisi degil, distinct Arac_ID sayisi kullanilir.
    """
    kiralik = results_df[results_df["Arac_Tipi"] == KIRALIK_TIPI]

    usage = (
        kiralik.groupby(["Tarih", "Cikis_TM", "Varis_TM", "Arac_Turu"])["Arac_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"Arac_ID": "Kullanilan_Arac_Sayisi"})
    )
    return usage


def build_validation_report(results_df: pd.DataFrame, contract_df: pd.DataFrame) -> pd.DataFrame:
    """Sozlesmeli her (rota, arac_turu) kombinasyonunu, sonuc verisinde
    gorulen HER GUNE genisletip iki kurala gore etiketler:
        - LIMIT_ASILDI  : kullanilan > limit
        - ATIL_KAPASITE : kullanilan < limit
        - OK            : kullanilan == limit
    """
    usage = build_usage_table(results_df)

    all_days = sorted(results_df["Tarih"].unique())
    if not all_days:
        raise ValueError("optimization_results.csv icinde hic tarih bulunamadi.")

    contract_days = (
        contract_df.assign(key=1)
        .merge(pd.DataFrame({"Tarih": all_days, "key": 1}), on="key")
        .drop(columns="key")
    )

    report = contract_days.merge(
        usage, on=["Tarih", "Cikis_TM", "Varis_TM", "Arac_Turu"], how="left"
    )
    report["Kullanilan_Arac_Sayisi"] = report["Kullanilan_Arac_Sayisi"].fillna(0).astype(int)

    def _durum(row: pd.Series) -> str:
        if row["Kullanilan_Arac_Sayisi"] > row["Limit"]:
            return "LIMIT_ASILDI"
        if row["Kullanilan_Arac_Sayisi"] < row["Limit"]:
            return "ATIL_KAPASITE"
        return "OK"

    report["Durum"] = report.apply(_durum, axis=1)
    report["Asim_Arac_Sayisi"] = (report["Kullanilan_Arac_Sayisi"] - report["Limit"]).clip(lower=0)
    report["Atil_Arac_Sayisi"] = (report["Limit"] - report["Kullanilan_Arac_Sayisi"]).clip(lower=0)

    ordered_cols = [
        "Tarih", "Cikis_TM", "Varis_TM", "Arac_Turu",
        "Limit", "Kullanilan_Arac_Sayisi", "Durum",
        "Asim_Arac_Sayisi", "Atil_Arac_Sayisi",
    ]
    report = report[ordered_cols].sort_values(
        by=["Durum", "Tarih", "Cikis_TM", "Varis_TM"],
        key=lambda s: s.map({"LIMIT_ASILDI": 0, "ATIL_KAPASITE": 1, "OK": 2}) if s.name == "Durum" else s,
    ).reset_index(drop=True)

    return report


def check_off_contract_routes(results_df: pd.DataFrame, contract_df: pd.DataFrame) -> pd.DataFrame:
    """Bonus kontrol: sozlesmede olmayan bir (rota, arac_turu) uzerinde
    kiralik arac kullanilmis mi? (Veri tutarliligi kontrolu.)"""
    kiralik = results_df[results_df["Arac_Tipi"] == KIRALIK_TIPI]
    contract_keys = set(zip(contract_df.Cikis_TM, contract_df.Varis_TM, contract_df.Arac_Turu))

    used_keys = kiralik[["Cikis_TM", "Varis_TM", "Arac_Turu"]].drop_duplicates()
    mask = ~used_keys.apply(lambda r: (r.Cikis_TM, r.Varis_TM, r.Arac_Turu) in contract_keys, axis=1)
    return used_keys[mask].reset_index(drop=True)


# ----------------------------------------------------------------------------
# pytest fixture'lari / test fonksiyonlari
# (bu dosya `pytest tests/` ile otomatik toplanir; `python tests/...py`
#  ile calistirildiginda ise asagidaki main() devreye girer)
# ----------------------------------------------------------------------------

def _load_default_report() -> pd.DataFrame:
    results_df = load_optimization_results(DEFAULT_RESULTS_CSV)
    contract_df = load_contract_limits(DEFAULT_CONTRACT_XLSX, sheet_name=DEFAULT_SHEET_NAME)
    return build_validation_report(results_df, contract_df)


def test_sozlesme_limiti_asilmadi() -> None:
    """(a) Hicbir (rota, arac_turu, gun) icin kullanilan kiralik arac
    sayisi sozlesme limitini asmamali. Bu ihlal KRITIK -> test FAIL olmali."""
    report = _load_default_report()
    violations = report[report["Durum"] == "LIMIT_ASILDI"]
    assert violations.empty, (
        f"Sozlesme limiti asilan {len(violations)} kayit bulundu:\n"
        f"{violations.to_string(index=False)}"
    )


def test_tum_kiralik_araclar_kullanildi() -> None:
    """(b) O gun mevcut olan TUM kiralik araclar fiilen kalkmis olmali.
    Bu bir verimlilik kontrolu; ihlal varsa test bilgilendirici sekilde
    FAIL olur (rapor mesaji atil kalan araclari listeler)."""
    report = _load_default_report()
    idle = report[report["Durum"] == "ATIL_KAPASITE"]
    assert idle.empty, (
        f"{idle['Atil_Arac_Sayisi'].sum()} arac-gun atil kalmis "
        f"(maliyeti sifir olan kiralik kapasite kullanilmamis):\n"
        f"{idle.to_string(index=False)}"
    )


# ----------------------------------------------------------------------------
# Raporlama / CLI
# ----------------------------------------------------------------------------

def print_summary(report: pd.DataFrame, off_contract: pd.DataFrame) -> None:
    counts = report["Durum"].value_counts().to_dict()
    total = len(report)

    print("=" * 70)
    print("KIRALIK ARAC KULLANIM DOGRULAMA RAPORU")
    print("=" * 70)
    print(f"Toplam kontrol edilen (rota x arac_turu x gun) satiri: {total}")
    print(f"  OK             : {counts.get('OK', 0)}")
    print(f"  LIMIT_ASILDI   : {counts.get('LIMIT_ASILDI', 0)}")
    print(f"  ATIL_KAPASITE  : {counts.get('ATIL_KAPASITE', 0)}")
    print()

    violations = report[report["Durum"] == "LIMIT_ASILDI"]
    if not violations.empty:
        print("--- (a) SOZLESME LIMITI ASILAN KAYITLAR ---")
        print(violations.to_string(index=False))
        print()

    idle = report[report["Durum"] == "ATIL_KAPASITE"]
    if not idle.empty:
        print("--- (b) ATIL KALAN KIRALIK ARAC KAPASITESI ---")
        print(idle.to_string(index=False))
        print(f"\nToplam atil arac-gun: {idle['Atil_Arac_Sayisi'].sum()}")
        print()

    if not off_contract.empty:
        print("--- UYARI: Sozlesme disi rotada kiralik arac kullanimi ---")
        print(off_contract.to_string(index=False))
        print()

    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiralik arac sozlesme limiti / atil kapasite testi")
    parser.add_argument(
        "--results-csv", type=Path, default=DEFAULT_RESULTS_CSV,
        help=f"optimization_results.csv yolu (varsayilan: {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--contract-xlsx", type=Path, default=DEFAULT_CONTRACT_XLSX,
        help=f"Kiralik_Araclar.xlsx yolu (varsayilan: {DEFAULT_CONTRACT_XLSX})",
    )
    parser.add_argument(
        "--sheet-name", type=str, default=DEFAULT_SHEET_NAME,
        help=f"Kiralik_Araclar.xlsx icindeki sheet adi (varsayilan: {DEFAULT_SHEET_NAME})",
    )
    parser.add_argument(
        "--report-out", type=Path, default=None,
        help="Detayli raporun CSV olarak kaydedilecegi yol (opsiyonel)",
    )
    args = parser.parse_args()

    try:
        results_df = load_optimization_results(args.results_csv)
        contract_df = load_contract_limits(args.contract_xlsx, sheet_name=args.sheet_name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    report = build_validation_report(results_df, contract_df)
    off_contract = check_off_contract_routes(results_df, contract_df)

    print_summary(report, off_contract)

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report_out, index=False, encoding="utf-8-sig")
        print(f"Detayli rapor kaydedildi: {args.report_out}")

    has_limit_violation = (report["Durum"] == "LIMIT_ASILDI").any()
    has_idle_capacity = (report["Durum"] == "ATIL_KAPASITE").any()

    if has_limit_violation:
        print("\nSONUC: BASARISIZ - sozlesme limiti asildi.")
        return 1
    if has_idle_capacity:
        print("\nSONUC: UYARI - limit asimi yok fakat atil kiralik arac kapasitesi var.")
        return 2

    print("\nSONUC: BASARILI - limit asimi yok, tum kiralik araclar kullanilmis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())