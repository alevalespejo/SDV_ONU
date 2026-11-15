"""Pulizia dei voti dell'Assemblea Generale ONU (Voeten et al.) per gli anni 2014-2025.

Scrive due file:
    dataset/un_votes_clean_<inizio>_<fine>.csv   dati puliti
    output/report_pulizia.txt                    cosa è stato tolto o ricodificato, più alcuni controlli

Uso, dalla cartella del progetto:
    python dataset/un_pipeline.py --input data_raw/UNVotes.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLONNE = ["rcid", "ccode", "Countryname", "year", "vote", "importantvote",
           "me", "nu", "di", "hr", "co", "ec"]
COLONNE_FACOLTATIVE = ["session", "date", "unres", "short", "descr"]
TEMI = ["me", "nu", "di", "hr", "co", "ec"]
# codici COW di Stati che non esistono più: Germania Ovest ed Est, Cecoslovacchia,
# Yemen del Sud, Vietnam del Sud
STATI_ESTINTI = [260, 265, 315, 680, 817]
PAESI_CONTROLLO = {"USA": 2, "ISR": 666, "CHN": 710, "RUS": 365, "IND": 750, "BRA": 140}


def sistema_colonne(df):
    """Uniforma i nomi delle colonne (maiuscole/minuscole, 'country') e ricava 'year' se manca."""
    note = []
    minuscole = {c.lower(): c for c in df.columns}
    for c in COLONNE + COLONNE_FACOLTATIVE:
        if c not in df.columns and c.lower() in minuscole:
            df = df.rename(columns={minuscole[c.lower()]: c})
            note.append(f"rinominata '{minuscole[c.lower()]}' in '{c}'")
    if "Countryname" not in df.columns and "country" in df.columns:
        df = df.rename(columns={"country": "Countryname"})
        note.append("rinominata 'country' in 'Countryname'")
    if "year" not in df.columns:
        if "date" in df.columns:
            df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
            note.append("'year' ricavato da 'date'")
        elif "session" in df.columns:
            df["year"] = pd.to_numeric(df["session"], errors="coerce") + 1945
            note.append("'year' ricavato da 'session' + 1945")
    return df, note


def pulisci(percorso, inizio=2014, fine=2025):
    grezzo = pd.read_csv(percorso, low_memory=False)
    colonne_grezze = list(grezzo.columns)
    forma_grezza = grezzo.shape
    grezzo, note = sistema_colonne(grezzo)
    mancanti = [c for c in COLONNE if c not in grezzo.columns]
    if mancanti:
        raise ValueError(f"Mancano le colonne {mancanti}. Trovate: {list(grezzo.columns)}")

    # anni richiesti
    df = grezzo[(grezzo.year >= inizio) & (grezzo.year <= fine)].copy()
    r = {"input": str(percorso), "finestra": f"{inizio}-{fine}",
         "anni": sorted(int(y) for y in df.year.dropna().unique()),
         "forma_grezza": forma_grezza, "righe_finestra": len(df),
         "colonne_grezze": colonne_grezze, "note_colonne": note,
         "codici_voto": {str(k): int(v) for k, v in df.vote.value_counts(dropna=False).items()},
         "non_membri": int((df.vote == 9).sum()),
         "assenti": int((df.vote == 8).sum()),
         "codici_strani": int((~df.vote.isin([1, 2, 3, 8, 9])).sum()),
         "righe_estinti": int(df.ccode.isin(STATI_ESTINTI).sum()),
         "estinti": df[df.ccode.isin(STATI_ESTINTI)].Countryname.unique().tolist()}

    # righe da togliere: Stati non ancora membri (9), Stati estinti, duplicati
    df = df[df.vote != 9]
    df = df[~df.ccode.isin(STATI_ESTINTI)]
    r["duplicati"] = int(df.duplicated(subset=["rcid", "ccode"]).sum())
    df = df.drop_duplicates(subset=["rcid", "ccode"])

    # un solo nome per Stato (es. Turkey / Türkiye), altrimenti risultano più di 193 Stati
    df["Countryname"] = df.ccode.map(df.groupby("ccode").Countryname.last().to_dict())

    # sì = 1, astenuto = 0, no = -1; assente = NaN, segnato a parte in is_absent
    df["vote_num"] = df.vote.map({1: 1.0, 2: 0.0, 3: -1.0, 8: np.nan})
    df["is_absent"] = (df.vote == 8).astype(int)

    # importantvote mancante negli anni recenti: 1 se la votazione è contesa
    # (il voto più frequente raccoglie meno dell'80% dei votanti)
    r["important_mancanti"] = int(df.importantvote.isna().sum())
    r["temi_mancanti"] = {c: int(df[c].isna().sum()) for c in TEMI}
    if r["important_mancanti"] > 0:
        consenso = (df.dropna(subset=["vote_num"]).groupby("rcid").vote_num
                    .apply(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) > 0 else 1.0))
        contese = set(consenso[consenso < 0.80].index)
        na = df.importantvote.isna()
        df.loc[na, "importantvote"] = df.loc[na, "rcid"].isin(contese).astype(int)
    for c in TEMI + ["importantvote"]:
        df[c] = df[c].fillna(0).astype(int)

    colonne = COLONNE + ["vote_num", "is_absent"] + [c for c in COLONNE_FACOLTATIVE if c in df.columns]
    df = df[colonne]
    r["forma_pulita"] = df.shape
    r["colonne_pulite"] = colonne
    r["n_stati"] = int(df.Countryname.nunique())
    r["n_votazioni"] = int(df.rcid.nunique())
    r.update(controlli(df))
    return df, r


def controlli(df):
    """Numeri di controllo: votazioni per anno e per tema, presenza, alcuni paesi."""
    c = {"per_anno": {int(k): int(v) for k, v in df.groupby("year").rcid.nunique().items()},
         "importanti_per_anno": {int(k): int(v) for k, v in
                                 df[df.importantvote == 1].groupby("year").rcid.nunique().items()},
         "per_tema": {t: int(df[df[t] == 1].rcid.nunique()) for t in TEMI}}

    v = df.dropna(subset=["vote_num"]).copy()
    moda = v.groupby("rcid").vote_num.agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    v = v.merge(moda.rename("maj_vote"), on="rcid", how="left")
    v["agree_maj"] = (v.vote_num == v.maj_vote).astype(float)

    paesi = {}
    for sigla, cc in PAESI_CONTROLLO.items():
        s = v[v.ccode == cc]
        hr = s[s.hr == 1]
        paesi[sigla] = {"nome": s.Countryname.iloc[0] if len(s) else "ASSENTE",
                        "voti": int(len(s)),
                        "maggioranza": round(float(s.agree_maj.mean() * 100), 1) if len(s) else 0.0,
                        "astensione": round(float((s.vote_num == 0).mean() * 100), 1) if len(s) else 0.0,
                        "astensione_hr": round(float((hr.vote_num == 0).mean() * 100), 1) if len(hr) else 0.0}
    c["paesi"] = paesi

    # accordo medio con la Russia prima e dopo il 2022
    rus = v[v.ccode == 365][["rcid", "vote_num"]].drop_duplicates("rcid").rename(columns={"vote_num": "v_rus"})
    m = v.merge(rus, on="rcid", how="inner")
    m["agree_rus"] = (m.vote_num == m.v_rus).astype(float)
    c["russia_prima"] = round(float(m[m.year <= 2021].agree_rus.mean() * 100), 1)
    c["russia_dopo"] = round(float(m[m.year >= 2022].agree_rus.mean() * 100), 1)

    presenza = df.pivot_table(index="Countryname", columns="rcid", values="vote_num", aggfunc="first")
    c["stati_75"] = int((presenza.notna().mean(axis=1) >= 0.75).sum())
    return c


def scrivi_report(r, percorso):
    anni = f"{min(r['anni'])}-{max(r['anni'])}" if r["anni"] else "nessuno"
    riga = "=" * 80
    t = [riga,
         " REPORT DI PULIZIA - VOTI ASSEMBLEA GENERALE ONU",
         riga, "",
         "1. DIMENSIONI",
         f"  - File grezzo                  : {r['input']}",
         f"  - Dataset grezzo               : {r['forma_grezza'][0]:,} righe x {r['forma_grezza'][1]} colonne",
         f"  - Anni richiesti / trovati     : {r['finestra']} / {anni}",
         f"  - Righe negli anni richiesti   : {r['righe_finestra']:,}",
         f"  - Dataset pulito               : {r['forma_pulita'][0]:,} righe x {r['forma_pulita'][1]} colonne",
         f"  - Stati                        : {r['n_stati']} ({r['stati_75']} presenti in almeno il 75% delle votazioni)",
         f"  - Votazioni (rcid)             : {r['n_votazioni']:,}",
         "",
         "2. COLONNE",
         f"  - Nel grezzo ({len(r['colonne_grezze'])})   : {', '.join(r['colonne_grezze'])}",
         f"  - Nel pulito ({len(r['colonne_pulite'])})   : {', '.join(r['colonne_pulite'])}",
         f"  - Correzioni ai nomi           : {'; '.join(r['note_colonne']) or 'nessuna'}",
         "",
         "3. PROBLEMI E DECISIONI",
         f"  - Codici di 'vote' negli anni richiesti: {r['codici_voto']}",
         "",
         f"  a) Stati non ancora membri (vote = 9): {r['non_membri']:,} righe -> tolte.",
         "  b) 'vote' è un codice (1 sì, 2 astenuto, 3 no, 8 assente), non una scala",
         "     -> nuova colonna vote_num: +1 sì, 0 astenuto, -1 no.",
         f"  c) Assenti (vote = 8): {r['assenti']:,} righe, più {r['codici_strani']} codici non previsti",
         "     -> NaN in vote_num e 1 in is_absent: un'assenza non è un'astensione.",
         f"  d) Stati che non esistono più (COW {STATI_ESTINTI}): {r['righe_estinti']} righe"
         f" ({', '.join(r['estinti']) or 'nessuno in questi anni'}) -> tolte.",
         "     Restano Serbia (345), Yemen (678), Vietnam (816) e Russia (365).",
         f"  e) Duplicati su (rcid, ccode): {r['duplicati']:,} -> tolti.",
         f"  f) importantvote mancante: {r['important_mancanti']:,} righe; temi mancanti: {r['temi_mancanti']}",
         "     -> importantvote = 1 se il voto più frequente ha meno dell'80% dei votanti; temi mancanti = 0.",
         "",
         "4. CONTROLLI",
         f"  - Votazioni per anno           : {r['per_anno']}",
         f"  - Con importantvote = 1        : {r['importanti_per_anno']}",
         f"  - Votazioni per tema           : {r['per_tema']}",
         f"  - Accordo medio con la Russia  : 2014-2021 {r['russia_prima']}% | 2022-2025 {r['russia_dopo']}%",
         "  - Alcuni paesi (voti espressi, % con la maggioranza, % astensioni, % astensioni sui diritti umani):"]
    for sigla, p in r["paesi"].items():
        t.append(f"    {sigla} ({p['nome']}): {p['voti']} voti | maggioranza {p['maggioranza']}% | "
                 f"astensione {p['astensione']}% | astensione hr {p['astensione_hr']}%")
    t.append(riga)
    percorso.write_text("\n".join(t), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Pulizia dei voti ONU")
    ap.add_argument("--input", required=True, help="CSV grezzo, es. data_raw/UNVotes.csv")
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--clean-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--results-dir", default="output")
    a = ap.parse_args()

    df, r = pulisci(a.input, a.start_year, a.end_year)
    Path(a.clean_dir).mkdir(parents=True, exist_ok=True)
    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    pulito = Path(a.clean_dir) / f"un_votes_clean_{a.start_year}_{a.end_year}.csv"
    df.to_csv(pulito, index=False)
    report = Path(a.results_dir) / "report_pulizia.txt"
    scrivi_report(r, report)
    print(report.read_text(encoding="utf-8"))
    print(f"\ndati puliti: {pulito}\nreport: {report}")


if __name__ == "__main__":
    main()
