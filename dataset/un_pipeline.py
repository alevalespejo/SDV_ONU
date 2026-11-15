"""
Pipeline di sola pulizia (Smell Test) e generazione del Report di Verifica
sui voti dell'Assemblea Generale ONU (2014-2025).

Produce ESCLUSIVAMENTE due file:
    1. dataset/un_votes_clean_<start>_<end>.csv -> Dataset pulito pronto per i tuoi colleghi
    2. output/report_pulizia.txt                -> Report completo sulla pulizia + diagnostica di verifica del dataset

Uso (dalla radice del progetto):
    python dataset/un_pipeline.py --input data_raw/UNVotes.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLS = ['rcid', 'ccode', 'Countryname', 'year', 'vote', 'importantvote',
                 'me', 'nu', 'di', 'hr', 'co', 'ec']
OPTIONAL_COLS = ['session', 'date', 'unres', 'short', 'descr']


class UNGACleanAndVerifyPipeline:
    def __init__(self, filepath: str, start_year: int = 2014, end_year: int = 2025):
        self.filepath = filepath
        self.start_year = start_year
        self.end_year = end_year
        self.issue_cols = ['me', 'nu', 'di', 'hr', 'co', 'ec']
        # Codici COW di Stati realmente dissolti (Germania Est/Ovest, Cecoslovacchia, Yemen Sud, Vietnam Sud)
        self.defunct_ccodes = [260, 265, 315, 680, 817]

    def _normalize_columns(self, df_raw: pd.DataFrame):
        schema_fixes = []
        col_map = {c.lower(): c for c in df_raw.columns}
        for req in REQUIRED_COLS + OPTIONAL_COLS:
            if req not in df_raw.columns and req.lower() in col_map:
                orig_name = col_map[req.lower()]
                df_raw = df_raw.rename(columns={orig_name: req})
                schema_fixes.append(f"Rinominata colonna '{orig_name}' -> '{req}'")

        if 'Countryname' not in df_raw.columns and 'country' in df_raw.columns:
            df_raw = df_raw.rename(columns={'country': 'Countryname'})
            schema_fixes.append("Rinominata colonna 'country' -> 'Countryname'")

        if 'year' not in df_raw.columns:
            if 'date' in df_raw.columns:
                df_raw['year'] = pd.to_datetime(df_raw['date'], errors='coerce').dt.year
                schema_fixes.append("Colonna 'year' estratta automaticamente da 'date'")
            elif 'session' in df_raw.columns:
                df_raw['year'] = pd.to_numeric(df_raw['session'], errors='coerce') + 1945
                schema_fixes.append("Colonna 'year' calcolata da 'session' + 1945")

        return df_raw, schema_fixes

    def clean_and_audit(self):
        df_raw = pd.read_csv(self.filepath, low_memory=False)
        raw_cols_original = list(df_raw.columns)
        raw_shape = df_raw.shape

        df_raw, schema_fixes = self._normalize_columns(df_raw)
        missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
        if missing:
            raise ValueError(f"Mancano le colonne obbligatorie {missing}. Trovate: {list(df_raw.columns)}")

        # 1. Filtro finestra temporale
        df = df_raw[(df_raw['year'] >= self.start_year) & (df_raw['year'] <= self.end_year)].copy()
        rows_in_window = len(df)
        years_available = sorted(int(y) for y in df['year'].dropna().unique())

        # 2. Conteggio anomalie prima della pulizia (Smell Test)
        raw_vote_dist = df['vote'].value_counts(dropna=False).to_dict()
        non_members_count = int((df['vote'] == 9).sum())
        absent_count = int((df['vote'] == 8).sum())
        unexpected_codes = int((~df['vote'].isin([1, 2, 3, 8, 9])).sum())
        defunct_rows_count = int(df['ccode'].isin(self.defunct_ccodes).sum())
        defunct_states_found = df[df['ccode'].isin(self.defunct_ccodes)]['Countryname'].unique().tolist()

        # 3. Rimozione righe invalide e duplicati
        df = df[df['vote'] != 9]
        df = df[~df['ccode'].isin(self.defunct_ccodes)]
        duplicates_count = int(df.duplicated(subset=['rcid', 'ccode']).sum())
        df = df.drop_duplicates(subset=['rcid', 'ccode'])

        # Unifica Countryname per ccode (evita che cambi di nome come Turkey -> Turkiye creino 200 stati invece di 193)
        canonical_names = df.groupby('ccode')['Countryname'].last().to_dict()
        df['Countryname'] = df['ccode'].map(canonical_names)

        # 4. Ricodifica numerica con zero significativo
        df['vote_num'] = df['vote'].map({1: 1.0, 2: 0.0, 3: -1.0, 8: np.nan})
        df['is_absent'] = (df['vote'] == 8).astype(int)

        # 5. Gestione NaN su importantvote e flag tematici
        important_nan_count = int(df['importantvote'].isna().sum())
        issue_nan_counts = {col: int(df[col].isna().sum()) for col in self.issue_cols}

        if important_nan_count > 0:
            rcid_consensus = (
                df.dropna(subset=['vote_num'])
                .groupby('rcid')['vote_num']
                .apply(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) > 0 else 1.0)
            )
            contested_rcids = set(rcid_consensus[rcid_consensus < 0.80].index)
            mask_na = df['importantvote'].isna()
            df.loc[mask_na, 'importantvote'] = df.loc[mask_na, 'rcid'].isin(contested_rcids).astype(int)

        for col in self.issue_cols + ['importantvote']:
            df[col] = df[col].fillna(0).astype(int)

        kept_optional_cols = [c for c in OPTIONAL_COLS if c in df.columns]
        final_cols = REQUIRED_COLS + ['vote_num', 'is_absent'] + kept_optional_cols
        df = df[final_cols]

        # 6. Diagnostica di verifica del dataset (per controllare che supporti le analisi e i Finding)
        res_per_year = df.groupby('year')['rcid'].nunique().to_dict()
        imp_per_year = df[df['importantvote'] == 1].groupby('year')['rcid'].nunique().to_dict()
        res_by_issue = {col: int(df[df[col] == 1]['rcid'].nunique()) for col in self.issue_cols}

        # Verifica copertura e statistiche Paesi chiave citati nel Finding (USA, ISR, CHN, RUS, IND, BRA)
        valid_v = df.dropna(subset=['vote_num']).copy()
        maj_vote = valid_v.groupby('rcid')['vote_num'].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        ).rename('maj_vote')
        valid_v = valid_v.merge(maj_vote, on='rcid', how='left')
        valid_v['agree_maj'] = (valid_v['vote_num'] == valid_v['maj_vote']).astype(float)

        key_ccodes = {'USA': 2, 'ISR': 666, 'CHN': 710, 'RUS': 365, 'IND': 750, 'BRA': 140}
        key_stats = {}
        for label, ccode in key_ccodes.items():
            sub_c = valid_v[valid_v['ccode'] == ccode]
            sub_hr = sub_c[sub_c['hr'] == 1]
            key_stats[label] = {
                "country_name_in_db": sub_c['Countryname'].iloc[0] if len(sub_c) > 0 else "ASSENTE",
                "total_votes_cast": int(len(sub_c)),
                "pct_align_un_majority": round(float(sub_c['agree_maj'].mean() * 100), 1) if len(sub_c) > 0 else 0.0,
                "pct_abstain_overall": round(float((sub_c['vote_num'] == 0.0).mean() * 100), 1) if len(sub_c) > 0 else 0.0,
                "pct_abstain_human_rights": round(float((sub_hr['vote_num'] == 0.0).mean() * 100), 1) if len(sub_hr) > 0 else 0.0,
            }

        # Verifica variazione accordo con la Russia pre-2022 (2014-2021) vs post-2022 (2022-2025)
        rus_v = valid_v[valid_v['ccode'] == 365][['rcid', 'vote_num']].drop_duplicates('rcid').rename(columns={'vote_num': 'v_rus'})
        merged_rus = valid_v.merge(rus_v, on='rcid', how='inner')
        merged_rus['agree_rus'] = (merged_rus['vote_num'] == merged_rus['v_rus']).astype(float)
        pre_2022_rus = round(float(merged_rus[merged_rus['year'] <= 2021]['agree_rus'].mean() * 100), 1)
        post_2022_rus = round(float(merged_rus[merged_rus['year'] >= 2022]['agree_rus'].mean() * 100), 1)

        # Verifica partecipazione degli Stati per la matrice Paesi x Voti
        pivot_part = df.pivot_table(index='Countryname', columns='rcid', values='vote_num', aggfunc='first')
        countries_ge_75 = int((pivot_part.notna().mean(axis=1) >= 0.75).sum())

        report = {
            "input_file": str(self.filepath),
            "window_requested": f"{self.start_year}-{self.end_year}",
            "years_available": years_available,
            "raw_shape": raw_shape,
            "rows_in_window_pre_clean": rows_in_window,
            "clean_shape": df.shape,
            "n_countries": int(df['Countryname'].nunique()),
            "n_countries_ge_75_participation": countries_ge_75,
            "n_resolutions": int(df['rcid'].nunique()),
            "raw_columns": raw_cols_original,
            "clean_columns": final_cols,
            "schema_fixes": schema_fixes,
            "raw_vote_distribution": {str(k): int(v) for k, v in raw_vote_dist.items()},
            "removed_non_members_vote9": non_members_count,
            "strategic_absences_vote8": absent_count,
            "unexpected_vote_codes": unexpected_codes,
            "defunct_states_found": defunct_states_found,
            "removed_defunct_rows": defunct_rows_count,
            "duplicates_removed": duplicates_count,
            "importantvote_missing_imputed": important_nan_count,
            "issue_nan_counts": issue_nan_counts,
            "resolutions_per_year": {int(k): int(v) for k, v in res_per_year.items()},
            "important_resolutions_per_year": {int(k): int(v) for k, v in imp_per_year.items()},
            "resolutions_by_issue": res_by_issue,
            "key_powers_check": key_stats,
            "mean_global_agreement_with_russia_pre2022_pct": pre_2022_rus,
            "mean_global_agreement_with_russia_post2022_pct": post_2022_rus,
        }
        return df, report


def write_full_report_txt(r: dict, txt_path: Path):
    years = r["years_available"]
    yr_str = f"{min(years)}-{max(years)}" if years else "Nessun anno"

    lines = [
        "================================================================================",
        " REPORT COMPLETO DI PULIZIA (SMELL TEST) E VERIFICA DATASET - VOTI ONU (UNGA)",
        "================================================================================",
        "",
        "SEZIONE 1: PANORAMICA DIMENSIONI E COPERTURA TEMPORALE",
        f"  - File grezzo analizzato            : {r['input_file']}",
        f"  - Dimensioni dataset grezzo         : {r['raw_shape'][0]:,} righe x {r['raw_shape'][1]} colonne",
        f"  - Finestra temporale richiesta      : {r['window_requested']} (Anni effettivi trovati: {yr_str})",
        f"  - Righe nella finestra pre-pulizia  : {r['rows_in_window_pre_clean']:,} righe",
        f"  - Dimensioni dataset PULITO finale  : {r['clean_shape'][0]:,} righe x {r['clean_shape'][1]} colonne",
        f"  - Stati sovrani unici nel pulito    : {r['n_countries']} Stati ({r['n_countries_ge_75_participation']} con presenza >= 75%)",
        f"  - Risoluzioni uniche (rcid)         : {r['n_resolutions']:,} votazioni nominali",
        "",
        "SEZIONE 2: SCHEMA DELLE COLONNE",
        f"  - Colonne trovate nel grezzo ({len(r['raw_columns'])}) : {', '.join(r['raw_columns'])}",
        f"  - Colonne salvate nel pulito ({len(r['clean_columns'])}) : {', '.join(r['clean_columns'])}",
        f"  - Correzioni strutturali applicate  : {'; '.join(r['schema_fixes']) if r['schema_fixes'] else 'Nessuna (schema gia conforme)'}",
        "",
        "SEZIONE 3: PROBLEMI TROVATI NEI DATI GREZZI E COME SONO STATI RISOLTI (SMELL TEST)",
        f"  - Distribuzione codici 'vote' grezzi nella finestra : {r['raw_vote_distribution']}",
        "",
        f"  [PROBLEMA 1] Stati non membri al momento del voto (vote = 9): {r['removed_non_members_vote9']:,} righe.",
        "   -> SOLUZIONE: Righe eliminate dal dataset (non rappresentano scelte diplomatiche).",
        "",
        "  [PROBLEMA 2] Codifica numerica arbitraria di 'vote' (1=Si', 2=Astensione, 3=No, 8=Assente).",
        "   -> SOLUZIONE: Creata la variabile 'vote_num' su scala simmetrica con zero significativo:",
        "                 +1.0 = Si' (Favorevole), 0.0 = Astensione, -1.0 = No (Contrario).",
        "",
        f"  [PROBLEMA 3] Assenze strategiche / non-voto (vote = 8): {r['strategic_absences_vote8']:,} casi (+ {r['unexpected_vote_codes']} codici anomali).",
        "   -> SOLUZIONE: Codificati come NaN in 'vote_num' e tracciati nella colonna binaria 'is_absent' (1/0)",
        "                 per distinguere chi si astiene alla luce del sole (0.0) da chi diserta l'aula.",
        "",
        f"  [PROBLEMA 4] Trappola degli Stati storici dissolti (codici COW [260, 265, 315, 680, 817]): {r['removed_defunct_rows']} righe ({r['defunct_states_found'] or 'nessuno nella finestra'}).",
        "   -> SOLUZIONE: Rimossi solo gli Stati realmente estinti, preservando i codici di Serbia (345), Yemen (678), Vietnam (816) e Russia (365).",
        "",
        f"  [PROBLEMA 5] Record duplicati sulla chiave primaria (rcid, ccode): {r['duplicates_removed']:,} duplicati.",
        "   -> SOLUZIONE: Eliminati tramite deduplicazione rigorosa sulla coppia (Votazione, Stato).",
        "",
        f"  [PROBLEMA 6] Valori mancanti su 'importantvote' ({r['importantvote_missing_imputed']:,} righe) e sui 6 flag tematici {r['issue_nan_counts']}.",
        "   -> SOLUZIONE: Se 'importantvote' era NaN negli anni recenti, e' stato ricostruito marcando a 1 le risoluzioni",
        "                 ad alta divisione (consenso < 80%); i NaN residui sui flag tematici sono stati imputati a 0.",
        "",
        "SEZIONE 4: DIAGNOSTICA DI VERIFICA PER LE ANALISI E I FINDING GEOPOLITICI",
        f"  - Votazioni totali per anno         : {r['resolutions_per_year']}",
        f"  - Votazioni 'importantvote=1' / anno: {r['important_resolutions_per_year']}",
        f"  - Copertura dei 6 macro-temi (n. risoluzioni): {r['resolutions_by_issue']}",
        f"  - Accordo medio globale con la Russia: Pre-2022 (2014-2021) = {r['mean_global_agreement_with_russia_pre2022_pct']}% | Post-2022 (2022-2025) = {r['mean_global_agreement_with_russia_post2022_pct']}%",
        "",
        "  - Verifica attori chiave (USA, ISR, CHN, RUS, IND, BRA):",
    ]

    for code, st in r["key_powers_check"].items():
        lines.append(
            f"    * {code} ({st['country_name_in_db']}): {st['total_votes_cast']} voti validi | "
            f"Allineamento maggioranza ONU = {st['pct_align_un_majority']}% | "
            f"Astensione totale = {st['pct_abstain_overall']}% | "
            f"Astensione Diritti Umani (hr=1) = {st['pct_abstain_human_rights']}%"
        )

    lines.append("================================================================================")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Pulizia voti ONU e generazione report completo di verifica")
    ap.add_argument('--input', required=True, help="CSV grezzo (es. data_raw/UNVotes.csv)")
    ap.add_argument('--start-year', type=int, default=2014)
    ap.add_argument('--end-year', type=int, default=2025)
    ap.add_argument('--clean-dir', default=str(Path(__file__).resolve().parent))
    ap.add_argument('--results-dir', default='output')
    args = ap.parse_args()

    pipe = UNGACleanAndVerifyPipeline(args.input, args.start_year, args.end_year)
    df_clean, report = pipe.clean_and_audit()

    clean_dir = Path(args.clean_dir)
    res_dir = Path(args.results_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    clean_path = clean_dir / f"un_votes_clean_{args.start_year}_{args.end_year}.csv"
    df_clean.to_csv(clean_path, index=False)

    txt_report_path = res_dir / "report_pulizia.txt"
    write_full_report_txt(report, txt_report_path)

    print(txt_report_path.read_text(encoding="utf-8"))
    print(f"\n[1/2] Dataset pulito salvato in : {clean_path}")
    print(f"[2/2] Report TXT salvato in     : {txt_report_path}")


if __name__ == '__main__':
    main()
