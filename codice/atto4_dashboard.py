"""Dati della dashboard (slide 31).

La scheda paese dei casi di studio, calcolata per tutti i 172 Stati presenti in almeno
il 75% delle votazioni ed esportata in un JSON letto dalla dashboard. Controlla che i
valori coincidano con quelli di atto3_livello3_casi.py.
Output: output/atto4_dashboard/ (dati_dashboard.json, report)
"""
import json
import numpy as np
import pandas as pd
import comune as c

OUT = c.cartella_atto("atto4_dashboard")
L1, L3 = c.OUTPUT / "atto3_livello1_mondo", c.OUTPUT / "atto3_livello3_casi"
df, P, W, meta = c.dati()
CL = pd.read_csv(L1 / "clusters.csv", index_col=0)
PC = pd.read_csv(L1 / "pcs.csv", index_col=0)
K = pd.read_csv(L1 / "kappa.csv", index_col=0)
R = c.Report("REPORT ATTO 4 - STRUMENTO: DATI PER LA DASHBOARD (SCHEDA PAESE)")
NAM = [i for i in W.columns if i in c.NAM]
anni = list(range(2014, 2026))
Wn, mn = W[meta.solo_usa == 0], meta[meta.solo_usa == 0]
PERIODI = {"2014-21": range(2014, 2022), "2022-25": range(2022, 2026)}
STATI = list(CL.index)

# ------------------------------------------------- calcolo vettorizzato
acc_anno = {}
for a in anni:
    sub = Wn[mn.year == a]
    for p in list(c.POTENZE):
        acc_anno[(a, p)] = c.accordo(sub, p).reindex(STATI)
        if p in STATI:
            acc_anno[(a, p)].loc[p] = np.nan          # una potenza non ha 'accordo con se stessa'
    acc_anno[(a, "MAJ")] = c.accordo_maggioranza(sub, mn[mn.year == a]).reindex(STATI)
segue = {}
for per, yy in PERIODI.items():
    for p in c.POTENZE:
        s, n = c.segue_in_minoranza(W[meta.year.isin(yy)], meta[meta.year.isin(yy)], p)
        segue[(per, p)] = (s.reindex(STATI), float(s[NAM].median()), n)
prof = {}
for k in c.TEMI:
    sub = Wn[mn[k] == 1]
    prof[k] = ((sub == 1).where(sub.notna()).astype(float).mean(), (sub == 0).where(sub.notna()).astype(float).mean())


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


schede = {}
for i in STATI:
    schede[i] = {
        "nome": c.nome(i) if i in c.NOMI else P.loc[i, "Countryname"].title(),
        "membro_nam": bool(i in c.NAM), "continente": P.loc[i, "continente"],
        "cluster": CL.loc[i, "gruppo"], "comunita": CL.loc[i, "comunita"],
        "pc1": r3(PC.loc[i, "PC1"]), "pc2": r3(PC.loc[i, "PC2"]),
        "assenza": r3(W[i].isna().mean()),
        "accordo_anno": {lab: [r3(acc_anno[(a, p)].get(i)) for a in anni]
                         for p, lab in [("USA", "USA"), ("CHN", "Cina"), ("RUS", "Russia"), ("MAJ", "Maggioranza")]},
        "segue_in_minoranza": {f"{c.nome(p)} {per}": r3(segue[(per, p)][0].get(i)) for (per, p) in segue},
        "profilo_si": {c.TEMI[k]: r3(prof[k][0][i]) for k in c.TEMI},
        "profilo_astenuto": {c.TEMI[k]: r3(prof[k][1][i]) for k in c.TEMI},
        "vicini": [{"iso3": j, "nome": c.nome(j) if j in c.NOMI else P.loc[j, "Countryname"].title(), "kappa": r3(v)}
                   for j, v in K[i].drop(i).sort_values(ascending=False).head(5).items()],
    }
riferimenti = {
    "anni": anni,
    "membro_mediano_segue": {f"{c.nome(p)} {per}": r3(v[1]) for (per, p), v in segue.items()},
    "membro_mediano_profilo_si": {c.TEMI[k]: r3(prof[k][0][NAM].median()) for k in c.TEMI},
    "note": ["Accordo per anno e profilo calcolati senza le votazioni 'USA contro tutti'.",
             "Segue in minoranza: votazioni in cui la potenza vota diversamente dalla maggioranza.",
             "Vicini: kappa di Cohen su tutte le votazioni 2014-2025 ('USA contro tutti' escluse)."],
}
with open(OUT / "dati_dashboard.json", "w", encoding="utf-8") as f:
    json.dump({"riferimenti": riferimenti, "paesi": schede}, f, ensure_ascii=False)

# ------------------------------------------------------------ report
R.sezione("Contenuto del file per la dashboard")
R.voce("File", "dati_dashboard.json")
R.voce("Dimensione", f"{(OUT / 'dati_dashboard.json').stat().st_size / 1024:.0f} KB")
R.voce("Paesi", f"{len(schede)} (presenti in >= 75% delle votazioni; {sum(s['membro_nam'] for s in schede.values())} membri del Movimento)")
R.voce("Campi per paese", ", ".join(next(iter(schede.values())).keys()))
R.lettura("La dashboard segue il mantra di Shneiderman: panoramica (mappa PCA/cluster), filtro "
          "(membri / non membri, cluster), dettaglio su richiesta (scheda paese).")

R.sezione("Verifica: i valori coincidono con il livello 3")
ris3 = json.load(open(L3 / "risultati.json", encoding="utf-8"))
ok = True
for i in ["IND", "ZAF", "MLI"]:
    for per, p in [("2014-21", "RUS"), ("2022-25", "RUS"), ("2022-25", "CHN")]:
        v4 = schede[i]["segue_in_minoranza"][f"{c.nome(p)} {per}"]; v3 = ris3[i]["segue"][f"{p}_{per}"]
        uguale = abs(v4 - v3) < 1e-3; ok &= uguale
        R.voce(f"{c.nome(i)}: segue {c.nome(p)} {per}", f"dashboard {v4:.3f} | livello 3 {v3:.3f} | {'OK' if uguale else 'DIVERSO'}")
R.voce("Esito", "tutti i valori coincidono" if ok else "ATTENZIONE: valori diversi")

R.sezione("Progetto della dashboard (slide 31)")
for k, v in [("Utente", "chi vuole sapere come vota un paese all'ONU (studenti, giornalisti, diplomatici)"),
             ("Domanda", "questo paese segue una potenza o vota in modo autonomo?"),
             ("Panoramica", "scatter PCA dei paesi colorati per cluster, con ricerca per nome"),
             ("Filtri", "membri / non membri; cluster; periodo 2014-21 / 2022-25"),
             ("Dettaglio", "scheda paese a 4 pannelli"),
             ("Motivo dell'interazione", "lettori diversi cercano paesi diversi: 172 schede non stanno in slide")]:
    R.voce(k, v)
R.salva(OUT, "report_atto4_dashboard.txt")
