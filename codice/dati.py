"""Parte 1 - Dati (slide 7-11).

Struttura del dataset, problemi trovati e decisioni prese, analisi esplorativa.
Descrive i dati con tutte le 1.190 votazioni; non fa test.
Output: output/atto2_dati/ (report, figure)
"""
import numpy as np
import pandas as pd
import comune as c

OUT = c.cartella_atto("atto2_dati")
plt = c.stile()
df, P, W, meta = c.dati()
R = c.Report("REPORT ATTO 2 - DATI: STRUTTURA, PULIZIA E ANALISI ESPLORATIVA")
anni = list(range(2014, 2026))

# ------------------------------------------------------------ 1. struttura
R.sezione("Struttura del dataset (slide 7-8)")
R.voce("File", c.file_dati().name)
R.voce("Dimensioni", f"{df.shape[0]:,} righe x {df.shape[1] - 1} colonne".replace(",", "."))
R.voce("Unità di osservazione", "voto di uno Stato in una votazione")
R.voce("Votazioni (rcid)", meta.shape[0])
R.voce("Stati", W.shape[1])
righe_per_voto = df.groupby("rcid").size()
R.voce("Righe per votazione", f"min {righe_per_voto.min()} / max {righe_per_voto.max()} (atteso 193)")
pres = W.notna().mean()
R.voce("Stati presenti in >= 75% votazioni", f"{(pres >= .75).sum()} (usati per PCA, UMAP, cluster)")
R.voce("Codici di risoluzione duplicati", int(meta.unres.duplicated().sum()))
R.lettura("Una votazione = una risoluzione votata per intero: nessuna votazione separata "
          "su paragrafi o emendamenti.")
R.attenzione("Solo votazioni REGISTRATE: le risoluzioni adottate per consenso non ci sono. "
             "Il dataset descrive i casi contesi, non tutta l'attività dell'Assemblea.")

# ------------------------------------------------------------- 2. pulizia
R.sezione("Problemi trovati e decisioni (slide 9)")
dist = df.vote.map({1: "Sì", 2: "Astenuto", 3: "No", 8: "Assente"}).value_counts()
R.voce("Distribuzione dei voti", ", ".join(f"{k} {v:,} ({v / len(df):.1%})".replace(",", ".")
                                          for k, v in dist.items()))

R.testo("")
R.testo("[PROBLEMA 1] Assenze (vote = 8): " + f"{(df.vote == 8).sum():,} righe".replace(",", "."))
R.testo("   -> DECISIONE: dato mancante (NaN), non un voto. Un'assenza può dipendere da delegazioni")
R.testo("                 piccole o dalla sospensione per morosità (art. 19 della Carta).")

sol = meta.groupby("year").solo_usa.sum()
R.testo("")
R.testo(f"[PROBLEMA 2] 2025 anomalo: {int((meta.year == 2025).sum())} votazioni contro "
        f"{int(meta[meta.year < 2025].groupby('year').size().min())}-{int(meta[meta.year < 2025].groupby('year').size().max())} negli altri anni.")
R.testo(f"   -> CAUSA: {int(sol[2025])} votazioni 'USA contro tutti' (USA contrari con al massimo 2 altri")
R.testo("             contrari), su testi prima adottati per consenso. Negli anni precedenti "
        f"{int(sol[sol.index < 2025].min())}-{int(sol[sol.index < 2025].max())} l'anno.")
R.testo("   -> DECISIONE: nessuna votazione rimossa; flag 'solo_usa' per escluderle dove")
R.testo("                 distorcerebbero i confronti tra anni (dichiarato in ogni didascalia).")
R.dati["solo_usa_per_anno"] = {int(k): int(v) for k, v in sol.items()}

coincide = ((meta.quota_maj < .8) == (meta.importantvote == 1)).mean()
R.testo("")
R.testo("[PROBLEMA 3] 'importantvote' non è l'elenco del Dipartimento di Stato USA.")
R.testo(f"   -> VERIFICA: coincide con 'voto più frequente < 80%' nel {coincide:.1%} delle votazioni.")
R.testo("   -> DECISIONE: non usarlo per selezionare le votazioni dei cluster (toglierebbe proprio")
R.testo("                 quelle che separano i paesi); usato solo come descrizione.")

senza = meta.n_temi == 0
R.testo("")
R.testo(f"[PROBLEMA 4] Votazioni senza tema: {int(senza.sum())} su {meta.shape[0]} "
        f"(dal {int(senza[meta.year <= 2016].sum() / 3)} l'anno nel 2014-16 a {int(senza[meta.year == 2025].sum())} nel 2025).")
ucr = meta.session.astype(str).eq("11emsp")
R.testo(f"   -> NOTA: {int((senza & ucr).sum())} delle {int(ucr.sum())} votazioni sull'Ucraina sono senza tema.")
R.testo("   -> DECISIONE: analisi per tema sulle "
        f"{int((~senza).sum())} votazioni con almeno un tema; estensione dei temi rimandata.")
R.testo(f"   -> NOTA: {int((meta.n_temi > 1).sum())} votazioni hanno più di un tema (i temi si sovrappongono).")

R.testo("")
R.testo(f"[PROBLEMA 5] Pareggi nel calcolo della maggioranza: {int(meta.maj_pari.sum())} votazioni.")
R.testo("   -> DECISIONE: maggioranza = voto più frequente tra sì/astenuto/no, assenti esclusi.")

R.testo("")
R.testo("[PROBLEMA 6] Codici Stato di ripiego: solo Guinea Equatoriale (ccode 1067).")
R.testo("   -> DECISIONE: nessun effetto, tutte le analisi usano il codice ISO3.")

R.testo("")
R.testo("[PROBLEMA 7] Variabili esterne assenti nel file.")
R.testo("   -> DECISIONE: aggiunti Movimento Non Allineati (119 membri, elenco esplicito in comune.py),")
R.testo("                 continente (pacchetto country_converter), G20 (19 Stati, UE esclusa).")

# --------------------------------------------------------- 3. esplorativa
R.sezione("Analisi esplorativa (slide 10-11)")

# 3a votazioni per anno
cnt = meta.groupby("year").agg(tot=("solo_usa", "size"), sol=("solo_usa", "sum"))
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(cnt.index, cnt.tot - cnt.sol, color=c.GRIGIO, width=.7, label="Altre votazioni")
ax.bar(cnt.index, cnt.sol, bottom=cnt.tot - cnt.sol, color=c.ACCENTO, width=.7,
       edgecolor=c.SFONDO, linewidth=2, label="USA contro (quasi) tutti: ≤ 3 voti contrari")
ax.set_xticks(anni); ax.set_ylabel("Votazioni registrate"); ax.legend(frameon=False, loc="upper left")
ax.annotate(f"{int(cnt.sol[2025])} su {int(cnt.tot[2025])}", xy=(2025, cnt.tot[2025] - 40),
            xytext=(2022.3, cnt.tot[2025] - 25), fontsize=10, arrowprops=dict(arrowstyle="-", color=c.GRIGIO))
c.griglia(ax)
ax.set_title(f"Il 2025 non è un errore: {int(cnt.sol[2025])} votazioni chieste dagli USA su testi prima adottati per consenso", fontsize=12)
R.voce("Votazioni per anno", ", ".join(f"{a}: {int(n)}" for a, n in cnt.tot.items()))
R.figura(c.salva(fig, OUT, "votazioni_per_anno"), "appendice")

# 3b distribuzione dei voti per anno
lab = {1: "Sì", 2: "Astenuto", 3: "No", 8: "Assente"}
col = c.COL_VOTI
D = pd.crosstab(df.year, df.vote.map(lab), normalize="index")[list(col)] * 100
fig, ax = plt.subplots(figsize=(9, 4.6)); base = np.zeros(len(D))
for k in col:
    kw_ = dict(hatch="///", edgecolor=c.GRIGIO, linewidth=0) if k == "Assente" else dict(edgecolor=c.SFONDO, linewidth=1.5)
    ax.bar(D.index, D[k], bottom=base, color=col[k], width=.7, label=k, **kw_)
    base += D[k].values
tot_v = df.vote.map(lab).value_counts(normalize=True) * 100
for k, y_ in {"Sì": 40, "Astenuto": 79, "No": 89, "Assente": 98.5}.items():
    ax.text(2025.6, y_, f"{c.T(k, {'Sì': 'Yes', 'Astenuto': 'Abstain', 'No': 'No', 'Assente': 'Absent'}[k])} {tot_v[k]:.0f}%", va="center", fontsize=11, color=c.BLU_SI if k == "Sì" else c.INCHIOSTRO,
            fontweight="bold" if k == "Sì" else "normal")
ax.set_xticks(anni[::2]); ax.set_ylabel("% dei voti"); ax.set_ylim(0, 100); ax.set_xlim(anni[0] - .6, anni[-1] + 3.4)
ax.set_title("Tre voti su quattro sono sì: da qui nasce l'effetto maggioranza")
R.voce("Quota di sì, min-max per anno", f"{D['Sì'].min():.1f}% - {D['Sì'].max():.1f}%")
R.figura(c.salva(fig, OUT, "distribuzione_voti"), 10)

# 3c quanto sono contese
q = meta.quota_maj
fig, ax = plt.subplots(figsize=(9, 4))
h_, e_ = np.histogram(q * 100, bins=np.arange(40, 101, 2.5))
ax.bar(e_[:-1] + 1.25, h_, width=2.3, color=[c.CORALLO if x_ >= 90 else c.GRIGIO for x_ in e_[:-1]])
ax.text(42, max(h_) * .78, c.T(f"{(q >= .9).mean() * 100:.0f}% delle votazioni: almeno\n9 Stati su 10 votano allo stesso modo →", f"{(q >= .9).mean() * 100:.0f}% of votes: at least\n9 states in 10 vote the same way →"),
        fontsize=10.5, color=c.ACCENTO_TESTO, fontweight="bold", va="center")
ax.axvline(80, color=c.INCHIOSTRO, ls=":", lw=1)
ax.text(79, ax.get_ylim()[1] * .9, "soglia di 'importantvote'", ha="right", fontsize=9, color=c.TENUE)
ax.set_xlabel("% di Stati votanti che esprimono il voto più frequente"); ax.set_ylabel("Votazioni")
c.griglia(ax)
ax.set_title(f"Metà delle votazioni è quasi unanime (mediana {q.median() * 100:.0f}%)")
R.voce("Quota del voto più frequente: mediana", c.pct(q.median()))
R.voce("Votazioni con quota >= 90%", f"{(q >= .9).sum()} ({c.pct((q >= .9).mean(), 0)})")
R.voce("Votazioni con quota < 80% (importantvote)", f"{(q < .8).sum()} ({c.pct((q < .8).mean(), 0)})")
R.figura(c.salva(fig, OUT, "quanto_sono_contese"), 10)

# 3d assenze per Stato
ass = W.isna().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4.2))
top = ass.head(12)[::-1]
ax.barh([c.nome(i) if i in c.NOMI else P.loc[i, "Countryname"].title() for i in top.index],
        top * 100, color=c.GRIGIO, height=.65)
ax.axvline(ass.median() * 100, color=c.EVIDENZA, lw=1.5)
ax.text(ass.median() * 100 + .8, -.6, f"mediana {ass.median() * 100:.1f}%", color=c.INCHOSTRO if False else c.INCHIOSTRO, fontsize=9)
ax.set_xlabel("% di votazioni in cui lo Stato è assente"); c.griglia(ax, "x")
ax.set_title("Le assenze sono concentrate in pochi Stati")
R.voce("Assenza: mediana per Stato", c.pct(ass.median()))
R.voce("Stati assenti in > 25% delle votazioni", f"{(ass > .25).sum()}: " +
       ", ".join(P.loc[i, "Countryname"].title() for i in ass[ass > .25].index))
R.figura(c.salva(fig, OUT, "appendice_assenze_per_stato"), "appendice")

# 3e temi nel tempo
T = meta.groupby("year")[list(c.TEMI)].sum()
fig, axs = plt.subplots(2, 4, figsize=(13, 5), sharey=True, sharex=True)
for ax, (k, n) in zip(axs.flat, list(c.TEMI.items()) + [("nessuno", "Nessun tema")]):
    s = T[k] if k != "nessuno" else meta[meta.n_temi == 0].groupby("year").size().reindex(anni, fill_value=0)
    ax.bar(s.index, s.values, color=c.CONFRONTO if k == "nessuno" else c.EVIDENZA, width=.75)
    ax.set_title(f"{n} ({int(s.sum())})", fontsize=10, fontweight="normal"); ax.set_xticks([2014, 2019, 2024])
axs.flat[-1].axis("off")
axs.flat[-1].text(0, .5, "I temi si sovrappongono:\n"
                  f"{int((meta.n_temi > 1).sum())} votazioni ne hanno più di uno.\n"
                  "Barre = votazioni per anno.", fontsize=9.5, color=c.TENUE, va="center",
                  transform=axs.flat[-1].transAxes)
fig.suptitle("Temi delle votazioni nel tempo (classificazione di Voeten)", x=.01, ha="left", fontweight="bold", fontsize=13)
for k, n in c.TEMI.items():
    R.voce(f"Votazioni con tema {n}", int(meta[k].sum()))
R.voce("Votazioni senza tema", int((meta.n_temi == 0).sum()))
R.figura(c.salva(fig, OUT, "appendice_temi_nel_tempo"), "appendice")

R.dati.update(votazioni=int(meta.shape[0]), stati_75=int((pres >= .75).sum()),
              quota_si=float((df.vote == 1).mean()), assenze=int((df.vote == 8).sum()),
              senza_tema=int(senza.sum()), mediana_quota_maj=float(q.median()))
R.salva(OUT, "report_atto2_dati.txt")
