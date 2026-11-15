"""Parte 4 - Casi (slide 28-30). Q6: chi fa eccezione, e perché?

Scheda paese (la stessa della dashboard) per India, Sudafrica e Mali: gruppo, accordo
con le potenze per anno, quanto segue ogni potenza in minoranza, temi, paesi più vicini.
Sulle slide: Pakistan e Sudafrica contro il proprio caso (figure_slide/numeri_slide.py)
e il Mali anno per anno (figure_slide/slide30_mali.py).
Richiede gli output della parte 2 (clusters.csv, pcs.csv, kappa.csv).
Output: output/atto3_livello3_casi/ (report, figure)
"""
import numpy as np
import pandas as pd
import comune as c

OUT = c.cartella_atto("atto3_livello3_casi")
L1 = c.OUTPUT / "atto3_livello1_mondo"
plt = c.stile()
df, P, W, meta = c.dati()

# ogni test e ogni confronto è fatto senza le votazioni "USA contro tutti"
W_TUTTE, META_TUTTE = W, meta
W, meta = W[meta.solo_usa == 0], meta[meta.solo_usa == 0]
CL = pd.read_csv(L1 / "clusters.csv", index_col=0)
PC = pd.read_csv(L1 / "pcs.csv", index_col=0)
K = pd.read_csv(L1 / "kappa.csv", index_col=0)
R = c.Report("REPORT ATTO 3 - LIVELLO 3 (CASI DI STUDIO): CHI FA ECCEZIONE?")
NAM = [i for i in W.columns if i in c.NAM]
anni = list(range(2014, 2026))
Wn, mn = W[meta.solo_usa == 0], meta[meta.solo_usa == 0]          # confronti tra anni senza "USA contro tutti"
PERIODI = {"2014-21": range(2014, 2022), "2022-25": range(2022, 2026)}


def nome_esteso(i):
    return c.nome(i) if i in c.NOMI else P.loc[i, "Countryname"].title()


def scheda(paese):
    """Calcola tutti gli indicatori della scheda paese."""
    s = {}
    s["accordo_anno"] = pd.DataFrame({
        **{c.nome(p): [c.accordo(Wn[mn.year == a], paese, p) for a in anni] for p in c.POTENZE},
        "Maggioranza": [float(Wn.loc[mn.year == a, paese].eq(mn.maj[mn.year == a]).where(Wn.loc[mn.year == a, paese].notna()).astype(float).mean()) for a in anni],
    }, index=anni)
    seg = {}
    for per, yy in PERIODI.items():
        for p in c.POTENZE:
            sp, n = c.segue_in_minoranza(W[meta.year.isin(yy)], meta[meta.year.isin(yy)], p)
            seg[(per, p)] = (sp[paese], sp[NAM].median(), n)
    s["segue"] = seg
    prof = []
    for k, nm in c.TEMI.items():
        sub = Wn[mn[k] == 1]
        si = (sub == 1).where(sub.notna()).astype(float).mean(); ab = (sub == 0).where(sub.notna()).astype(float).mean()
        prof.append(dict(tema=nm, si=si[paese], si_nam=si[NAM].median(), ast=ab[paese], ast_nam=ab[NAM].median()))
    s["profilo"] = pd.DataFrame(prof).set_index("tema")
    s["vicini"] = K[paese].drop(paese).sort_values(ascending=False).head(5)
    s["cluster"] = CL.loc[paese, "gruppo"]; s["comunita"] = CL.loc[paese, "comunita"]
    s["pc1"] = PC.loc[paese, "PC1"]; s["pc1_rank"] = int((PC.PC1 > PC.loc[paese, "PC1"]).sum()) + 1
    s["assenza"] = float(W[paese].isna().mean())
    return s


def scrivi(paese, s, ruolo):
    R.voce("Ruolo nel racconto", ruolo)
    R.voce("Cluster (livello 1) / comunità di rete", f"{s['cluster']} / {s['comunita']}")
    R.voce("Posizione su PC1", f"{s['pc1']:.1f} ({s['pc1_rank']}° su {len(PC)} in ordine decrescente; positivo = lato 'resto del mondo')")
    R.voce("Assenze", c.pct(s["assenza"]))
    a = s["accordo_anno"]
    for col in a.columns:
        R.voce(f"Accordo con {col}: 2014-21 / 2022-25", f"{c.pct(a.loc[:2021, col].mean(), 0)} / {c.pct(a.loc[2022:, col].mean(), 0)}")
    for (per, p), (v, med, n) in s["segue"].items():
        R.voce(f"Segue {c.nome(p)} in minoranza, {per}", f"{c.pct(v, 0)} (membro mediano {c.pct(med, 0)}, n = {n})")
    pr = s["profilo"]
    diff = (pr.si - pr.si_nam).abs().sort_values(ascending=False)
    R.voce("Temi più lontani dal membro mediano", ", ".join(f"{t} (sì {pr.loc[t, 'si']:.0%} vs {pr.loc[t, 'si_nam']:.0%})" for t in diff.index[:3]))
    R.voce("Paesi più vicini (kappa)", ", ".join(f"{nome_esteso(i)} {v:.2f}" for i, v in s["vicini"].items()))


def figura(paese, s, titolo, nome_file, slide):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.2))
    ax = axs[0, 0]; a = s["accordo_anno"] * 100
    colori = {"USA": c.COL_POTENZE["USA"], "Cina": c.COL_POTENZE["CHN"], "Russia": c.COL_POTENZE["RUS"], "Maggioranza": c.GRIGIO}
    for col in a.columns:
        ax.plot(a.index, a[col], color=colori[col], lw=2.3 if col != "Maggioranza" else 1.8,
                ls="-" if col != "Maggioranza" else "--")
        ax.text(2025.3, a[col].iloc[-1], col, fontsize=10, va="center")
    ax.axvline(2022, color=c.GRIGIO, ls=":", lw=1)
    ax.set_ylim(0, 102); ax.set_xlim(2014, 2027.2); ax.set_xticks(range(2014, 2026, 2))
    ax.set_ylabel("% voti uguali"); ax.set_title("Accordo per anno", fontsize=12)
    ax = axs[0, 1]
    pots = list(c.POTENZE); x = np.arange(len(pots)); w = .36
    for j, (per, col) in enumerate([("2014-21", c.GRIGIO), ("2022-25", c.INCHIOSTRO)]):
        v = [s["segue"][(per, p)][0] * 100 for p in pots]
        ax.bar(x + (j - .5) * w, v, width=w, color=col, label=per, edgecolor=c.SFONDO, linewidth=1.5)
        for xi, vi in zip(x + (j - .5) * w, v):
            ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=9)
    for xi, p in zip(x, pots):
        med = s["segue"][("2014-21", p)][1] * 100
        ax.plot([xi - .4, xi + .4], [med, med], color=c.INCHIOSTRO, lw=1.2, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([c.nome(p) for p in pots]); ax.set_ylim(0, 80)
    ax.set_ylabel("% votazioni"); ax.legend(frameon=False, fontsize=9, title="periodo", title_fontsize=9)
    ax.set_title("Segue la potenza in minoranza\n(punteggiato: membro mediano 2014-21)", fontsize=12)
    ax = axs[1, 0]; pr = s["profilo"]; y = np.arange(len(pr))[::-1]
    for yi, t in zip(y, pr.index):
        ax.plot([pr.loc[t, "si_nam"] * 100, pr.loc[t, "si"] * 100], [yi, yi], color=c.GRIGIO_CHIARO, lw=2)
    ax.scatter(pr.si_nam * 100, y, s=55, color=c.GRIGIO, label="membro mediano", zorder=3)
    ax.scatter(pr.si * 100, y, s=70, color=c.EVIDENZA, label=c.nome(paese), zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(pr.index); ax.set_xlim(0, 105); ax.set_xlabel("% di sì")
    ax.legend(frameon=False, fontsize=9, loc="lower left"); c.griglia(ax, "x")
    ax.set_title("Profilo per tema", fontsize=12)
    ax = axs[1, 1]; v = s["vicini"][::-1]
    ax.barh([nome_esteso(i) for i in v.index], v.values, color=c.GRIGIO, height=.6)
    for i_, vv in enumerate(v.values):
        ax.text(vv + .01, i_, f"{vv:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, max(.8, v.max() + .1)); ax.set_xlabel("κ di Cohen")
    ax.set_title(f"I 5 paesi che votano in modo più simile · cluster: {s['cluster']}", fontsize=12)
    fig.suptitle(titolo, x=.01, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    R.figura(c.salva(fig, OUT, nome_file, rect_layout=(0, 0, 1, .96)), slide)


# ------------------------------------------------------- 1. criterio
R.sezione("Criterio di scelta dei casi")
seg_cina, _ = c.segue_in_minoranza(W, meta, "CHN")
seg_usa, _ = c.segue_in_minoranza(W, meta, "USA")
R.voce("Universo", "membri del Movimento presenti in >= 75% delle votazioni")
R.voce("Chi segue", f"India: segue la Cina in minoranza nel {c.pct(seg_cina['IND'], 0)} dei casi "
       f"(membro mediano {c.pct(seg_cina[NAM].median(), 0)}). Più alti solo Corea del Nord, Siria, Iran, Pakistan, Zimbabwe: casi attesi.")
R.voce("Chi è autonomo", f"Sudafrica: segue la Cina nel {c.pct(seg_cina['ZAF'], 0)} dei casi e gli USA nel {c.pct(seg_usa['ZAF'], 0)}; "
       "come l'India è nei BRICS.")
R.voce("Chi cambia", "Mali: maggiore avvicinamento alla Russia tra i membri dal 2021 al 2023 (vedi livello 2).")

casi = {"IND": ("chi segue", "India: il membro che segue di più la Cina, senza seguire gli USA", "scheda_india", "appendice"),
        "ZAF": ("chi è autonomo", "Sudafrica: stessi BRICS dell'India, ma non segue nessuna potenza", "scheda_sudafrica", 29),
        "MLI": ("chi cambia", "Mali: dopo i colpi di stato si allontana dalla maggioranza, verso Mosca e Pechino", "scheda_mali", 30)}
schede = {}
for paese, (ruolo, titolo, nf, sl) in casi.items():
    R.sezione(f"Scheda {c.nome(paese)} (slide {sl})")
    s = scheda(paese); schede[paese] = s
    scrivi(paese, s, ruolo)
    figura(paese, s, titolo, nf, sl)
    R.dati[paese] = dict(cluster=s["cluster"], comunita=s["comunita"],
                         accordo_2014_21={k: float(v) for k, v in s["accordo_anno"].loc[:2021].mean().items()},
                         accordo_2022_25={k: float(v) for k, v in s["accordo_anno"].loc[2022:].mean().items()},
                         segue={f"{p}_{per}": float(v[0]) for (per, p), v in s["segue"].items()},
                         vicini=s["vicini"].round(3).to_dict())

# -------------------------------------------------- confronti
R.sezione("India contro Sudafrica")
for p in c.POTENZE:
    R.voce(f"Segue {c.nome(p)} in minoranza (2014-25): India / Sudafrica", f"{c.pct(c.segue_in_minoranza(W, meta, p)[0]['IND'], 0)} / {c.pct(c.segue_in_minoranza(W, meta, p)[0]['ZAF'], 0)}")
R.voce("Kappa India-Sudafrica", f"{K.loc['IND', 'ZAF']:.2f}")
R.voce("Cluster India / Sudafrica", f"{CL.loc['IND', 'gruppo']} / {CL.loc['ZAF', 'gruppo']}")
R.lettura("Due membri dei BRICS stanno in cluster diversi: l'India nel blocco Cina–Russia, il "
          "Sudafrica nella maggioranza del Sud globale. I BRICS non sono un blocco di voto.")

R.sezione("Mali prima e dopo (slide 30)")
a = schede["MLI"]["accordo_anno"]
R.voce("Accordo con la Russia per anno", ", ".join(f"{y}: {v:.0%}" for y, v in a["Russia"].items()))
R.voce("Accordo con gli USA per anno", ", ".join(f"{y}: {v:.0%}" for y, v in a["USA"].items()))
sg = schede["MLI"]["segue"]
R.voce("Segue la Russia in minoranza: 2014-21 -> 2022-25", f"{c.pct(sg[('2014-21', 'RUS')][0], 0)} -> {c.pct(sg[('2022-25', 'RUS')][0], 0)} "
       f"(membro mediano {c.pct(sg[('2022-25', 'RUS')][1], 0)})")
R.voce("Segue la Cina in minoranza: 2014-21 -> 2022-25", f"{c.pct(sg[('2014-21', 'CHN')][0], 0)} -> {c.pct(sg[('2022-25', 'CHN')][0], 0)}")
R.lettura("L'accordo medio con la Russia oscilla (picco 75% nel 2023); il segnale netto è un altro: "
          f"quando la Russia è isolata, il Mali la segue nel {sg[('2022-25', 'RUS')][0]:.0%} dei casi dopo il 2022 "
          f"contro il {sg[('2014-21', 'RUS')][0]:.0%} di prima (anno per anno: figure_slide/slide30_mali.py). "
          "Coincide con i colpi di stato del 2020-21 e con l'arrivo della presenza militare russa. "
          "Associazione nel tempo, non prova di causa.")
R.salva(OUT, "report_atto3_livello3_casi.txt")
