"""
Figure di supporto al racconto.
  1. Mappa: quanto ogni paese vota come la Cina quando la Cina è in minoranza (slide 23).
  2. Chord diagram: legami forti (kappa >= 0,4) tra i quattro gruppi (appendice).
  3. Le votazioni "USA contro tutti" del 2025 (appendice).
  4. Una votazione tipica, con la Cina in mezzo alla folla (slide 33).
  5. Controllo: chi segue gli USA quando sono in minoranza (slide 24, la slide usa figure_slide/slide24_us.py).
Richiede gli output delle parti 2 e 3 (clusters.csv, kappa.csv, segue_in_minoranza.csv).
Output: output/extra/ (report, figure)
"""
import numpy as np
import pandas as pd
try:
    import geopandas as gpd                              # serve solo alla mappa
except ImportError:
    gpd = None
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Wedge, Patch
import comune as c

OUT = c.cartella_atto("extra")
plt = c.stile()
df, P, W, meta = c.dati()
L1, L2 = c.OUTPUT / "atto3_livello1_mondo", c.OUTPUT / "atto3_livello2_non_allineati"
CL = pd.read_csv(L1 / "clusters.csv", index_col=0)
K = pd.read_csv(L1 / "kappa.csv", index_col=0)
F = pd.read_csv(L2 / "segue_in_minoranza.csv", index_col=0)
R = c.Report("REPORT VISUALIZZAZIONI EXTRA: MAPPA, CHORD DIAGRAM, USA CONTRO TUTTI")

# ------------------------------------------------------------ 1. mappa
if gpd is not None:
    R.sezione("Mappa coropletica: chi segue la Cina quando è in minoranza")
    mondo = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    mondo = mondo[mondo.continent != "Antarctica"].copy()
    mondo.loc[mondo.name == "Kosovo", "iso_a3"] = "XKX"
    mondo["valore"] = mondo.iso_a3.map(F["CHN"])
    mondo = mondo.to_crs("ESRI:54030")                       # proiezione di Robinson
    classi = [0, .05, .10, .20, .40, 1.01]
    etichette = ["meno del 5%", "5–10%", "10–20%", "20–40%", "oltre il 40%"]
    scala = c.SCALA_CINA                                                    # dal grigio all'ocra (colore della Cina): più la segue, più è scuro
    mondo["classe"] = pd.cut(mondo.valore, classi, right=False, labels=False)
    senza = mondo.valore.isna()
    R.voce("Valore", "quota di votazioni in cui il paese vota come la Cina, quando la Cina vota diversamente dalla maggioranza (250 votazioni)")
    R.voce("Paesi sulla mappa con dato / senza", f"{(~senza).sum()} / {senza.sum()} (piccoli Stati non disegnati a questa scala: {len(F) - (~senza).sum()})")
    for k, e in enumerate(etichette):
        nomi = mondo[mondo.classe == k].name.tolist()
        R.voce(f"Classe {e}", f"{len(nomi)} paesi" + (f": {', '.join(nomi)}" if len(nomi) <= 10 else ""))
    R.lettura("Un mondo quasi tutto chiaro con poche macchie scure (i 7 paesi sopra il 40%). Da usare per incuriosire: l'area dei paesi grandi pesa "
              "più del loro numero, e i piccoli Stati non si vedono.")
    fig, ax = plt.subplots(figsize=(13, 6.6))
    mondo[senza].plot(ax=ax, color=c.LINEA, edgecolor=c.SFONDO, linewidth=.4)
    for k, col in enumerate(scala):
        mondo[mondo.classe == k].plot(ax=ax, color=col, edgecolor=c.SFONDO, linewidth=.4)
    for iso, dx, dy in [("PRK", 2.6e6, 0.2e6), ("IRN", -0.6e6, -2.6e6), ("SYR", -3.4e6, 0.4e6), ("PAK", -0.2e6, -4.2e6),
                        ("IND", 3.2e6, -2.2e6), ("RUS", 0, 1.3e6), ("ZWE", 1.8e6, -1.2e6)]:
        r_ = mondo[mondo.iso_a3 == iso]
        if r_.empty:
            continue
        p = r_.geometry.iloc[0].representative_point()
        ax.annotate(f"{c.nome(iso) if iso in c.NOMI else r_.name.iloc[0]} {F.loc[iso, 'CHN']:.0%}", (p.x, p.y),
                    xytext=(p.x + dx, p.y + dy), fontsize=9.5, color=c.INCHIOSTRO, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=.2", fc=c.SCHEDA, ec="none", alpha=.9),
                    arrowprops=dict(arrowstyle="-", color=c.TENUE, lw=.6))
    p = mondo[mondo.iso_a3 == "CHN"].geometry.iloc[0].representative_point()
    ax.annotate("Cina", (p.x, p.y), fontsize=10, color=c.INCHIOSTRO, ha="center", fontweight="bold")
    mondo[mondo.iso_a3 == "CHN"].plot(ax=ax, facecolor="none", edgecolor=c.TESTO_OCRA, linewidth=1.6, hatch="////")
    ax.set_axis_off()
    ax.legend([Patch(color=col) for col in scala] + [Patch(color=c.LINEA)], etichette + ["nessun dato"],
              frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(.5, -.06), fontsize=9.5,
              title="Quando la Cina vota diversamente dalla maggioranza, il paese vota come lei:", title_fontsize=10)
    ax.set_title("Quando la Cina resta sola, la segue quasi nessuno")
    R.figura(c.salva(fig, OUT, "mappa_segue_cina"), 23)


# ------------------------------------------------------------ 2. chord
R.sezione("Chord diagram sui quattro gruppi")
g = CL.gruppo
gruppi = list(c.GCOL)
S = list(K.index); a = K.values
M = pd.DataFrame(0, index=gruppi, columns=gruppi)
for i in range(len(S)):
    for j in range(i + 1, len(S)):
        if a[i, j] >= .4:
            gi, gj = g[S[i]], g[S[j]]
            M.loc[gi, gj] += 1
            if gi != gj:
                M.loc[gj, gi] += 1
corti = {"Maggioranza del Sud globale": "Sud globale", "Europa e alleati": "Europa e alleati",
         "Blocco Cina–Russia": "Cina–Russia", "USA, Israele e alleati stretti": "USA–Israele"}
R.voce("Archi esterni", "lunghezza proporzionale al numero di Stati del gruppo")
R.voce("Nastri", "legami forti (κ di Cohen ≥ 0,4) tra paesi di gruppi DIVERSI")
for x in gruppi:
    R.voce(f"{x} (n = {(g == x).sum()})", f"legami interni {M.loc[x, x]} | " +
           ", ".join(f"con {corti[y]} {M.loc[x, y]}" for y in gruppi if y != x))
R.lettura("I legami forti stanno quasi tutti dentro i gruppi. Tra gruppi diversi ci sono solo due ponti: "
          "Sud globale–blocco Cina/Russia (59) ed Europa–USA/Israele (66). Tra Sud globale ed Europa: 1.")
n = np.array([(g == x).sum() for x in gruppi], float)
gap = np.deg2rad(6); tot = 2 * np.pi - gap * len(gruppi)
span = n / n.sum() * tot
inizio = np.concatenate([[0], np.cumsum(span + gap)[:-1]]) + np.pi / 2
fig, ax = plt.subplots(figsize=(9, 9)); ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
r0, r1 = 1.0, 1.08
for k, x in enumerate(gruppi):
    th1, th2 = np.rad2deg(inizio[k]), np.rad2deg(inizio[k] + span[k])
    ax.add_patch(Wedge((0, 0), r1, th1, th2, width=r1 - r0, color=c.GCOL[x]))
    mid = inizio[k] + span[k] / 2
    ax.text(1.2 * np.cos(mid), 1.2 * np.sin(mid), f"{corti[x]}\n{int(n[k])} Stati", ha="center", va="center",
            fontsize=12, color=c.INCHIOSTRO, fontweight="bold")
# nastri tra gruppi diversi: larghezza proporzionale ai legami, centrati nel proprio arco
tra = [(i, j, M.iloc[i, j]) for i in range(4) for j in range(i + 1, 4) if M.iloc[i, j] > 0]
scala_rad = .5 * min(span[i] / max(1, sum(M.iloc[i, j] for j in range(4) if j != i)) for i in range(4))
cursore = inizio + span / 2 - np.array([sum(M.iloc[i, j] for j in range(4) if j != i) for i in range(4)]) * scala_rad / 2
def pt(t, r=r0):
    return (r * np.cos(t), r * np.sin(t))
for i, j, v in sorted(tra, key=lambda x: -x[2]):
    w = max(v * scala_rad, np.deg2rad(1.2))
    a1, a2 = cursore[i], cursore[i] + w; cursore[i] += w
    b1, b2 = cursore[j], cursore[j] + w; cursore[j] += w
    archi = lambda t1, t2: [pt(t) for t in np.linspace(t1, t2, 20)]
    vert = archi(a1, a2) + [(0, 0), pt(b1)] + archi(b1, b2)[1:] + [(0, 0), pt(a1)]
    codici = ([MPath.MOVETO] + [MPath.LINETO] * 19 + [MPath.CURVE3, MPath.CURVE3]
              + [MPath.LINETO] * 19 + [MPath.CURVE3, MPath.CURVE3])
    col = c.GCOL[gruppi[i]] if v > 5 else c.GRIGIO
    ax.add_patch(PathPatch(MPath(vert, codici), facecolor=col, alpha=.55, edgecolor="none"))
    m = (a1 + a2) / 2; mb = (b1 + b2) / 2
    xm, ym = (np.cos(m) + np.cos(mb)) * .32, (np.sin(m) + np.sin(mb)) * .32
    ax.text(xm, ym, f"{int(v)}", ha="center", va="center", fontsize=12, color=c.INCHIOSTRO,
            bbox=dict(boxstyle="round,pad=.25", fc=c.SCHEDA, ec="none"))
ax.text(0, -1.42, "Nastri: coppie di paesi di gruppi diversi con accordo forte (κ di Cohen ≥ 0,4).\n"
        "Legami interni ai gruppi (non disegnati): " + ", ".join(f"{corti[x]} {M.loc[x, x]:,}".replace(",", ".") for x in gruppi) + ".",
        ha="center", va="top", fontsize=9.5, color=c.TENUE)
ax.set_title("Tra i blocchi ci sono solo due ponti", loc="center")
R.figura(c.salva(fig, OUT, "chord_gruppi"), "appendice")

# ---------------------------------------------------- 3. USA contro tutti
R.sezione("USA contro tutti nel 2025")
idx = meta.index[(meta.year == 2025) & (meta.solo_usa == 1)]
sub = W.loc[idx]
C = pd.DataFrame({"Sì": (sub == 1).sum(axis=1), "Astenuto": (sub == 0).sum(axis=1),
                  "Assente": sub.isna().sum(axis=1), "No": (sub == -1).sum(axis=1)}).sort_values(["Sì", "Astenuto"], ascending=False)
compagni = (sub == -1).drop(columns="USA").sum().sort_values(ascending=False)
R.voce("Votazioni", f"{len(idx)} nel 2025 (su 192): USA contrari con al massimo altri 2 contrari")
R.voce("Sì per votazione: mediana (min-max)", f"{C['Sì'].median():.0f} ({C['Sì'].min()}-{C['Sì'].max()})")
R.voce("No per votazione", ", ".join(f"{k}: {v} votazioni" for k, v in C["No"].value_counts().sort_index().items()))
R.voce("Chi vota no con gli USA", ", ".join(f"{P.loc[i, 'Countryname'].title()} {v}" for i, v in compagni[compagni > 0].items()))
R.voce("Temi", ", ".join(f"{c.TEMI[k]} {int(meta.loc[idx, k].sum())}" for k in c.TEMI) + f", senza tema {int((meta.loc[idx, 'n_temi'] == 0).sum())}")
R.attenzione("Il dataset non dice chi ha chiesto il voto registrato: che siano stati gli USA è solo una deduzione.")
# voto tipico: una votazione reale vicina ai valori mediani delle 114
ESEMPIO = 11234
v = W.loc[ESEMPIO]
R.voce("Votazione d'esempio (pannello sinistro)", f"{meta.loc[ESEMPIO, 'date'].date()}, {meta.loc[ESEMPIO, 'short'].split(' : ')[0]}: "
       f"sì {(v == 1).sum()}, astenuti {(v == 0).sum()}, no {(v == -1).sum()} ({', '.join(c.nome(i) for i in v.index[v == -1])}), assenti {v.isna().sum()}")
R.lettura("A sinistra un'aula intera di sì contro due no; a destra quante volte succede: 5-30 l'anno fino al 2024, 114 nel 2025.")
sol = meta.groupby("year").solo_usa.sum()
fig, ax = plt.subplots(figsize=(10, 6.2)); ax.set_aspect("equal"); ax.axis("off")
ordine = ([("Sì", i) for i in v.index[v == 1]] + [("Astenuto", i) for i in v.index[v == 0]]
          + [("Assente", i) for i in v.index[v.isna()]] + [("No", i) for i in v.index[v == -1]])
colonne = 20
col_punto = c.COL_VOTI
for k, (tipo, iso) in enumerate(ordine):
    x, y = k % colonne, -(k // colonne)
    ax.add_patch(plt.Circle((x, y), .38, color=col_punto[tipo], ec=c.NERO if tipo == "No" else "none", lw=1.5))
righe_tot = (len(ordine) - 1) // colonne
nno = 0
for k, (tipo, iso) in enumerate(ordine):
    if tipo == "No":
        x, y = k % colonne, -(k // colonne)
        dx = -1.6 if nno == 0 else 1.6; nno += 1
        ax.annotate(c.nome(iso), (x, y), xytext=(x + dx, y - 1.7), ha="center", fontsize=12, fontweight="bold",
                    color=c.PRUGNA_NO, arrowprops=dict(arrowstyle="-", color=c.PRUGNA_NO, lw=.8))
ax.set_xlim(-.8, colonne - .2); ax.set_ylim(-righe_tot - 2.6, 2.6)
ax.text(-.5, 1.5, f"{(v == 1).sum()} sì", fontsize=22, color=c.BLU_SI, fontweight="bold", va="center")
ax.text(5.2, 1.5, f"{v.isna().sum()} assenti", fontsize=14, color=c.TENUE, va="center")
ax.text(colonne - .5, 1.5, f"{(v == -1).sum()} no", fontsize=22, color=c.PRUGNA_NO, fontweight="bold", va="center", ha="right")
ax.text(-.5, -righe_tot - 2.4, "Un punto per Stato · Convenzione sui diritti delle persone con disabilità, 17 dicembre 2025",
        fontsize=10, color=c.TENUE)
ax.set_title("Nel 2025, 176 Stati contro 2: gli Stati Uniti restano soli", loc="left")
R.figura(c.salva(fig, OUT, "usa_una_votazione"), "appendice")

# una votazione tipica, non "USA contro tutti". Criterio: Cina con la maggioranza,
# maggioranza = sì, USA contrari, anni 2022-2024 (dopo l'invasione, prima dell'anomalo 2025), quota del voto più frequente
# più vicina alla mediana di tutte le votazioni (84,4%).
cand_ = meta[(meta.year >= 2022) & (meta.year <= 2024) & (meta.solo_usa == 0) & (meta.maj == 1) & (W["CHN"] == meta.maj) & (W["USA"] == -1)]
ESEMPIO_RIB = (cand_.quota_maj - meta.quota_maj.median()).abs().idxmin()
v = W.loc[ESEMPIO_RIB]
R.sezione("Votazione tipica (slide 33)")
R.voce("Criterio", "Cina e maggioranza votano sì, USA no, esclusa 'USA contro tutti' e il 2025; quota più vicina alla mediana")
R.voce("Mediana della quota del voto più frequente", c.pct(meta.quota_maj.median()))
R.voce("Votazione scelta", f"rcid {ESEMPIO_RIB}, {meta.loc[ESEMPIO_RIB, 'date'].date()}, {meta.loc[ESEMPIO_RIB, 'short'].split(' : ')[0]}")
R.voce("Quota del voto più frequente", c.pct(meta.loc[ESEMPIO_RIB, "quota_maj"]))
R.voce("Voti", f"sì {(v == 1).sum()}, astenuti {(v == 0).sum()}, no {(v == -1).sum()} "
       f"({', '.join(c.nome(i) if i in c.NOMI else P.loc[i, 'Countryname'].title() for i in v.index[v == -1])}), assenti {v.isna().sum()}")
n_si, n_ast, n_no, n_ass = int((v == 1).sum()), int((v == 0).sum()), int((v == -1).sum()), int(v.isna().sum())
fig, ax = plt.subplots(figsize=(10, 6.6)); ax.set_aspect("equal"); ax.axis("off")
si_ = [i for i in v.index[v == 1] if i != "CHN"]; pos_cina = 4 * colonne
si_.insert(pos_cina, "CHN")
no_ = ["USA"] + [i for i in v.index[v == -1] if i != "USA"]
ordine_c = ([("Sì", i) for i in si_] + [("Astenuto", i) for i in v.index[v == 0]]
            + [("Assente", i) for i in v.index[v.isna()]] + [("No", i) for i in no_])
righe_tot = (len(ordine_c) - 1) // colonne
for k, (tipo, iso) in enumerate(ordine_c):
    x, y = k % colonne, -(k // colonne)
    ax.add_patch(plt.Circle((x, y), .38, color=col_punto[tipo], ec="none"))
    if iso == "CHN":
        ax.add_patch(plt.Circle((x, y), .38, color=c.COL_CINA))
        ax.add_patch(plt.Circle((x, y), .52, fill=False, ec=c.COL_CINA, lw=3.5))
        ax.annotate(f"Cina:\nuno dei {n_si}", (x - .55, y), xytext=(x - 1.6, y), ha="right", va="center", fontsize=13, fontweight="bold",
                    color=c.TESTO_OCRA, arrowprops=dict(arrowstyle="-", color=c.COL_CINA, lw=1.6))
    if iso == "USA":
        ax.add_patch(plt.Circle((x, y), .52, fill=False, ec=c.NERO, lw=2.5))
        ax.annotate("USA", (x, y - .5), xytext=(x, y - 1.6), ha="center", va="top", fontsize=13, fontweight="bold",
                    color=c.NERO, arrowprops=dict(arrowstyle="-", color=c.NERO, lw=1))
ax.set_xlim(-5.2, colonne - .2); ax.set_ylim(-righe_tot - 3.2, 2.6)
ax.text(-.5, 1.5, f"{n_si} sì", fontsize=22, color=c.BLU_SI, fontweight="bold", va="center")
ax.text(7.5, 1.5, f"{n_ast} astenuti · {n_ass} assenti", fontsize=14, color=c.TENUE, va="center")
ax.text(colonne - .5, 1.5, f"{n_no} no", fontsize=22, color=c.PRUGNA_NO, fontweight="bold", va="center", ha="right")
ax.text(-.5, -righe_tot - 2.9, f"Un punto per Stato · Insediamenti israeliani nei Territori palestinesi occupati, "
        f"{meta.loc[ESEMPIO_RIB, 'date'].day} dicembre {meta.loc[ESEMPIO_RIB, 'year']}\n"
        f"Votazione tipica: il voto più frequente raccoglie il {meta.loc[ESEMPIO_RIB, 'quota_maj'] * 100:.0f}% dei votanti (mediana di tutte le votazioni: {meta.quota_maj.median() * 100:.0f}%)",
        fontsize=10, color=c.TENUE, va="top")
ax.set_title("La Cina non guida la folla: ne fa parte", loc="left")
R.figura(c.salva(fig, OUT, "votazione_tipica"), 33)
fig, ax2 = plt.subplots(figsize=(10, 5.6))
anni_ = sol.index.tolist()
ax2.bar(anni_, sol.values, color=[c.NERO if a == 2025 else c.GRIGIO for a in anni_], width=.7)
for a, n in sol.items():
    ax2.text(a, n + 2, str(int(n)), ha="center", fontsize=10 if a < 2025 else 16,
             color=c.NERO if a == 2025 else c.TENUE, fontweight="bold" if a == 2025 else "normal")
ax2.set_xticks(anni_); ax2.set_xticklabels([str(a)[2:] if a < 2025 else "2025" for a in anni_])
ax2.set_ylim(0, 130); c.griglia(ax2)
ax2.set_ylabel("votazioni in cui gli USA votano no\ncon al massimo altri 2 paesi")
ax2.set_title("Non è un caso isolato: nel 2025 succede 114 volte", loc="left")
R.figura(c.salva(fig, OUT, "usa_per_anno"), "appendice")



# ------------------------------------------------ 4. controllo positivo: chi segue gli USA in minoranza
R.sezione("Controllo positivo: quando gli USA restano in minoranza, chi li segue?")
NAMl = [i for i in F.index if i in c.NAM]; ALTRIl = [i for i in F.index if i not in c.NAM and i != "USA"]
top = F.USA.drop("USA").sort_values(ascending=False).head(8)
med_m, med_a = F.USA.reindex(NAMl).median(), F.USA.reindex(ALTRIl).median()
R.voce("Paesi che seguono di più gli USA in minoranza", ", ".join(f"{c.nome(i) if i in c.NOMI else P.loc[i, 'Countryname'].title()} {v:.0%}" for i, v in top.items()))
R.voce("Non membro mediano / membro mediano", f"{med_a:.0%} / {med_m:.0%}")
R.lettura("Il test riconosce un follower quando c'è: gli alleati degli USA li seguono anche da soli. I non allineati no.")
def nome_(i):
    n = c.nome(i) if i in c.NOMI else P.loc[i, "Countryname"].title()
    return {"Micronesia (Federated States Of)": "Micronesia"}.get(n, n)
# due righe sulla stessa scala: non membri (contesto, grigio) e membri (colore del Movimento);
# un solo accento per il valore estremo (Israele) e mediane come segni verticali
va_ = F.USA.reindex(ALTRIl).dropna() * 100; vm_ = F.USA.reindex(NAMl).dropna() * 100
# boxplot orizzontali con notch sulla stessa scala: il notch (mediana ± 1,57·IQR/√n, circa un IC al 95%)
# dice se la differenza tra le mediane è credibile. Grigio = altri paesi, corallo = Movimento.
Y_A, Y_M = 1.0, 0.0
fig, ax = plt.subplots(figsize=(11, 5.2))
def box(ax_, v, y0, fc, ec, w=.42, lw=1.3):
    ax_.boxplot(v.values, positions=[y0], vert=False, notch=True, widths=w, patch_artist=True, whis=1.5, showfliers=False,
                boxprops=dict(facecolor=fc, edgecolor=ec, lw=lw), medianprops=dict(color=c.SFONDO if fc == c.ACCENTO else "#22282A", lw=2.4),
                whiskerprops=dict(color=ec, lw=lw), capprops=dict(color=ec, lw=lw))
    q1, q3 = np.percentile(v, [25, 75]); fuori = v[(v > q3 + 1.5 * (q3 - q1)) | (v < q1 - 1.5 * (q3 - q1))].drop("ISR", errors="ignore")
    ax_.vlines(fuori, y0 - w * .22, y0 + w * .22, color=ec, lw=1.6, alpha=.8, zorder=3)
box(ax, va_, Y_A, c.GRIGIO_CHIARO, c.TENUE)
box(ax, vm_, Y_M, c.ACCENTO, c.ACCENTO_TESTO)
# Israele: l'unico accento
ax.vlines([va_["ISR"]], Y_A - .13, Y_A + .13, color=c.NERO, lw=4, zorder=5)
ax.text(va_["ISR"], Y_A + .2, f"Israele {va_['ISR']:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold", color=c.NERO)
ax.text(va_["CAN"], Y_A - .2, f"{nome_('CAN')} {va_['CAN']:.0f}%", ha="center", va="top", fontsize=11, color=c.TENUE)
ax.text(med_a * 100, Y_A + .27, f"mediana {med_a:.0%}", ha="center", va="bottom", fontsize=12, color="#22282A")
# focus + contesto: i membri sono tutti tra 0 e 10%, la scatola sulla scala comune è una striscia;
# una lente con la stessa scatola su 0-10% rende leggibili mediana e notch
ZX0, ZX1 = 0, 10
ins = ax.inset_axes([22, Y_M - .42, 56, .62], transform=ax.transData)
box(ins, vm_, 0, c.ACCENTO, c.ACCENTO_TESTO, w=.5, lw=1.4)
ins.set_xlim(ZX0 - .2, ZX1); ins.set_ylim(-.5, .62); ins.set_yticks([])
ins.set_xticks(range(0, 11, 2)); ins.set_xticklabels([f"{v}%" for v in range(0, 11, 2)], fontsize=9.5, color=c.TENUE)
ins.set_facecolor(c.SCHEDA); ins.grid(axis="x", color=c.LINEA, lw=.8); ins.grid(axis="y", visible=False); ins.set_axisbelow(True)
for sp in ins.spines.values():
    sp.set_color(c.GRIGIO); sp.set_linewidth(.8)
ins.text(vm_.median(), .33, f"mediana {med_m:.0%}", ha="center", va="bottom", fontsize=12, color=c.ACCENTO, fontweight="bold")
ax.text(22, Y_M + .24, c.T(f"Lente: i membri tra 0 e {ZX1}% · {int((vm_ < 2).sum())} su {len(vm_)} sotto il 2%, il più alto {nome_(vm_.idxmax())} {vm_.max():.0f}%",
                           f"Zoom: members between 0 and {ZX1}% · {int((vm_ < 2).sum())} of {len(vm_)} below 2%, highest {nome_(vm_.idxmax())} {vm_.max():.0f}%"),
        ha="left", va="bottom", fontsize=10, color=c.TENUE, style="italic")
# riquadro sulla scala comune e raccordi verso la lente
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((ZX0 - .6, Y_M - .3), ZX1 + 1.2, .6, fill=False, ec=c.GRIGIO, lw=.9, ls=(0, (3, 2)), zorder=2))
for yy in (Y_M - .3, Y_M + .3):
    ax.plot([ZX1 + .6, 22], [yy, Y_M - .42 if yy < Y_M else Y_M + .2], color=c.GRIGIO, lw=.8, ls=(0, (3, 2)), zorder=1)
ax.set_ylim(-.6, Y_A + .6); ax.set_xlim(-1, 100)
ax.set_yticks([Y_A, Y_M]); ax.set_yticklabels([f"Non membri\n({len(va_)} paesi)", f"Membri del\nMovimento ({len(vm_)})"], fontsize=12)
ax.get_yticklabels()[1].set_color(c.ACCENTO); ax.get_yticklabels()[1].set_fontweight("bold")
ax.set_xticks(range(0, 101, 20)); ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)])
ax.grid(axis="x", color=c.LINEA, lw=.8); ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
ax.set_xlabel("Quando gli USA votano contro la maggioranza: % di volte in cui il paese vota come loro")
for sp in ("left", "top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Quando gli USA restano in minoranza, i loro alleati li seguono; i non allineati no", loc="left")
R.figura(c.salva(fig, OUT, "chi_segue_gli_usa"), "24 (la slide usa figure_slide/slide24_us.py)")

R.salva(OUT, "report_extra.txt")
