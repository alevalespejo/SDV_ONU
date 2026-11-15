"""
Figure delle slide 15, 19 e 22, da eseguire dopo atto3_livello1 e atto3_livello2.
  1. umap: mappa UMAP (n_neighbors = 15, seed = 0) con i gruppi e tre casi cerchiati (slide 15)
  2. umap_miniature: le versioni con 5, 15 e 50 vicini (appendice)
  3. ponte_cluster_rete: dai 4 gruppi k-means alle comunità della rete (slide 19)
  4. seguire_in_minoranza: membro mediano, atteso per caso, non membro mediano (appendice;
     la slide 22 usa figure_slide/slide22_follow.py)
Robustezza di UMAP: per ogni combinazione di parametri, quota di paesi il cui gruppo k-means
è indovinato dai 5 vicini sulla mappa (classificatore 5-NN, validazione a 5 blocchi).
Output: output/figure_aggiuntive/ (report, figure)
"""
import pandas as pd
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import comune as c

OUT = c.cartella_atto("figure_aggiuntive")
plt = c.stile()
L1, L2 = c.OUTPUT / "atto3_livello1_mondo", c.OUTPUT / "atto3_livello2_non_allineati"
CL = pd.read_csv(L1 / "clusters.csv", index_col=0)
U = pd.read_csv(L1 / "umap_griglia.csv")
R = c.Report("REPORT FIGURE AGGIUNTIVE: UMAP, PONTE CLUSTER-RETE, SEGUIRE IN MINORANZA")
EN = {"Maggioranza del Sud globale": "Global South", "Europa e alleati": "Europe and allies",
      "Blocco Cina–Russia": "China–Russia", "USA, Israele e alleati stretti": "US–Israel"}
S = 1 / c.SCALA_TESTO            # le dimensioni dei testi qui sotto sono quelle finali sulla figura: salva() le moltiplica per 1,3

# ------------------------------------------------------------------ 1-2. UMAP
R.sezione("UMAP singola e robustezza")
U["g"] = U.iso3.map(CL.gruppo)
acc = {}
for (nn, s), d in U.groupby(["nn", "seed"]):
    d = d.dropna(subset=["g"])
    acc[(nn, s)] = cross_val_score(KNeighborsClassifier(5), d[["x", "y"]].values, d.g.values, cv=5).mean()
    R.voce(f"n_neighbors = {nn}, seed = {s}", f"5-NN accuratezza {acc[(nn, s)]:.1%}")
R.voce("Minimo / massimo sulle 9 versioni", f"{min(acc.values()):.1%} / {max(acc.values()):.1%}")
R.dati["umap_knn"] = {"min": float(min(acc.values())), "max": float(max(acc.values()))}

g = U[(U.nn == 15) & (U.seed == 0)].dropna(subset=["g"])
fig, ax = plt.subplots(figsize=(10, 7))
for k, sub in g.groupby("g"):
    ax.scatter(sub.x, sub.y, s=230, color=c.GCOL[k], edgecolor=c.SFONDO, linewidth=1.3, zorder=3)
# etichette dirette, posizionate rispetto all'ingombro di ogni gruppo
for k in c.GCOL:
    sub = g[g.g == k]; n = len(sub)
    if k == "Europa e alleati":
        x_, y_, ha = sub.x.max() + .7, sub.y.quantile(.3), "left"
    elif k == "USA, Israele e alleati stretti":
        core = sub[sub.x < sub.x.median() + 2]
        x_, y_, ha = core.x.max() + .7, core.y.max() + .2, "left"
    elif k == "Blocco Cina–Russia":
        x_, y_, ha = sub.x.max() + 1.1, sub.y.median(), "left"
    else:
        x_, y_, ha = sub.x.max() + 1.1, sub.y.quantile(.8), "left"
    col_t = c.TESTO_OCRA if k == "Blocco Cina–Russia" else c.GCOL[k]
    ax.text(x_, y_, EN[k], color=col_t, fontsize=23 * S, fontweight="bold", ha=ha, va="bottom")
    ax.text(x_, y_ - .08, f"{n} states", color=c.TENUE, fontsize=17 * S, ha=ha, va="top")
xr = g.x.max() - g.x.min(); yr = g.y.max() - g.y.min()
for iso, name, dx, dy in [("PAK", "Pakistan", -.2, -.15), ("ZAF", "South Africa", -.2, .13), ("MLI", "Mali", -.25, -.01)]:
    r_ = g[g.iso3 == iso].iloc[0]
    ax.scatter(r_.x, r_.y, s=430, facecolor="none", edgecolor=c.INCHIOSTRO, linewidth=2.2, zorder=4)
    ax.annotate(name, (r_.x, r_.y), xytext=(r_.x + dx * xr, r_.y + dy * yr), ha="right", va="center", fontsize=18 * S,
                color=c.INCHIOSTRO, fontweight="semibold", arrowprops=dict(arrowstyle="-", color=c.INCHIOSTRO, lw=1, shrinkB=7))
ax.set_xlim(g.x.min() - .06 * xr, g.x.max() + .34 * xr); ax.set_ylim(g.y.min() - .06 * yr, g.y.max() + .14 * yr)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("UMAP (n_neighbors = 15, seed = 0)")
R.figura(c.salva(fig, OUT, "umap"), 15)

fig, axs = plt.subplots(1, 3, figsize=(9, 3.4))
for ax, nn in zip(axs, [5, 15, 50]):
    d = U[(U.nn == nn) & (U.seed == 0)].dropna(subset=["g"])
    ax.scatter(d.x, d.y, s=9, c=d.g.map(c.GCOL), linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(c.LINEA)
    ax.set_facecolor(c.SCHEDA)
    ax.set_xlabel(f"{nn} neighbours", fontsize=14 * S, color=c.TENUE)
fig.tight_layout(w_pad=1.2)
R.figura(c.salva(fig, OUT, "umap_miniature"), "appendice")

# ------------------------------------------------------------------ 3. ponte cluster -> rete
R.sezione("Ponte tra cluster k-means e comunità della rete")
CC_EN = {"Occidente, USA e Israele": "West, US & Israel", "Cina, Russia, India, Iran…": "China, Russia, India, Iran…",
         "Sud globale": "Global South", "America Latina e Pacifico": "Latin America & Pacific", "Senza legami forti": "No strong ties"}
Lord = ["USA, Israele e alleati stretti", "Europa e alleati", "Blocco Cina–Russia", "Maggioranza del Sud globale"]
Rord = ["Occidente, USA e Israele", "Cina, Russia, India, Iran…", "Sud globale", "America Latina e Pacifico", "Senza legami forti"]
ct = pd.crosstab(CL.gruppo, CL.comunita).reindex(index=Lord, columns=Rord, fill_value=0)
for l in Lord:
    R.voce(l, ", ".join(f"{r} {int(ct.loc[l, r])}" for r in Rord if ct.loc[l, r]))
attraversano = int(ct.loc[["Europa e alleati", "USA, Israele e alleati stretti"], ["Sud globale", "America Latina e Pacifico"]].values.sum()
                   + ct.loc["Maggioranza del Sud globale", "Occidente, USA e Israele"])
R.voce("Paesi che passano tra Occidente e Sud globale", str(attraversano))
R.dati["ponte"] = {l: {r: int(ct.loc[l, r]) for r in Rord} for l in Lord}
CCOLP = dict(c.CCOL); CCOLP["Senza legami forti"] = c.GRIGIO
fig, ax = plt.subplots(figsize=(15, 6.6)); ax.axis("off")
tot = int(ct.values.sum()); top, bot, gap = 0, 100, 2.2
H = (bot - top - gap * 4) / tot
xl, xr_, w = 30, 70, 1.6
def colonna(ordine, size):
    y = top; p = {}
    for k in ordine:
        p[k] = y; y += size[k] * H + gap
    return p
Ls, Rs = ct.sum(1), ct.sum(0)
pl, pr = colonna(Lord, Ls), colonna(Rord, Rs)
oL, oR = dict(pl), dict(pr)
for l in Lord:
    for r in Rord:
        n = int(ct.loc[l, r])
        if not n:
            continue
        h = n * H; y1, y2 = oL[l], oR[r]; xm = (xl + w + xr_) / 2
        verts = [(xl + w, y1), (xm, y1), (xm, y2), (xr_, y2), (xr_, y2 + h), (xm, y2 + h), (xm, y1 + h), (xl + w, y1 + h), (xl + w, y1)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=c.GCOL[l], alpha=.32, lw=0))
        oL[l] += h; oR[r] += h
for k in Lord:
    ax.add_patch(Rectangle((xl, pl[k]), w, Ls[k] * H, color=c.GCOL[k]))
    ax.text(xl - 1, pl[k] + Ls[k] * H / 2, f"{EN[k]}  {Ls[k]}", ha="right", va="center", fontsize=17 * S, color=c.INCHIOSTRO, fontweight="semibold")
for k in Rord:
    ax.add_patch(Rectangle((xr_, pr[k]), w, Rs[k] * H, color=CCOLP[k]))
    ax.text(xr_ + w + 1, pr[k] + Rs[k] * H / 2, f"{CC_EN[k]}  {Rs[k]}", ha="left", va="center", fontsize=17 * S,
            color=c.TENUE if k == "Senza legami forti" else c.INCHIOSTRO, fontweight="semibold")
ax.text(xl + w / 2, -4, "k-means clusters\nwhere a country sits", ha="center", va="bottom", fontsize=14 * S, color=c.TENUE)
ax.text(xr_ + w / 2, -4, "network communities\nwho it votes with most", ha="center", va="bottom", fontsize=14 * S, color=c.TENUE)
ax.set_xlim(0, 100); ax.set_ylim(bot + 2, -14)
R.figura(c.salva(fig, OUT, "ponte_cluster_rete"), 19)

# ------------------------------------------------------------------ 4. seguire in minoranza
R.sezione("Seguire una potenza in minoranza")
F = pd.read_csv(L2 / "segue_in_minoranza.csv", index_col=0)
T = pd.read_csv(L2 / "segue_vs_caso.csv", index_col=[0, 1])
NAM = [i for i in F.index if i in c.NAM]; ALTRI = [i for i in F.index if i not in c.NAM]
righe = {}
for pot in ["CHN", "RUS", "USA"]:
    t_ = T.loc[pot]
    righe[pot] = dict(membri=F[pot].reindex(NAM).median(), altri=F[pot].reindex(ALTRI).drop(pot, errors="ignore").median(),
                      caso=t_.reindex(NAM).att.median())
    R.voce(c.nome(pot), f"membro mediano {righe[pot]['membri']:.1%} · atteso per caso {righe[pot]['caso']:.1%} · non membro mediano {righe[pot]['altri']:.1%}")
isr = F.loc["ISR", "USA"]; R.voce("Israele segue gli USA", f"{isr:.1%}")
R.dati["seguire"] = {p: {k: float(v) for k, v in d.items()} for p, d in righe.items()}; R.dati["israele_usa"] = float(isr)
NOMI_EN = {"CHN": "China", "RUS": "Russia", "USA": "US"}
fig, ax = plt.subplots(figsize=(11, 6.2))
yy = {"CHN": 2, "RUS": 1, "USA": 0}
for p, y in yy.items():
    r = righe[p]
    ax.plot([r["membri"] * 100, r["altri"] * 100], [y, y], color=c.LINEA, lw=3, zorder=1)
    ax.scatter(r["caso"] * 100, y, s=260, facecolor="none", edgecolor=c.CORALLO, linewidth=2.2, zorder=2)
    ax.scatter(r["altri"] * 100, y, s=200, color=c.COL_ALTRI, zorder=3, edgecolor=c.SFONDO, linewidth=1.5)
    ax.scatter(r["membri"] * 100, y, s=260, color=c.CORALLO, zorder=4, edgecolor=c.SFONDO, linewidth=1.5)
    ax.text(r["membri"] * 100, y - .42, f"{r['membri'] * 100:.0f}%", ha="center", fontsize=17 * S, color=c.ACCENTO_TESTO, fontweight="bold")
    ax.text(r["altri"] * 100, y + .28, f"{r['altri'] * 100:.0f}%", ha="center", fontsize=15 * S, color=c.GRIGIO_SCURO)
ax.scatter(isr * 100, yy["USA"], s=200, marker="D", color=c.INCHIOSTRO, zorder=4)
ax.text(isr * 100, yy["USA"] + .28, f"Israel {isr * 100:.0f}%", ha="center", fontsize=15 * S, color=c.INCHIOSTRO, fontweight="bold")
ax.set_yticks(list(yy.values())); ax.set_yticklabels([f"{NOMI_EN[p]} in minority" for p in yy], fontsize=17 * S)
ax.set_ylim(-.6, 2.7); ax.set_xlim(0, 100)
ax.set_xlabel("% of these votes in which the country votes like the power")
ax.spines["left"].set_visible(False); ax.tick_params(left=False); c.griglia(ax, "x")
h_ = [Line2D([], [], marker="o", ls="", color=c.CORALLO, ms=12, label="median member"),
      Line2D([], [], marker="o", ls="", mfc="none", mec=c.CORALLO, mew=2, ms=12, label="member, by chance"),
      Line2D([], [], marker="o", ls="", color=c.COL_ALTRI, ms=11, label="median non-member")]
ax.legend(handles=h_, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.45, -.36), fontsize=15 * S)
R.figura(c.salva(fig, OUT, "seguire_in_minoranza"), "appendice")

R.salva(OUT, "report_figure_aggiuntive.txt")
