"""Slide 17 - Cosa distingue i quattro gruppi (Kruskal-Wallis sulla metà B delle votazioni).

Legge i test calcolati da atto3_livello1_mondo.py (output/atto3_livello1_mondo/
test_kruskal_wallis.csv): mediana di ogni variabile per gruppo ed epsilon quadro.
Si mostrano le 12 variabili con effetto grande (epsilon quadro >= 0,26); restano fuori
l'assenza e l'astensione su diritti umani, sviluppo e colonialismo.
Output: figure/slide17_profili.png
"""
import pandas as pd
from matplotlib.lines import Line2D

import base

kw = pd.read_csv(base.ANALISI / "atto3_livello1_mondo" / "test_kruskal_wallis.csv", index_col=0)
kw = kw.drop(index="Assenza").sort_values("eps2", ascending=False)
kw = kw[kw.eps2 >= 0.26]
print(kw[["eps2", "p_BH"]].round(4))
print("variabili con p corretto < 0,001:", (pd.read_csv(base.ANALISI / "atto3_livello1_mondo" /
                                                     "test_kruskal_wallis.csv").p_BH < 0.001).sum(), "su 16")

NOMI = {"Accordo con USA": "Agreement with the US", "Accordo con Cina": "Agreement with China",
        "Accordo con Russia": "Agreement with Russia", "Sì": "Yes", "Astenuto": "Abstain",
        "Diritti umani": "human rights", "Disarmo": "disarmament", "Armi nucleari": "nuclear weapons",
        "Colonialismo": "colonialism", "Medio Oriente": "Middle East", "Sviluppo economico": "development"}


def inglese(v):
    if " · " in v:
        a, b = v.split(" · ")
        return f"{NOMI[a]} · {NOMI[b]}"
    return NOMI[v]


GRUPPI = [("Maggioranza del Sud globale", "Global South (107)", "#E76F51", "#B8452C"),
          ("Europa e alleati", "Europe and allies (47)", "#7B6BA8", "#6A5A98"),
          ("Blocco Cina–Russia", "China–Russia (11)", "#E9B949", "#8F6C12"),
          ("USA, Israele e alleati stretti", "US–Israel (7)", "#33415E", "#33415E")]

plt = base.stile()
fig = plt.figure(figsize=(12.15, 7.285), dpi=100)
fig.patch.set_facecolor(base.SFONDO)
ax = fig.add_axes([.22, .17, .575, .76], facecolor=base.SFONDO)
bx = fig.add_axes([.815, .17, .13, .76], facecolor=base.SFONDO, sharey=ax)
n = len(kw)
for i, (var, r) in enumerate(kw.iterrows()):
    y = n - 1 - i
    if i % 2:
        ax.axhspan(y - .5, y + .5, color="#EFF1F4", zorder=0, lw=0)
    valori = [r[g] * 100 for g, *_ in GRUPPI]
    ax.plot([min(valori), max(valori)], [y, y], color="#CDD1D8", lw=2, zorder=1)
    for (g, _, col, _), v in zip(GRUPPI, valori):
        ax.scatter(v, y, s=120, color=col, edgecolor="white", lw=1.2, zorder=3)
    if i == 0:                                    # la variabile principale: valori scritti sopra i punti
        for (g, _, col, ct), v in zip(GRUPPI, valori):
            ax.text(v, y + .55, f"{v:.0f}%", ha="center", va="bottom", fontsize=12, color=ct, fontweight="bold")
    ax.text(-2, y, inglese(var), ha="right", va="center", fontsize=13, color=base.NAVY)
    bx.barh(y, r.eps2, height=.5, color=base.GRIGIO_LINEA)
    bx.text(r.eps2 + .01, y, f"{r.eps2:.2f}", va="center", fontsize=12, color=base.GRIGIO_TESTO)

ax.set_xlim(-2, 102)
ax.set_ylim(-.6, n - .1 + .6)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.tick_params(axis="x", labelsize=12, colors=base.GRIGIO_TESTO, length=0)
ax.set_yticks([])
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(base.GRIGIO_ASSE)
ax.set_xlabel("Group median", fontsize=12.5, color=base.GRIGIO_TESTO, labelpad=6)

bx.axvline(0.26, color=base.NAVY, lw=1.2, ls=(0, (4, 3)))
bx.set_xlim(0, 0.85)
bx.axis("off")
bx.text(0, n + .25, "effect size ε²", fontsize=13, color=base.NAVY, fontweight="bold", va="bottom")
bx.text(0.27, -1.0, "large ≥ 0.26", fontsize=12, color=base.NAVY, va="center")

simboli = [Line2D([], [], marker="o", ls="", ms=11, color=col, markeredgecolor="white") for _, _, col, _ in GRUPPI]
fig.legend(simboli, [lab for _, lab, _, _ in GRUPPI], loc="lower center", ncol=4, frameon=False,
           fontsize=12, bbox_to_anchor=(.55, .0), handletextpad=.3, columnspacing=1.6)
base.salva(fig, "slide17_profili.png")
