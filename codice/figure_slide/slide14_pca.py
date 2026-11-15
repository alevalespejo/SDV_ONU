"""Slide 14 - PCA: la mappa dei 172 Stati e i quattro blocchi di voto.

Senza le 282 votazioni "USA contro tutti"; solo Stati presenti in almeno il 75%
delle votazioni; voti mancanti sostituiti con la media della votazione.
I blocchi sono quelli di k-means con k = 4 sulle stesse coordinate.
Output: figure/slide14_pca.png, figure/pca_scores.csv
"""
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import MultiPoint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import base

# dati
df = base.carica()
d = df[~df.rcid.isin(base.usa_contro_tutti(df))]
M = d.pivot_table(index="Countryname", columns="rcid", values="vote_num")
pres = base.presenza(df)
X = M.loc[pres[pres >= 0.75].index]
X = X.fillna(X.mean())
Xc = X - X.mean()

pca = PCA().fit(Xc)
S = pca.transform(Xc)[:, :2]
print("Stati:", len(X), " varianza PC1, PC2:", pca.explained_variance_ratio_[:2].round(3))

# di cosa è fatta ogni componente: quota dei loading al quadrato per tema
# (una votazione con più temi divide il suo peso in parti uguali)
temi = df.drop_duplicates("rcid").set_index("rcid").loc[X.columns, ["hr", "di", "nu", "me", "ec", "co"]]
peso_temi = temi.div(temi.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
for i, pc in enumerate(["PC1", "PC2"]):
    l2 = pd.Series(pca.components_[i] ** 2, index=X.columns)
    quote = {t: (l2 * peso_temi[t]).sum() / l2.sum() for t in peso_temi}
    quote["senza tema"] = l2[temi.sum(axis=1) == 0].sum() / l2.sum()
    print(pc, {t: f"{q:.0%}" for t, q in sorted(quote.items(), key=lambda x: -x[1])})

cluster = KMeans(4, n_init=20, random_state=0).fit_predict(Xc)
o = pd.DataFrame({"pc1": S[:, 0], "pc2": S[:, 1], "k": cluster}, index=X.index)
for k, g in o.groupby("k"):
    print(k, len(g), round(g.pc1.mean(), 1), round(g.pc2.mean(), 1), list(g.index[:6]))
o.to_csv(base.OUT / "pca_scores.csv")

# nome dei blocchi: il numero di k-means cambia, la posizione no
dim = o.groupby("k").size()
medie = o.groupby("k")[["pc1", "pc2"]].mean()
nome = {}
for k, m in medie.iterrows():
    if dim[k] > 100:
        nome[k] = "gs"            # Sud globale
    elif m.pc1 < 0 and dim[k] > 20:
        nome[k] = "eu"            # Europa e alleati
    elif m.pc1 < 0:
        nome[k] = "ui"            # USA-Israele
    else:
        nome[k] = "cr"            # Cina-Russia
o["b"] = o.k.map(nome)

# figura
plt = base.stile()
plt.rcParams.update({"font.size": 14, "axes.edgecolor": base.GRIGIO_ASSE})
BG = base.SFONDO
COL = {"gs": "#E76F51", "eu": "#7B6BA8", "cr": "#E9B949", "ui": "#33415E"}
COL_TESTO = {"gs": "#B8452C", "eu": "#6A5A98", "cr": "#8F6C12", "ui": "#33415E"}
ETICHETTA = {"gs": "Global South\nmajority · %d", "eu": "Europe and allies · %d",
             "cr": "China–Russia\nbloc · %d", "ui": "US–Israel bloc · %d"}

fig, ax = base.nuova_figura(plt, 11, 6)
for b in ["eu", "ui", "gs", "cr"]:
    s = o[o.b == b]
    area = MultiPoint(list(map(tuple, s[["pc1", "pc2"]].values))).convex_hull.buffer(1.5, resolution=24)
    bordo = np.array(area.exterior.coords)
    ax.fill(bordo[:, 0], bordo[:, 1], color=COL[b], alpha=.13, lw=0, zorder=1)
    ax.plot(bordo[:, 0], bordo[:, 1], color=COL[b], alpha=.35, lw=1, zorder=1)
    ax.scatter(s.pc1, s.pc2, s=46, color=COL[b], edgecolor=BG, lw=.8, zorder=3)

posizioni = {"gs": (15, 4, "left"), "eu": (-21, -10, "center"),
             "cr": (17, -12, "left"), "ui": (-30.5, 21.7, "left")}
for b, (x, y, ha) in posizioni.items():
    ax.text(x, y, ETICHETTA[b] % (o.b == b).sum(), color=COL_TESTO[b], fontsize=17,
            fontweight="bold", ha=ha, va="center", linespacing=1.15)
for stato, lab, dx, dy, ha in [("RUSSIAN FEDERATION", "Russia", 1.2, 0, "left"),
                               ("CHINA", "China", 1.2, -.6, "left"),
                               ("UNITED STATES", "US", 0, 1.7, "center")]:
    r = o.loc[stato]
    ax.text(r.pc1 + dx, r.pc2 + dy, lab, fontsize=14, color=base.NAVY, ha=ha, va="center", zorder=4)

ax.axhline(0, color="#DCE0E6", lw=1, zorder=0)
ax.axvline(0, color="#DCE0E6", lw=1, zorder=0)
ax.set_xlim(-32, 30)
ax.set_ylim(-22, 24.5)
for s in ["top", "right", "bottom"]:
    ax.spines[s].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.text(-30.5, 19.6, "set apart by PC2 (8%): the Middle East", fontsize=13.5,
        color=base.GRIGIO_TESTO, ha="left", va="center")
ax.text(-31.6, 24.3, "PC2 ↑", fontsize=13, color=base.GRIGIO_TESTO, ha="left", va="center")
fig.subplots_adjust(left=.02, right=.99, top=.97, bottom=.13)

# asse di PC1 disegnato sotto il grafico, con i due estremi
fig.add_artist(FancyArrowPatch((0.03, 0.10), (0.985, 0.10), transform=fig.transFigure,
                               arrowstyle="<|-|>", mutation_scale=18, color=base.NAVY, lw=1.6))
fig.text(0.03, 0.045, "◀ Europe and allies", fontsize=14, color=base.GRIGIO_TESTO, ha="left", va="center")
fig.text(0.985, 0.045, "Global South ▶", fontsize=14, color=base.GRIGIO_TESTO, ha="right", va="center")
fig.text(0.5, 0.045, "main divide · PC1 · 60% of variance", fontsize=16, color=base.NAVY,
         fontweight="bold", ha="center", va="center")
base.salva(fig, "slide14_pca.png")
