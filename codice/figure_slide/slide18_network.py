"""Slide 18 - Rete di accordo dell'Assemblea Generale, 2014-2025.

Metodo
1. Votazioni senza le 282 "USA contro tutti" (908); Stati presenti in almeno il 75%
   delle 1.190 votazioni (172).
2. Per ogni coppia di Stati, kappa di Cohen su sì / astenuto / no, sulle votazioni in cui
   hanno votato entrambi. Legame forte: kappa >= 0.4.
3. Comunità con Louvain pesato sui kappa, ripetuto con 50 seed: ogni Stato va nella
   comunità in cui finisce più spesso. India e Pakistan sono al confine (32 run su 50
   con Cina-Russia).
4. Disegno: ogni comunità è un disco di punti; gli Stati con legami verso un'altra
   comunità stanno sul lato rivolto verso di essa.
Stampa anche il controllo con soglie 0.3 e 0.5.
Output: figure/slide18_network.png
"""
import collections

import networkx as nx
import numpy as np

import base

df = base.carica()
pres = base.presenza(df)
stati = pres[pres >= 0.75].index
d = df[~df.rcid.isin(base.usa_contro_tutti(df)) & df.Countryname.isin(stati)]
V = d.pivot_table(index="Countryname", columns="rcid", values="vote").reindex(stati)
V = V.where(V.isin([1, 2, 3]))            # assenti = NaN
print("Stati x votazioni:", V.shape)

# kappa di Cohen per ogni coppia
A = V.values
nomi = list(V.index)
n = len(nomi)
K = np.full((n, n), np.nan)
for i in range(n):
    for j in range(i + 1, n):
        entrambi = ~np.isnan(A[i]) & ~np.isnan(A[j])
        x, y = A[i][entrambi], A[j][entrambi]
        po = (x == y).mean()
        pe = sum((x == c).mean() * (y == c).mean() for c in (1, 2, 3))
        K[i, j] = K[j, i] = (po - pe) / (1 - pe)

G = nx.Graph()
G.add_nodes_from(nomi)
for i in range(n):
    for j in range(i + 1, n):
        if K[i, j] >= 0.4:
            G.add_edge(nomi[i], nomi[j], weight=K[i, j])

# comunità: Louvain 50 volte, poi la comunità più frequente per ogni Stato
ancore = {"W": "FRANCE", "C": "CHINA", "L": "CHILE", "S": "NIGERIA"}
conteggi = collections.defaultdict(collections.Counter)
for seed in range(50):
    for c in nx.community.louvain_communities(G, weight="weight", seed=seed):
        lab = [k for k, a in ancore.items() if a in c]
        lab = lab[0] if len(lab) == 1 else "other"
        for x in c:
            conteggi[x][lab] += 1
finale = {x: conteggi[x].most_common(1)[0][0] for x in nomi}
gruppi = {k: {x for x in nomi if finale[x] == k} for k in "SWLC"}
W = gruppi["W"]
gruppo_di = {x: k for k, g in gruppi.items() for x in g}
isolati = [x for x in nomi if finale[x] == "other"]
partizione = [gruppi[k] for k in "SWLC"] + [{x} for x in isolati]

al_confine = {x: dict(conteggi[x]) for x in nomi
              if conteggi[x].most_common(1)[0][1] < 40 and x not in isolati}
print("al confine:", al_confine)
print("dimensioni:", {k: len(g) for k, g in gruppi.items()}, "isolati:", isolati)
print("modularità Q:", round(nx.community.modularity(G, partizione, weight="weight"), 2))
ponti = collections.Counter(tuple(sorted((gruppo_di[a], gruppo_di[b]))) for a, b in G.edges()
                            if gruppo_di.get(a) != gruppo_di.get(b))
print("legami tra comunità:", dict(ponti))
for soglia in (0.3, 0.4, 0.5):
    k = sum(1 for i in range(n) for j in range(i + 1, n)
            if K[i, j] >= soglia and (nomi[i] in W) != (nomi[j] in W)
            and nomi[i] not in isolati and nomi[j] not in isolati)
    print(f"kappa >= {soglia}: legami Occidente-resto = {k}")

# figura
plt = base.stile()
BG = base.SFONDO
COL = {"S": "#E76F51", "W": "#7B6BA8", "L": "#C25A7C", "C": "#E9B949"}
COL_TESTO = {"S": "#B8452C", "W": "#6A5A98", "L": "#A1405F", "C": "#8F6C12"}
centro = {"W": (-3.2, 0.0), "S": (2.5, 0.0), "C": (0.3, 1.9), "L": (0.4, -1.95)}
verso = {"W": "C", "S": "C", "C": "S", "L": "S"}


def disco(k, R):
    """k punti distribuiti in modo uniforme in un cerchio di raggio R (spirale di girasole)."""
    i = np.arange(k) + 0.5
    r = R * np.sqrt(i / k)
    t = i * np.pi * (3 - np.sqrt(5))
    return np.c_[r * np.cos(t), r * np.sin(t)]


pos = {}
for k, g in gruppi.items():
    punti = disco(len(g), 0.17 * np.sqrt(len(g)))
    c0 = np.array(centro[k])
    direzione = np.array(centro[verso[k]]) - c0
    direzione /= np.linalg.norm(direzione)
    ordine_posti = np.argsort(-(punti @ direzione))

    def legami_fuori(x):
        return sum(1 for y in G.neighbors(x) if gruppo_di.get(y) != k)

    for stato, p in zip(sorted(g, key=lambda x: (-legami_fuori(x), x)), ordine_posti):
        pos[stato] = c0 + punti[p]
for i, x in enumerate(isolati):
    pos[x] = np.array([-4.35 + 0.25 * i, -2.55])

fig, ax = base.nuova_figura(plt, 11, 6.4)
for a, b in G.edges():
    ga, gb = gruppo_di.get(a), gruppo_di.get(b)
    if ga == gb:
        ax.plot(*zip(pos[a], pos[b]), color=COL[ga], alpha=.04, lw=.4, zorder=1)
    elif {ga, gb} == {"S", "C"}:
        ax.plot(*zip(pos[a], pos[b]), color=base.NAVY, alpha=.28, lw=.9, zorder=2)
    else:
        ax.plot(*zip(pos[a], pos[b]), color=base.GRIGIO_LINEA, alpha=.6, lw=.8, zorder=2)
for k, g in gruppi.items():
    xy = np.array([pos[x] for x in g])
    ax.scatter(xy[:, 0], xy[:, 1], s=46, color=COL[k], edgecolor=BG, lw=.7, zorder=3)
xy = np.array([pos[x] for x in isolati])
ax.scatter(xy[:, 0], xy[:, 1], s=46, color="#C5CCDA", zorder=3)
ax.text(-3.75, -2.55, "no strong ties: Argentina, Costa Rica", fontsize=15,
        color=base.GRIGIO_TESTO, va="center")

etichette = {"W": ("West, US and Israel", (-3.2, -1.5), "center"),
             "S": ("Global South", (4.15, -0.1), "left"),
             "C": ("China, Russia, India, Iran…", (0.95, 2.4), "left"),
             "L": ("Latin America and Pacific", (1.35, -2.3), "left")}
for k, (t, (x, y), ha) in etichette.items():
    ax.text(x, y, t, fontsize=21, color=COL_TESTO[k], fontweight="bold", ha=ha, va="bottom")
    ax.text(x, y - 0.04, f"{len(gruppi[k])} states", fontsize=17, color=COL_TESTO[k], ha=ha, va="top")
ax.annotate("", xy=(0.75, 0.0), xytext=(-1.75, 0.0),
            arrowprops=dict(arrowstyle="<->", color="#B8452C", lw=1.6, ls=(0, (4, 3))))
ax.text(-0.5, 0.12, "0 ties", fontsize=22, color="#B8452C", fontweight="bold", ha="center", va="bottom")
ax.text(1.62, 1.18, f"{ponti[('C', 'S')]} ties", fontsize=20, color=base.NAVY, fontweight="bold",
        ha="left", va="center", bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none"), zorder=5)
ax.set_xlim(-4.6, 6.0)
ax.set_ylim(-2.85, 2.85)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
base.salva(fig, "slide18_network.png")
