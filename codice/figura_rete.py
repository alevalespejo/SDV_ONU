"""
Rete di accordo disegnata per comunità, con accanto la rete ridotta (appendice; la slide 18
usa figure_slide/slide18_network.py). Legge kappa.csv e clusters.csv di atto3_livello1_mondo.py.

- sinistra: nodi colorati per comunità (Louvain), niente etichette dei paesi, nome della comunità accanto al gruppo;
  archi DENTRO le comunità leggeri, archi TRA comunità scuri: si vede dove sono i ponti e dove non ci sono.
- destra: la stessa rete ridotta a un nodo per comunità, con il numero di legami forti tra comunità.
"""
import json
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import comune as c

c.stile()
OUT = c.OUTPUT / "atto3_livello1_mondo"
SOGLIA = 0.4
K = pd.read_csv(OUT / "kappa.csv", index_col=0)
CL = pd.read_csv(OUT / "clusters.csv", index_col=0)
Q = json.loads((OUT / "risultati.json").read_text()).get("rete", {}).get("Q", 0.51) if (OUT / "risultati.json").exists() else 0.51
ACC = c.ACCENTO

ORD = ["Occidente, USA e Israele", "Cina, Russia, India, Iran…", "Sud globale", "America Latina e Pacifico"]
G = nx.Graph()
G.add_nodes_from(K.index)
for a, b in combinations(K.index, 2):
    if K.loc[a, b] >= SOGLIA:
        G.add_edge(a, b, weight=float(K.loc[a, b]))
com = CL["comunita"].reindex(K.index)
isolati = [n for n in G if com[n] not in ORD]

fig = plt.figure(figsize=(14, 7.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=.04)
ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---- rete nodi-archi
H = G.subgraph([n for n in G if n not in isolati])
# layout per comunità: ogni comunità disposta con un force-directed al suo interno, poi messa nel suo "quartiere"
# (stessa disposizione della rete ridotta a destra), con un raggio proporzionale alla radice del numero di Stati
CENTRI = {"Occidente, USA e Israele": (-1.15, 0.0), "Cina, Russia, India, Iran…": (0.3, 0.95),
          "Sud globale": (1.1, -0.05), "America Latina e Pacifico": (0.25, -1.05)}
nmax = max(int((com == g).sum()) for g in ORD)
pos = {}
for g in ORD:
    sub = H.subgraph([n for n in H if com[n] == g])
    pg = nx.spring_layout(sub, weight="weight", seed=3, k=1.4 / np.sqrt(max(len(sub), 2)), iterations=300)
    arr = np.array(list(pg.values())); arr = arr - np.median(arr, 0)
    nr = np.linalg.norm(arr, axis=1); sc = np.percentile(nr, 85) + 1e-9
    arr = arr / sc; nr2 = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = np.where(nr2 > 1.25, arr / nr2 * 1.25, arr)             # i pochi punti lontani restano vicini al gruppo
    r = .42 * np.sqrt(len(sub) / nmax) + .08
    for n, v in zip(pg.keys(), arr):
        pos[n] = np.array(CENTRI[g]) + v * r
dentro = [(u, v) for u, v in H.edges() if com[u] == com[v]]
tra = [(u, v) for u, v in H.edges() if com[u] != com[v]]
for g in ORD:
    e_g = [(u, v) for u, v in dentro if com[u] == g]
    nx.draw_networkx_edges(H, pos, edgelist=e_g, ax=ax, edge_color=c.CCOL[g], width=.3, alpha=.14)
nx.draw_networkx_edges(H, pos, edgelist=tra, ax=ax, edge_color=c.INCHIOSTRO, width=.7, alpha=.55)
forza = dict(H.degree(weight="weight"))
for g in ORD:
    nodi = [n for n in H if com[n] == g]
    nx.draw_networkx_nodes(H, pos, nodelist=nodi, ax=ax, node_color=c.CCOL[g], edgecolors=c.SFONDO, linewidths=.6,
                           node_size=[12 + 1.6 * forza[n] for n in nodi])
ETI = {"Occidente, USA e Israele": ("below", ), "Cina, Russia, India, Iran…": ("above", ),
       "Sud globale": ("below", ), "America Latina e Pacifico": ("below", )}
for g in ORD:
    p_ = np.array([pos[n] for n in H if com[n] == g]); x0 = p_[:, 0].mean()
    if ETI[g][0] == "above":
        ax.text(x0, p_[:, 1].max() + .1, f"{g}\n{len(p_)} Stati", ha="center", va="bottom", fontsize=14, fontweight="bold", color=(c.TESTO_OCRA if c.CCOL[g] == c.COL_CINA else c.CCOL[g]), linespacing=1.15)
    else:
        ax.text(x0, p_[:, 1].min() - .1, f"{g}\n{len(p_)} Stati", ha="center", va="top", fontsize=14, fontweight="bold", color=(c.TESTO_OCRA if c.CCOL[g] == c.COL_CINA else c.CCOL[g]), linespacing=1.15)
ax.axis("off"); ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.6); ax.set_aspect("equal")

# ---- rete ridotta (riassunto)
cnt = pd.DataFrame(0, index=ORD, columns=ORD)
for u, v in H.edges():
    cnt.loc[com[u], com[v]] += 1
    if com[u] != com[v]:
        cnt.loc[com[v], com[u]] += 1
posc = {"Occidente, USA e Israele": (-1.0, 0.0), "Cina, Russia, India, Iran…": (0.25, 0.78),
        "Sud globale": (1.0, -0.05), "America Latina e Pacifico": (0.15, -0.85)}
nn = {g: int((com == g).sum()) for g in ORD}
rad = {g: .13 + .2 * np.sqrt(nn[g] / max(nn.values())) for g in ORD}
tra_c = {(a, b): int(cnt.loc[a, b]) for a, b in combinations(ORD, 2)}
mx = max(tra_c.values())
for (a, b), v in tra_c.items():
    zero_chiave = v == 0 and a == "Occidente, USA e Israele" and b == "Sud globale"
    if v == 0 and not zero_chiave:
        continue
    (xa, ya), (xb, yb) = posc[a], posc[b]
    if zero_chiave:
        ax2.plot([xa, xb], [ya, yb], color=ACC, lw=2, ls=(0, (3, 3)), zorder=1)
    else:
        ax2.plot([xa, xb], [ya, yb], color="#7F7A70", lw=1.2 + 14 * v / mx, alpha=.75, solid_capstyle="butt", zorder=1)
    ax2.text((xa + xb) / 2, (ya + yb) / 2, str(v), ha="center", va="center", fontsize=17 if zero_chiave else 14,
             fontweight="bold", color=ACC if zero_chiave else c.INCHIOSTRO,
             bbox=dict(boxstyle="round,pad=.25", fc=c.SFONDO, ec="none"), zorder=4)
for g in ORD:
    x, y = posc[g]
    ax2.add_patch(Circle((x, y), rad[g], color=c.CCOL[g], zorder=3))
    ax2.text(x, y, str(nn[g]), ha="center", va="center", fontsize=15, color=(c.INCHIOSTRO if c.CCOL[g] == c.COL_CINA else "white"), fontweight="bold", zorder=5)
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.4, 1.25); ax2.set_aspect("equal"); ax2.axis("off")
c.salva(fig, OUT, "rete_comunita")
print("ponti:", tra_c, "isolati:", isolati)
