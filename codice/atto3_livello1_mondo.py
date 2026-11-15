"""Parte 2 - Mondo (slide 12-19). Q1: ci sono blocchi di voto? I non allineati sono della Cina?

Allineamento con la maggioranza per anno, PCA, UMAP, k-means costruito sulla metà A
delle votazioni e descritto sulla metà B, validazione con l'appartenenza al Movimento,
profili dei gruppi (Kruskal-Wallis), rete di accordo (kappa di Cohen, Louvain).
Tutto senza le 282 votazioni "USA contro tutti".
Output: output/atto3_livello1_mondo/ (report, figure, clusters.csv, pcs.csv, kappa.csv,
        test_kruskal_wallis.csv, letti dalle parti successive)
"""
import numpy as np
import pandas as pd
import networkx as nx
import community as louvain
import umap
from scipy.stats import kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import trustworthiness
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import comune as c

OUT = c.cartella_atto("atto3_livello1_mondo")
plt = c.stile()
df, P, W, meta = c.dati()

# ogni test e ogni confronto è fatto senza le votazioni "USA contro tutti"
W_TUTTE, META_TUTTE = W, meta
W, meta = W[meta.solo_usa == 0], meta[meta.solo_usa == 0]
R = c.Report("REPORT ATTO 3 - LIVELLO 1 (MONDO): CHE FORMA HA L'ASSEMBLEA?")
anni = list(range(2014, 2026))

# --------------------------------------------- 1. allineamento con la maggioranza
R.sezione("Allineamento con la maggioranza nel tempo (slide 13)")
am = W.eq(meta.maj, axis=0).where(W.notna()).groupby(meta.year).mean().T
am_no = (W[meta.solo_usa == 0].eq(meta.maj[meta.solo_usa == 0], axis=0)
         .where(W[meta.solo_usa == 0].notna()).groupby(meta.year[meta.solo_usa == 0]).mean().T)
for iso in ["CHN", "RUS", "USA", "GBR", "FRA", "IND", "BRA", "ARG"]:
    R.voce(f"{c.nome(iso)}: min-max 2014-24 / 2025", f"{c.pct(am.loc[iso, :2024].min(), 0)}-"
           f"{c.pct(am.loc[iso, :2024].max(), 0)} / {c.pct(am.loc[iso, 2025], 0)}")
R.voce("Stato mediano 2024 / 2025", f"{c.pct(am[2024].median(), 0)} / {c.pct(am[2025].median(), 0)}")
R.voce("Stato mediano 2025 senza 'USA contro tutti'", c.pct(am_no[2025].median(), 0))
R.lettura("Senza le votazioni 'USA contro tutti': la Cina vota spesso con la maggioranza; gli USA sempre meno, fino al 13% nel 2025 (5% contandole).")
R.dati["allineamento"] = {iso: {int(a): float(v) for a, v in am.loc[iso].items()} for iso in ["CHN", "RUS", "USA", "ARG"]}

fig, ax = plt.subplots(figsize=(9, 5))
for iso in P.index[P.g20 == 1]:
    if iso not in ["USA", "CHN", "RUS"]:
        ax.plot(anni, am.loc[iso, anni] * 100, color=c.GRIGIO_CHIARO, lw=1.2)
med_nam_am = am.loc[[i for i in am.index if i in c.NAM]].median()
R.voce("Membro mediano del Movimento con la maggioranza, min-max", f"{c.pct(med_nam_am.min(), 0)}-{c.pct(med_nam_am.max(), 0)}")
R.dati["allineamento_membro_mediano"] = {int(a): float(v) for a, v in med_nam_am.items()}
ax.plot(anni, med_nam_am[anni] * 100, color=c.CORALLO, lw=3.6, zorder=5)
for iso, ls_ in [("CHN", "-"), ("RUS", (0, (4, 2))), ("USA", "-")]:
    ax.plot(anni, am.loc[iso, anni] * 100, color=c.COL_POTENZE[iso], lw=2.6, ls=ls_, zorder=4)
ytxt = {i: am.loc[i, 2025] * 100 for i in ["CHN", "RUS", "USA"]}
for iso in ["CHN", "RUS", "USA"]:
    ax.text(2025.3, ytxt[iso], c.nome(iso), va="center", fontsize=11, fontweight="bold",
            color=c.TESTO_OCRA if iso != "USA" else c.COL_USA)
ax.text(2025.3, 97, c.T("membro mediano\ndel Movimento", "median Movement\nmember"), va="center", fontsize=10, fontweight="bold", color=c.ACCENTO_TESTO, linespacing=1.05)
ax.text(2025.3, 78.5, "altri G20", va="center", fontsize=9.5, color=c.GRIGIO_SCURO)
for a, t, y in [(2014, "Crimea", 15), (2022, "Invasione\ndell'Ucraina", 15)]:
    ax.axvline(a, color=c.GRIGIO, ls=":", lw=1); ax.text(a + .08, 3, t, fontsize=8.5, color=c.TENUE, va="bottom")
ax.set_ylim(0, 102); ax.set_xlim(2013.8, 2027.6); ax.set_xticks(anni[::2]); c.griglia(ax)
ax.set_ylabel("% di voti uguali alla maggioranza")
ax.set_title("La Cina vota con la maggioranza, gli USA sempre meno (grigio: altri G20)")
R.figura(c.salva(fig, OUT, "allineamento_maggioranza"), 13)

# ----------------------------------------------------------------- 2. PCA
R.sezione("PCA sulla matrice Stati x votazioni (slide 14)")
X = W.T[W_TUTTE.notna().mean() >= .75]   # 172 Stati: presenza calcolata su tutte le votazioni
Xi = X.fillna(X.mean())
pca = PCA(random_state=c.SEED).fit(Xi); Z = pca.transform(Xi); ev = pca.explained_variance_ratio_
R.voce("Stati / votazioni", f"{X.shape[0]} / {X.shape[1]}")
R.voce("Scaling", "centrato, non standardizzato (+1/0/-1 già su scala comune)")
R.voce("Valori mancanti", f"{X.isna().values.mean():.1%} delle celle, sostituiti con la media della votazione")
R.voce("Varianza PC1 / PC2 / PC3", f"{c.pct(ev[0])} / {c.pct(ev[1])} / {c.pct(ev[2])}")
R.voce("Varianza cumulata prime 35 PC", c.pct(ev[:35].sum()))
L = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
temi_pc = {}
for pc in ["PC1", "PC2"]:
    top = L[pc].abs().sort_values(ascending=False).head(100).index
    temi_pc[pc] = meta.loc[top, list(c.TEMI)].sum().to_dict()
    R.voce(f"{pc}: temi delle 100 votazioni più pesanti", ", ".join(f"{c.TEMI[k]} {v}" for k, v in temi_pc[pc].items()))
# peso medio dei temi sulle prime due componenti (|loading| medio delle votazioni di ciascun tema)
pesi = pd.DataFrame({c.TEMI[t]: L.loc[meta.index[meta[t] == 1].intersection(L.index)].abs().mean() for t in c.TEMI}).T
pesi.to_csv(OUT / "pca_pesi_temi.csv")
R.voce("|loading| medio per tema, PC1", ", ".join(f"{k} {v:.4f}" for k, v in pesi.PC1.sort_values(ascending=False).items()))
R.voce("|loading| medio per tema, PC2", ", ".join(f"{k} {v:.4f}" for k, v in pesi.PC2.sort_values(ascending=False).items()))
R.dati["pca_pesi_temi"] = pesi.to_dict()
PC = pd.DataFrame(Z[:, :35], index=X.index, columns=[f"PC{i + 1}" for i in range(35)])
PC.to_csv(OUT / "pcs.csv")
R.lettura("Una frattura principale (PC1): Occidente contro il resto, soprattutto su diritti umani, "
          "nucleare e disarmo. La PC2 riguarda il Medio Oriente e isola USA e Israele.")
R.dati["pca"] = dict(pc1=ev[0], pc2=ev[1], cum35=ev[:35].sum())

ordine_t = pesi.PC1.sort_values().index
fig, ax = plt.subplots(figsize=(9, 4.6))
yy_t = np.arange(len(ordine_t))
ax.barh(yy_t + .19, pesi.loc[ordine_t, "PC1"] * 1000, height=.36, color=c.INCHIOSTRO, label=f"PC1 ({ev[0]:.0%} della varianza)")
ax.barh(yy_t - .19, pesi.loc[ordine_t, "PC2"] * 1000, height=.36, color=c.GRIGIO, label=f"PC2 ({ev[1]:.0%})")
ax.set_yticks(yy_t); ax.set_yticklabels(ordine_t)
ax.set_xlabel("Peso medio delle votazioni del tema sulla componente (|loading| × 1000)")
ax.legend(frameon=False, loc="upper right"); c.griglia(ax, "x")
ax.set_title("PC1 è nucleare, diritti umani e disarmo; PC2 è il Medio Oriente")
R.figura(c.salva(fig, OUT, "pca_pesi_temi"), "appendice")

# ------------------------------------- clustering (calcolo; il report è nella sezione 4)
# calcolato prima delle figure PCA e UMAP, che sono colorate per cluster
A, B = c.dividi_votazioni(META_TUTTE)   # divisione su tutte le votazioni, poi tolte le 282
A = [v for v in A if v in meta.index]; B = [v for v in B if v in meta.index]
XA = Xi[A]; ZA = PCA(n_components=10, random_state=c.SEED).fit_transform(XA)
righe = {}
for k in range(2, 8):
    km = KMeans(k, n_init=50, random_state=0).fit(ZA)
    aris = []
    for b in range(100):
        r = np.random.default_rng(b); ii = r.choice(len(ZA), len(ZA), replace=True)
        aris.append(adjusted_rand_score(km.labels_, KMeans(k, n_init=10, random_state=b).fit(ZA[ii]).predict(ZA)))
    w = AgglomerativeClustering(k, linkage="ward").fit(ZA).labels_
    righe[k] = dict(silhouette=silhouette_score(ZA, km.labels_), ari_bootstrap=np.median(aris),
                    ari_ward=adjusted_rand_score(km.labels_, w),
                    dimensioni=sorted(np.bincount(km.labels_).tolist(), reverse=True))
K = pd.DataFrame(righe).T
K.to_csv(OUT / "scelta_k.csv")
K_SCELTO = 4
lab = KMeans(K_SCELTO, n_init=50, random_state=0).fit(ZA).labels_
cl = pd.Series(lab, index=X.index)
nomi_cl = {}
for g in range(K_SCELTO):
    m = set(cl[cl == g].index)
    nomi_cl[g] = ("USA, Israele e alleati stretti" if "USA" in m else "Blocco Cina–Russia" if "CHN" in m
                  else "Europa e alleati" if "DEU" in m else "Maggioranza del Sud globale")
CL = pd.DataFrame({"cluster": cl, "gruppo": cl.map(nomi_cl), "nam": P.nam, "continente": P.continente}).loc[X.index]
J = {g: [] for g in range(K_SCELTO)}
for b in range(200):
    r = np.random.default_rng(b); ii = np.unique(r.choice(len(ZA), len(ZA), replace=True))
    lb = KMeans(K_SCELTO, n_init=10, random_state=b).fit(ZA[ii]).labels_
    for g in range(K_SCELTO):
        S1 = set(ii[lab[ii] == g])
        J[g].append(max(len(S1 & set(ii[lb == d])) / len(S1 | set(ii[lb == d])) for d in range(K_SCELTO)))

GCOL = c.GCOL



# figura PCA
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 2.3]})
axs[0].bar(range(1, 11), ev[:10] * 100, color=[c.EVIDENZA] + [c.GRIGIO] * 9, width=.7)
axs[0].set_xticks(range(1, 11)); axs[0].set_xlabel("Componente"); axs[0].set_ylabel("% varianza spiegata")
axs[0].set_title("Scree plot")
ax = axs[1]
for g, col in GCOL.items():
    ii = CL.index[CL.gruppo == g]
    ax.scatter(PC.loc[ii, "PC1"], PC.loc[ii, "PC2"], s=40, color=col, edgecolor=c.SFONDO, linewidth=1, label=f"{g} ({len(ii)})")
# Occidente: etichette con linea guida (USA e Israele in alto a sinistra, Giappone sotto la massa viola)
_lg = dict(arrowstyle="-", color=c.TENUE, lw=.6, shrinkA=0, shrinkB=2)
_ovest_min = PC.loc[CL.index[CL.gruppo == "Europa e alleati"], "PC2"].min()
for i, xt, yt, ha in [("USA", PC.loc["USA", "PC1"] - 1.5, PC.loc["USA", "PC2"] + 4.5, "left"),
                      ("ISR", PC.loc["ISR", "PC1"] - 3.5, PC.loc["ISR", "PC2"] - 4.5, "left"),
                      ("JPN", PC.loc["JPN", "PC1"], _ovest_min - 4.5, "center")]:
    ax.annotate(c.nome(i), (PC.loc[i, "PC1"], PC.loc[i, "PC2"]), xytext=(xt, yt), textcoords="data", fontsize=9.5,
                ha=ha, va="center", color=c.INCHIOSTRO, arrowprops=_lg)
# massa densa del Sud globale: etichette in colonna a destra, con linee guida
densi = ["RUS", "CHN", "PAK", "MLI", "ZAF"]
yy_l = np.linspace(7, -9, len(densi))
for i, yl in zip(sorted(densi, key=lambda i: -PC.loc[i, "PC2"]), yy_l):
    ax.annotate(c.nome(i), (PC.loc[i, "PC1"], PC.loc[i, "PC2"]), xytext=(17.5, yl), textcoords="data", fontsize=9.5,
                va="center", color=c.INCHIOSTRO, arrowprops=dict(arrowstyle="-", color=c.TENUE, lw=.6, shrinkA=0, shrinkB=2))
ax.set_xlim(left=PC.PC1.min() - 5, right=25)
ax.set_xlabel(c.T(f"PC1 ({ev[0] * 100:.0f}%) · pesano soprattutto diritti umani, nucleare, disarmo", f"PC1 ({ev[0] * 100:.0f}%) · driven by human rights, nuclear weapons, disarmament"))
ax.set_ylabel(f"PC2 ({ev[1] * 100:.0f}%) · Medio Oriente")
ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2); ax.set_title("PCA: una frattura principale, Occidente contro il resto")
R.figura(c.salva(fig, OUT, "pca"), "14 (la slide usa figure_slide/slide14_pca.py)")

# ----------------------------------------------------------------- 3. UMAP
R.sezione("UMAP robusto sulle prime 35 PC (slide 15)")
out = []
for nn in [5, 15, 50]:
    for s in [0, 1, 2]:
        E = umap.UMAP(n_neighbors=nn, min_dist=.1, random_state=s).fit_transform(PC.values)
        out.append(pd.DataFrame({"iso3": PC.index, "x": E[:, 0], "y": E[:, 1], "nn": nn, "seed": s}))
U = pd.concat(out); U.to_csv(OUT / "umap_griglia.csv", index=False)
tw = {(nn, s): trustworthiness(PC.values, g[["x", "y"]].values, n_neighbors=10) for (nn, s), g in U.groupby(["nn", "seed"])}
R.voce("Parametri", "n_neighbors in {5, 15, 50} x seed in {0, 1, 2}, min_dist = 0,1, metrica euclidea")
R.voce("Trustworthiness (10 vicini)", f"min {min(tw.values()):.3f} / max {max(tw.values()):.3f}")
R.lettura("In tutte le 9 versioni: due masse (Occidente e resto), con il blocco Cina–Russia al bordo "
          "della massa del Sud. Non si interpretano distanze né dimensioni dei gruppi.")
U["g"] = U.iso3.map(CL.gruppo)
fig, axs = plt.subplots(3, 4, figsize=(14, 10))
for r, nn in enumerate([5, 15, 50]):
    for s in range(3):
        ax = axs[r, s]; d = U[(U.nn == nn) & (U.seed == s)]
        for g, col in GCOL.items():
            dd = d[d.g == g]; ax.scatter(dd.x, dd.y, s=10, color=col)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"n_neighbors={nn}, seed={s}", fontsize=11, fontweight="normal")
    d = U[(U.nn == nn) & (U.seed == 0)]
    axs[r, 3].scatter(d.x, d.y, s=10, color=c.GRIGIO); axs[r, 3].set_xticks([]); axs[r, 3].set_yticks([])
    axs[r, 3].set_title(f"senza colori, n_neighbors={nn}", fontsize=11, fontweight="normal")
fig.legend([Line2D([], [], marker="o", ls="", color=col) for col in GCOL.values()], list(GCOL),
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, 0))
fig.suptitle(f"UMAP sulle prime 35 componenti ({ev[:35].sum() * 100:.0f}% della varianza), min_dist=0,1: la struttura regge a ogni parametro",
             x=.01, ha="left", fontweight="bold", fontsize=13)
R.figura(c.salva(fig, OUT, "umap_griglia"), "appendice")



# ------------------------------------------------- 4. clustering (report)
R.sezione("Clustering senza circolarità: costruzione e scelta di k (slide 16)")
R.voce("Votazioni metà A (costruzione)", f"{len(A)} (scelte a caso, stratificate per anno)")
R.voce("Votazioni metà B (descrizione)", f"{len(B)}")
R.voce("Algoritmo", "k-means (n_init=50) sulle prime 10 PC della metà A; confronto con Ward")
for k, r in K.iterrows():
    R.voce(f"k = {k}", f"silhouette {r.silhouette:.2f} | stabilità bootstrap {r.ari_bootstrap:.2f} | "
           f"accordo Ward {r.ari_ward:.2f} | dimensioni {r.dimensioni}")
R.testo("")
R.voce("k scelto", f"{K_SCELTO} (k = 3 è il più netto; k = 4 separa il blocco Cina–Russia, interpretabile)")
for g in range(K_SCELTO):
    m = cl[cl == g].index
    R.voce(f"{nomi_cl[g]} (n = {len(m)})", f"stabilità Jaccard {np.mean(J[g]):.2f}")
    if len(m) <= 15:
        R.testo(f"      membri: {', '.join(c.nome(i) if i in c.NOMI else P.loc[i, 'Countryname'].title() for i in m)}")
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.axvspan(K_SCELTO - .3, K_SCELTO + .3, color="#FBE0D8", zorder=0)
for col_, nome_, st_, mk_, lab_ in [("silhouette", "Silhouette", "-", "o", "Silhouette"),
                                    ("ari_bootstrap", "Stabilità", "--", "s", "Stabilità bootstrap\n(ARI mediano)"),
                                    ("ari_ward", "Ward", ":", "^", "Accordo con Ward\n(ARI)")]:
    ax.plot(K.index, K[col_], marker=mk_, color=c.INCHIOSTRO if col_ == "silhouette" else c.TENUE, lw=2, ls=st_, ms=7)
    pos_ = {"silhouette": (7.15, K.loc[7, "silhouette"]), "ari_ward": (6.1, 1.035), "ari_bootstrap": (6.15, .63)}[col_]
    ax.text(*pos_, lab_.replace("\n", " ") if col_ != "ari_bootstrap" else lab_, fontsize=10, va="center", color=c.INCHIOSTRO)
for col_ in ["silhouette", "ari_bootstrap", "ari_ward"]:
    ax.scatter([K_SCELTO], [K.loc[K_SCELTO, col_]], s=90, color=c.CORALLO, zorder=5, edgecolor=c.SFONDO, linewidth=1.4)
ax.text(K_SCELTO, 1.05, f"k = {K_SCELTO} scelto", ha="center", fontsize=10, color=c.ACCENTO_TESTO, fontweight="bold")
ax.annotate(c.T(f"accordo\ncon Ward {K.loc[K_SCELTO, 'ari_ward']:.2f}", f"agreement\nwith Ward {K.loc[K_SCELTO, 'ari_ward']:.2f}"), (K_SCELTO, K.loc[K_SCELTO, "ari_ward"]),
            xytext=(2.05, .8), fontsize=9.5, color=c.TENUE, arrowprops=dict(arrowstyle="-", color=c.TENUE, lw=.7))
ax.set_ylim(.3, 1.1); ax.set_xlim(1.7, 8.4); ax.set_xticks(K.index); ax.set_xlabel("k (numero di gruppi)"); c.griglia(ax)
ax.set_title("Scelta di k: 3 è il più netto, 4 separa il blocco Cina–Russia")
R.figura(c.salva(fig, OUT, "scelta_k"), 16)

# ------------------------------------------------ 5. validazione + profili
R.sezione("Validazione esterna e profili dei cluster (slide 16-17)")
def cramer(t):
    chi2, p, _, _ = chi2_contingency(t); n = t.values.sum()
    return chi2, p, np.sqrt(chi2 / (n * (min(t.shape) - 1)))
for v, nm in [("nam", "Movimento Non Allineati"), ("continente", "continente")]:
    t = pd.crosstab(CL.gruppo, CL[v]); chi2, p, V = cramer(t)
    R.voce(f"Cluster x {nm}", f"chi2 = {chi2:.1f}, p = {p:.1e}, V di Cramér = {V:.2f}")
t = pd.crosstab(CL.gruppo, CL.nam)
for g in t.index:
    R.testo(f"      {g:<34} membri NAM {t.loc[g, 1]:>3} | non membri {t.loc[g, 0]:>3}")
R.lettura("I cluster, costruiti solo sui voti, coincidono in gran parte con l'appartenenza al "
          "Movimento: una variabile che l'algoritmo non ha mai visto.")
# test di permutazione: quante volte un'etichetta 'membro' assegnata a caso dà una V così alta?
V_oss = cramer(t)[2]; rng_p = np.random.default_rng(0); N_PERM = 10000
V_perm = np.array([cramer(pd.crosstab(CL.gruppo, rng_p.permutation(CL.nam.values)))[2] for _ in range(N_PERM)])
p_perm = (1 + (V_perm >= V_oss).sum()) / (N_PERM + 1)
R.voce("Test di permutazione su V (10.000 rimescolamenti)", f"V osservata {V_oss:.2f}; V casuali: mediana {np.median(V_perm):.2f}, "
       f"massimo {V_perm.max():.2f}; p = {p_perm:.4f}")
R.dati["cramer_perm"] = dict(V=float(V_oss), V_perm_mediana=float(np.median(V_perm)), V_perm_max=float(V_perm.max()), p=float(p_perm))
# distanze tra i blocchi nello spazio dei voti (prime 10 PC della metà A)
ZAd = pd.DataFrame(ZA, index=X.index)
cent = ZAd.groupby(CL.gruppo).mean()
from itertools import combinations
DIST = {f"{a} - {b}": float(np.linalg.norm(cent.loc[a] - cent.loc[b])) for a, b in combinations(cent.index, 2)}
for k_, v_ in sorted(DIST.items(), key=lambda x: x[1]):
    R.voce(f"Distanza tra centri: {k_}", f"{v_:.1f}")
gs, cr, eu = "Maggioranza del Sud globale", "Blocco Cina–Russia", "Europa e alleati"
d_gs_cr = np.linalg.norm(cent.loc[gs] - cent.loc[cr]); d_gs_eu = np.linalg.norm(cent.loc[gs] - cent.loc[eu])
d_cr_eu = np.linalg.norm(cent.loc[cr] - cent.loc[eu])
R.lettura(f"Il Sud globale è più vicino al blocco Cina–Russia ({d_gs_cr:.1f}) che all'Europa ({d_gs_eu:.1f}): "
          f"sui voti è dalla stessa parte della frattura principale. Ma resta un gruppo distinto, e la distanza "
          f"Sud–Cina/Russia è {d_gs_cr / d_cr_eu:.0%} di quella Cina/Russia–Europa. Vicini non vuol dire 'seguire': "
          f"lo verifica il livello 2.")
R.dati["distanze_blocchi"] = DIST
Bd = [r for r in B if meta.loc[r, "solo_usa"] == 0]
WB = W.loc[Bd, CL.index]; mB = meta.loc[Bd]
feat = {}
for k, n in c.TEMI.items():
    s = WB[mB[k] == 1]
    feat[f"Sì · {n}"] = (s == 1).where(s.notna()).mean()
    feat[f"Astenuto · {n}"] = (s == 0).where(s.notna()).mean()
for p, n in c.POTENZE.items():
    feat[f"Accordo con {n}"] = WB.eq(W.loc[Bd, p], axis=0).where(WB.notna()).mean()
feat["Assenza"] = WB.isna().mean()
F = pd.DataFrame(feat)
gruppi = list(GCOL)
rows = []
for col in F.columns:
    g = [F.loc[CL.gruppo == n, col].dropna().astype(float).values for n in gruppi]
    try:
        H, p = kruskal(*g)
    except ValueError:
        continue
    n = sum(len(x) for x in g)
    rows.append(dict(variabile=col, H=H, p=p, eps2=(H - len(g) + 1) / (n - len(g)), **{gg: np.median(x) for gg, x in zip(gruppi, g)}))
T = pd.DataFrame(rows); T["p_BH"] = multipletests(T.p, method="fdr_bh")[1]
T = T.sort_values("eps2", ascending=False); T.to_csv(OUT / "test_kruskal_wallis.csv", index=False)
R.voce("Votazioni usate per descrivere", f"{len(Bd)} (metà B, 'USA contro tutti' escluse)")
R.voce("Test", "Kruskal-Wallis, correzione Benjamini-Hochberg, effect size epsilon quadro")
R.voce("Variabili con p corretto < 0,05", f"{(T.p_BH < .05).sum()} su {len(T)}")
R.testo("")
R.testo(f"  {'variabile':<32}{'eps2':>6}  " + "  ".join(f"{g.split()[0][:9]:>9}" for g in gruppi))
for _, r in T.iterrows():
    R.testo(f"  {r.variabile:<32}{r.eps2:>6.2f}  " + "  ".join(f"{r[g] * 100:>8.0f}%" for g in gruppi))
R.testo("  (mediane per cluster; ordine colonne: Sud globale, Europa, Cina–Russia, USA–Israele)")

Tm = T[T.p_BH < .05].head(12); M = Tm.set_index("variabile")[gruppi]
Zs = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(9, 6.5))
cm = LinearSegmentedColormap.from_list("d", c.DIVERGENTE)
im = ax.imshow(Zs.values, cmap=cm, vmin=-1.6, vmax=1.6, aspect="auto")
for i in range(Zs.shape[0]):
    for j in range(Zs.shape[1]):
        ax.text(j, i, f"{M.values[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9.5,
                color="white" if abs(Zs.values[i, j]) > 1.05 else c.INCHIOSTRO)
nn_ = CL.gruppo.value_counts()
ax.set_xticks(range(4)); ax.set_xticklabels([f"Sud globale\n(n={nn_[gruppi[0]]})", f"Europa e alleati\n(n={nn_[gruppi[1]]})",
                                             f"Cina–Russia\n(n={nn_[gruppi[2]]})", f"USA–Israele\n(n={nn_[gruppi[3]]})"], fontsize=10)
ax.set_yticks(range(len(Zs))); ax.set_yticklabels([f"{v}  (ε²={e:.2f})" for v, e in zip(Tm.variabile, Tm.eps2)], fontsize=9.5)
ax.spines[:].set_visible(False); cb = fig.colorbar(im, ax=ax, shrink=.6); cb.set_label("z-score per riga")
ax.set_title("Profili dei cluster su votazioni NON usate per costruirli\n(mediane; Kruskal–Wallis, p corretti BH < 0,001)", fontsize=12)
R.figura(c.salva(fig, OUT, "heatmap_cluster"), "17 (la slide usa figure_slide/slide17_profili.py)")

# ------------------------------------------------------------------ 6. rete
R.sezione("Rete di accordo come conferma (slide 18)")
S = list(CL.index)
Kp, Np = c.kappa(W.loc[meta.solo_usa == 0, S])
Kp.to_csv(OUT / "kappa.csv")
R.voce("Nodi", f"{len(S)} Stati (gli stessi della PCA)")
R.voce("Peso degli archi", "kappa di Cohen (accordo corretto per la frequenza dei voti)")
R.voce("Votazioni", f"{int((meta.solo_usa == 0).sum())} ('USA contro tutti' escluse)")
rob = {}
for th in [.3, .4, .5]:
    G = c.rete(Kp, Np, th); part = louvain.best_partition(G, weight="weight", random_state=0)
    cs = pd.Series(part).value_counts()
    rob[th] = dict(archi=G.number_of_edges(), Q=louvain.modularity(part, G), com3=int((cs >= 3).sum()),
                   isolati=int(sum(1 for n in G if G.degree(n) == 0)))
    R.voce(f"Soglia kappa >= {th}", f"{rob[th]['archi']} archi, Q = {rob[th]['Q']:.2f}, "
           f"{rob[th]['com3']} comunità con >= 3 Stati, {rob[th]['isolati']} isolati")
    if th == .4:
        G04, part04 = G, part
com = pd.Series(part04)
def nome_com(m):
    m = set(m)
    return ("Occidente, USA e Israele" if "USA" in m else "Cina, Russia, India, Iran…" if "CHN" in m
            else "Sud globale" if "NGA" in m else "America Latina e Pacifico" if "MEX" in m else "Senza legami forti")
nomi_com = {g: nome_com(m.index) for g, m in com.groupby(com)}
CL["comunita"] = com.map(nomi_com)
for g, m in com.groupby(com):
    R.voce(f"Comunità: {nomi_com[g]}", f"{len(m)} Stati" + (f" ({', '.join(c.nome(i) if i in c.NOMI else P.loc[i, 'Countryname'].title() for i in m.index)})" if len(m) <= 13 else ""))
R.voce("Accordo comunità / cluster (ARI)", f"{adjusted_rand_score(CL.cluster, com[CL.index]):.2f}")
bc = nx.betweenness_centrality(G04)
R.voce("Ponti (betweenness più alta)", ", ".join(f"{c.nome(i) if i in c.NOMI else P.loc[i, 'Countryname'].title()} {v:.2f}"
                                                 for i, v in sorted(bc.items(), key=lambda x: -x[1])[:4]))
CL.to_csv(OUT / "clusters.csv")
nx.write_gexf(G04, OUT / "rete_kappa04.gexf")

righe = []
for a in anni:
    sub = W.loc[(meta.year == a) & (meta.solo_usa == 0), S]
    Ky, Ny = c.kappa(sub); G = c.rete(Ky, Ny, .4, min_comuni=15)
    part = louvain.best_partition(G, weight="weight", random_state=0)
    forza = pd.Series(dict(G.degree(weight="weight"))) / (len(S) - 1)   # forza normalizzata
    righe.append(dict(anno=a, Q=louvain.modularity(part, G), cina_russia=part["CHN"] == part["RUS"],
                      india_cina=part["IND"] == part["CHN"],
                      **{f"forza_{k}": forza.get(k, np.nan) for k in ["USA", "CHN", "RUS", "IND"]},
                      forza_membri=forza.reindex([s for s in S if P.loc[s, "nam"] == 1]).median(),
                      forza_altri=forza.reindex([s for s in S if P.loc[s, "nam"] == 0]).median()))
Y = pd.DataFrame(righe).set_index("anno")
Y.to_csv(OUT / "rete_per_anno.csv")
R.voce("Centralità (forza normalizzata), metodo", "somma dei κ dei propri archi / (Stati - 1), rete di ciascun anno")
for k, nm in [("USA", "USA"), ("CHN", "Cina"), ("RUS", "Russia"), ("IND", "India"), ("membri", "membro mediano"), ("altri", "non membro mediano")]:
    R.voce(f"   {nm}: 2021 -> 2023 (min-max)", f"{Y.loc[2021, 'forza_' + k]:.3f} -> {Y.loc[2023, 'forza_' + k]:.3f} "
           f"({Y['forza_' + k].min():.3f}-{Y['forza_' + k].max():.3f})")
R.voce("Modularità per anno: min-max", f"{Y.Q.min():.2f} - {Y.Q.max():.2f}")
R.voce("Cina e Russia nella stessa comunità", f"{int(Y.cina_russia.sum())} anni su {len(Y)}")
R.voce("India nella comunità della Cina", f"{int(Y.india_cina.sum())} anni su {len(Y)} "
       f"(no: {', '.join(str(a) for a in Y.index[~Y.india_cina])})")
R.lettura("La rete ritrova gli stessi blocchi dei cluster con un metodo diverso, e la divisione "
          "è stabile anno per anno.")
R.lettura("Centralità: Cina e Russia hanno pochi legami forti in tutto il periodo (stanno al bordo della rete, "
          "non al centro del Sud globale). Gli USA diventano più centrali dal 2019 al 2024 (legami più forti con "
          "l'Occidente) e crollano nel 2025, quando votano contro quasi tutti anche fuori dalle votazioni 'USA contro tutti'.")
R.dati["rete"] = dict(Q=rob[.4]["Q"], india_cina_anni=int(Y.india_cina.sum()))

# ---- matrice di adiacenza ordinata per comunità + rete ridotta tra comunità (appendice).
# Con 172 nodi e migliaia di archi il disegno nodi-archi diventa un gomitolo; la matrice riordinata per
# comunità mostra ogni coppia di Stati: i blocchi sulla diagonale sono le comunità, ciò che sta fuori
# dalla diagonale sono i ponti. A destra la stessa rete ridotta a 4 nodi (uno per comunità).
from matplotlib.patches import Rectangle, Circle
ORD_COM = ["Occidente, USA e Israele", "Cina, Russia, India, Iran…", "Sud globale", "America Latina e Pacifico"]
ordine_nodi = []
for g_ in ORD_COM:
    m_ = CL.index[CL.comunita == g_]
    ordine_nodi += list(PC.loc[m_, "PC1"].sort_values().index)
senza_ = [n for n in CL.index if CL.loc[n, "comunita"] not in ORD_COM]
ordine_nodi += senza_
Km = Kp.loc[ordine_nodi, ordine_nodi].astype(float).clip(lower=0).to_numpy(copy=True)
np.fill_diagonal(Km, np.nan)
fig = plt.figure(figsize=(15, 7.2))
gsp = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=.12)
ax = fig.add_subplot(gsp[0]); ax2 = fig.add_subplot(gsp[1])
cm_k = LinearSegmentedColormap.from_list("k", [c.SCHEDA, "#CFC8BA", "#7F7A70", c.INCHIOSTRO])
cm_k.set_bad(c.SCHEDA)
im = ax.imshow(Km, cmap=cm_k, vmin=0, vmax=.8, interpolation="nearest")
n_ = len(ordine_nodi); x0 = 0
for g_ in ORD_COM:
    k_ = int((CL.comunita == g_).sum())
    ax.add_patch(Rectangle((x0 - .5, x0 - .5), k_, k_, fill=False, ec=c.CCOL[g_], lw=2.6))
    ax.add_patch(Rectangle((-6.5, x0 - .5), 4, k_, color=c.CCOL[g_], clip_on=False))
    ax.add_patch(Rectangle((x0 - .5, -6.5), k_, 4, color=c.CCOL[g_], clip_on=False))
    ax.text(-8.5, x0 + k_ / 2 - .5, f"{g_}\n({k_} Stati)", ha="right", va="center", fontsize=10.5,
            color=c.INCHIOSTRO, linespacing=1.15)
    x0 += k_
if senza_:
    ax.text(-8.5, x0 + len(senza_) / 2, "senza legami forti (" + ", ".join(P.loc[i, "Countryname"].title() for i in senza_) + ")",
            ha="right", va="center", fontsize=9, color=c.TENUE)
ax.set_xlim(-7, n_ - .5); ax.set_ylim(n_ - .5, -7)
ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
ax.set_title("Ogni Stato con ogni Stato (172 × 172)", fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=.55, pad=.03, ticks=[0, .2, .4, .6, .8])
cb.ax.set_yticklabels(["0", "0,2", "0,4 soglia", "0,6", "0,8"]); cb.outline.set_visible(False)
cb.ax.axhline(.4, color=c.INCHIOSTRO, lw=2)
cb.ax.set_title("κ di Cohen", fontsize=10, loc="left", pad=8)
ax.text(n_ / 2, n_ + 4, "Righe e colonne: gli Stati, ordinati per comunità (Louvain) e, dentro ogni comunità, per PC1.\n"
        "Blocchi scuri sulla diagonale = comunità; macchie fuori dalla diagonale = legami tra comunità.",
        ha="center", va="top", fontsize=9, color=c.TENUE)
# rete ridotta: un nodo per comunità, archi = legami forti (κ >= 0,4) tra Stati di comunità diverse
cnt_ = pd.DataFrame(0, index=ORD_COM, columns=ORD_COM)
for u_, v_ in G04.edges():
    gu, gv = CL.loc[u_, "comunita"], CL.loc[v_, "comunita"]
    if gu in ORD_COM and gv in ORD_COM:
        cnt_.loc[gu, gv] += 1
        if gu != gv:
            cnt_.loc[gv, gu] += 1
posc = {"Occidente, USA e Israele": (-1.0, 0.0), "Cina, Russia, India, Iran…": (0.25, 0.78),
        "Sud globale": (1.0, -0.05), "America Latina e Pacifico": (0.15, -0.85)}
nn_c = {g_: int((CL.comunita == g_).sum()) for g_ in ORD_COM}
rad = {g_: .13 + .2 * np.sqrt(nn_c[g_] / max(nn_c.values())) for g_ in ORD_COM}
from itertools import combinations as _comb
tra_ = {(a_, b_): int(cnt_.loc[a_, b_]) for a_, b_ in _comb(ORD_COM, 2)}
mx_ = max(tra_.values())
for (a_, b_), v_ in tra_.items():
    (xa, ya), (xb, yb) = posc[a_], posc[b_]
    if v_ == 0 and not (a_ == "Occidente, USA e Israele" and b_ == "Sud globale"):
        continue                                  # si disegna solo lo zero che conta: Occidente–Sud globale
    if v_ == 0:
        ax2.plot([xa, xb], [ya, yb], color=c.GRIGIO, lw=1.2, ls=(0, (2, 3)), zorder=1)
    else:
        ax2.plot([xa, xb], [ya, yb], color="#7F7A70", lw=1.2 + 16 * v_ / mx_, alpha=.75, solid_capstyle="butt", zorder=1)
    lab_ = f"{v_}"
    ax2.text((xa + xb) / 2, (ya + yb) / 2, lab_, ha="center", va="center", fontsize=11, fontweight="bold",
             color=c.INCHIOSTRO if v_ else c.GRIGIO, bbox=dict(boxstyle="round,pad=.25", fc=c.SFONDO, ec="none"), zorder=4)
for g_ in ORD_COM:
    x_, y_ = posc[g_]
    ax2.add_patch(Circle((x_, y_), rad[g_], color=c.CCOL[g_], zorder=3))
    ax2.text(x_, y_, str(nn_c[g_]), ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=5)
    dy_ = rad[g_] + .07
    sopra = g_ == "Cina, Russia, India, Iran…"
    ax2.text(x_, y_ + dy_ if sopra else y_ - dy_, f"{g_}\nlegami interni: " + f"{int(cnt_.loc[g_, g_]):,}".replace(",", "."),
             ha="center", va="bottom" if sopra else "top", fontsize=10, color=c.INCHIOSTRO, zorder=5, linespacing=1.15)
ax2.set_xlim(-1.6, 1.6); ax2.set_ylim(-1.62, 1.3); ax2.set_aspect("equal"); ax2.axis("off")
ax2.set_title(f"La stessa rete, un nodo per comunità (Q = {rob[.4]['Q']:.2f})", fontsize=12)
ax2.text(0, -1.66, "Numero nel nodo = Stati · numero sull'arco = coppie di Stati\ncon κ ≥ 0,4 tra le due comunità (Occidente–Sud globale: nessuna)",
         ha="center", va="top", fontsize=9, color=c.TENUE)
R.voce("Legami forti tra comunità (k >= 0,4)", ", ".join(f"{a_[:12]}–{b_[:12]} {v_}" for (a_, b_), v_ in tra_.items()))
R.voce("Legami forti dentro le comunità", ", ".join(f"{g_[:12]} {int(cnt_.loc[g_, g_])}" for g_ in ORD_COM))
fig.suptitle(f"Rete di accordo 2014–2025: quattro comunità e pochi ponti", x=.01, ha="left", fontweight="bold", fontsize=13)
R.figura(c.salva(fig, OUT, "appendice_rete_matrice"), "appendice")

pos = nx.spring_layout(G04, weight="weight", seed=3, k=.25, iterations=200)
fig, ax = plt.subplots(figsize=(11, 9))
nx.draw_networkx_edges(G04, pos, ax=ax, edge_color=c.LINEA, width=.3, alpha=.6)
ccol = c.CCOL
for g, col in ccol.items():
    nodi = [n for n in G04 if CL.loc[n, "comunita"] == g]
    nx.draw_networkx_nodes(G04, pos, nodelist=nodi, node_color=col, node_size=55, edgecolors=c.SFONDO,
                           linewidths=.8, ax=ax, label=f"{g} ({len(nodi)})")
altri = [n for n in G04 if CL.loc[n, "comunita"] not in ccol]
if altri:
    nx.draw_networkx_nodes(G04, pos, nodelist=altri, node_color=c.SCHEDA, edgecolors=c.GRIGIO, node_size=55, ax=ax,
                           label="Senza legami forti: " + ", ".join(P.loc[i, "Countryname"].title() for i in altri))
for n in ["USA", "ISR", "CHN", "RUS", "IND", "BRA", "GBR", "IRN", "PAK", "MEX", "ZAF", "BLR", "MLI", "JPN"]:
    x, y = pos[n]; ax.text(x, y + .025, c.nome(n), fontsize=8.5, ha="center")
ax.legend(frameon=False, loc="lower left", fontsize=9.5); ax.axis("off")
ax.set_title(f"Rete di accordo 2014–2025: arco se κ di Cohen ≥ 0,4, comunità con Louvain (Q = {rob[.4]['Q']:.2f})")
R.figura(c.salva(fig, OUT, "appendice_rete_grafo"), "appendice")

fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axs[0].plot(Y.index, Y.Q, marker="o", color=c.INCHIOSTRO, lw=2); axs[0].set_ylim(.4, .6); axs[0].set_ylabel("Modularità Q")
axs[0].set_title("La divisione in blocchi è stabile anno per anno; l'India sta quasi sempre con la Cina")
axs[1].scatter(Y.index, [0] * len(Y), s=140, marker="s", color=[c.COL_CINA if v else c.GRIGIO_CHIARO for v in Y.india_cina])
axs[1].set_yticks([0]); axs[1].set_yticklabels(["India con\nla Cina"]); axs[1].set_ylim(-1, 1)
axs[1].spines["left"].set_visible(False); axs[1].set_xticks(anni)
axs[1].text(2014, -.85, "ocra = stessa comunità Louvain (soglia κ 0,4); grigio = comunità diversa", fontsize=8.5, color=c.TENUE)
R.figura(c.salva(fig, OUT, "appendice_rete_nel_tempo"), "appendice")

fig, ax = plt.subplots(figsize=(9, 4.8)); fine = []
for k, nm, col, lw in [("membri", "Membro mediano", c.ACCENTO, 2.4), ("altri", "Non membro mediano", c.COL_ALTRI, 2.4),
                       ("USA", "USA", c.COL_POTENZE["USA"], 1.6), ("CHN", "Cina", c.COL_POTENZE["CHN"], 1.6),
                       ("RUS", "Russia", c.COL_POTENZE["RUS"], 1.6), ("IND", "India", c.TENUE, 1.6)]:
    ax.plot(Y.index, Y["forza_" + k], color=col, lw=lw, ls="-" if lw > 2 else "--", marker="o", ms=3.5)
    fine.append([Y["forza_" + k].iloc[-1], nm, col])
fine.sort(key=lambda t: t[0])
for i_ in range(1, len(fine)):                       # etichette finali senza sovrapposizioni
    fine[i_][0] = max(fine[i_][0], fine[i_ - 1][0] + .011)
for yv, nm, col in fine:
    ax.text(Y.index[-1] + .2, yv, nm, color=col, va="center", fontsize=9)
ax.axvline(2022, color=c.GRIGIO, lw=.8, ls=":")
ax.set_xlim(2014, 2026.6); ax.set_xticks(anni); ax.set_ylabel("Centralità (forza normalizzata)")
c.griglia(ax, "y")
ax.set_title("Centralità nella rete di accordo, anno per anno")
R.figura(c.salva(fig, OUT, "appendice_centralita_nel_tempo"), "appendice")

R.dati["cluster"] = {g: int(n) for g, n in CL.gruppo.value_counts().items()}
R.dati["cramer_nam"] = float(cramer(pd.crosstab(CL.gruppo, CL.nam))[2])
R.salva(OUT, "report_atto3_livello1_mondo.txt")
