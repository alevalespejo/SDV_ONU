"""Parte 3 - Non Allineati (slide 20-27). Q2-Q5.

Q2 effetto maggioranza: accordo con la Cina contro accordo con la maggioranza.
Q3 seguire una potenza quando è in minoranza, contro l'accordo atteso per caso.
Q4 prima e dopo il 2022; astensione per tema e sull'Ucraina.
Q5 random forest sui membri: addestramento 2015-22, verifica 2023-25.
Output: output/atto3_livello2_non_allineati/ (report, figure, segue_in_minoranza.csv)
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, binomtest, wilcoxon, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import comune as c

OUT = c.cartella_atto("atto3_livello2_non_allineati")
plt = c.stile()
df, P, W, meta = c.dati()
R = c.Report("REPORT ATTO 3 - LIVELLO 2 (NON ALLINEATI): SEGUONO UNA POTENZA?")
NAM = [i for i in W.columns if i in c.NAM]
ALTRI = [i for i in W.columns if i not in c.NAM]

# ---------------------------------------------------------- 1. prima / dopo
R.sezione("Prima/dopo: l'accordo con la Cina è accordo con la maggioranza (slide 21)")
acc_cina = c.accordo(W, "CHN")
acc_maj = c.accordo_maggioranza(W, meta).drop("CHN")
r, p = pearsonr(acc_cina, acc_maj[acc_cina.index])
rho = spearmanr(acc_cina, acc_maj[acc_cina.index])[0]
for pot in c.POTENZE:
    R.voce(f"{c.nome(pot)} vota con la maggioranza", c.pct(c.accordo_maggioranza(W, meta)[pot]))
R.voce("Correlazione accordo-Cina / accordo-maggioranza", f"r = {r:.2f} (n = {len(acc_cina)} Stati)")
R.voce("Mediana accordo con la Cina: membri / altri", f"{c.pct(acc_cina[NAM].median())} / {c.pct(acc_cina[[a for a in ALTRI if a != 'CHN']].median())}")
R.lettura("Chi vota con la maggioranza vota per forza spesso come la Cina, perché la Cina vota "
          "quasi sempre con la maggioranza. L'accordo grezzo non distingue 'seguire la Cina' da "
          "'votare con la maggioranza': serve guardare i casi in cui le due cose divergono.")
R.dati["prima_dopo"] = dict(r=r, cina_con_maggioranza=float(c.accordo_maggioranza(W, meta)["CHN"]))

fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axs[0]
m_ = [i in c.NAM for i in acc_cina.index]
ax.hist([acc_cina[m_] * 100, acc_cina[[not x for x in m_]] * 100], bins=np.arange(15, 101, 2.5), stacked=True,
        color=[c.COL_MEMBRI, c.COL_ALTRI], edgecolor=c.SFONDO, linewidth=.6)
ax.axvline(acc_cina.median() * 100, color=c.SECONDA, lw=2)
ax.text(acc_cina.median() * 100 - 1.5, ax.get_ylim()[1] * .95, f"mediana {acc_cina.median() * 100:.0f}%", ha="right", fontsize=10, color=c.SECONDA)
ax.text(24, ax.get_ylim()[1] * .55, c.T("membri del\nMovimento", "Movement\nmembers"), fontsize=11, color=c.ACCENTO_TESTO, fontweight="bold")
ax.text(24, ax.get_ylim()[1] * .33, c.T("altri paesi", "other countries"), fontsize=11, color=c.GRIGIO_MEDIO, fontweight="bold")
ax.set_xlabel("% di voti uguali a quelli della Cina"); ax.set_ylabel("Numero di paesi")
ax.set_title("PRIMA: tutti sembrano vicini alla Cina", fontsize=12)
ax = axs[1]
cols = [c.COL_MEMBRI if i in c.NAM else c.COL_ALTRI for i in acc_cina.index]
ax.scatter(acc_maj[acc_cina.index] * 100, acc_cina * 100, s=22, c=cols, edgecolor=c.SFONDO, linewidth=.6)
ax.plot([0, 100], [0, 100], color=c.GRIGIO, ls=":", lw=1)
for i in ["USA", "ISR", "RUS"]:
    ax.annotate(c.nome(i), (acc_maj[i] * 100, acc_cina[i] * 100), xytext=(5, 3), textcoords="offset points", fontsize=9.5)
for i, (xl, yl) in zip(["PRK", "IND", "MLI", "ZAF"], [(52, 92), (52, 84), (60, 30), (84, 30)]):
    ax.annotate(c.nome(i), (acc_maj[i] * 100, acc_cina[i] * 100), xytext=(xl, yl), fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=c.TENUE, lw=.6, shrinkB=2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xlabel("% di voti uguali alla maggioranza"); ax.set_ylabel("% di voti uguali alla Cina")
ax.text(98, 4, f"r = {r:.2f} (Spearman {rho:.2f})", ha="right", fontsize=11)
ax.set_title("DOPO: è quasi solo l'accordo con la maggioranza", fontsize=12)
R.figura(c.salva(fig, OUT, "effetto_maggioranza"), 21)

# ------------------------------------------------- 2. seguire in minoranza
R.sezione("Quando una potenza è in minoranza, chi la segue? (slide 22-24)")
# il seguire si misura senza le votazioni "USA contro tutti", per tutte le potenze
W_TUTTE, META_TUTTE = W, meta
W, meta = W[meta.solo_usa == 0], meta[meta.solo_usa == 0]
R.voce("Definizione", "votazioni in cui la potenza vota diversamente dalla maggioranza; quota in cui")
R.testo("                                            lo Stato vota come lei (stesso sì, no o astensione)")
periodi = {"2014-2021": range(2014, 2022), "2022-2024": range(2022, 2025), "2025": [2025], "Tutto": range(2014, 2026)}
F = {}
for pot, nm in c.POTENZE.items():
    s, n = c.segue_in_minoranza(W, meta, pot); F[pot] = s
    R.testo("")
    R.voce(f"{nm} in minoranza", f"{n} votazioni")
    R.voce(f"   membro mediano la segue", c.pct(s[NAM].median()))
    R.voce(f"   non membro mediano la segue", c.pct(s.reindex(ALTRI).drop(pot, errors="ignore").median()))
    for per, yy in periodi.items():
        if per == "Tutto":
            continue
        sp, npp = c.segue_in_minoranza(W[meta.year.isin(yy)], meta[meta.year.isin(yy)], pot)
        R.voce(f"   membro mediano, {per}", f"{c.pct(sp[NAM].median())} (n = {npp})")
    top = s[NAM].sort_values(ascending=False).head(6)
    R.voce(f"   membri che la seguono di più", ", ".join(f"{c.nome(i) if i in c.NOMI else P.loc[i, 'Countryname'].title()} {v:.0%}" for i, v in top.items()))
F = pd.DataFrame(F); F.to_csv(OUT / "segue_in_minoranza.csv")
MED_ALTRI = {p: float(F[p].reindex(ALTRI).drop(p, errors="ignore").median()) for p in c.POTENZE}
R.lettura(f"Il membro mediano segue la Cina nel {c.pct(F.CHN[NAM].median(), 0)} dei casi, la Russia nel "
          f"{c.pct(F.RUS[NAM].median(), 0)}, gli USA nel {c.pct(F.USA[NAM].median(), 0)}. Le eccezioni sono pochi "
          f"alleati stretti (Corea del Nord, Siria, Iran, Pakistan, India per la Cina). Per confronto, il non "
          f"membro mediano segue gli USA nel {c.pct(MED_ALTRI['USA'], 0)} dei casi.")
R.lettura(f"Membri contro non membri: quando la Cina resta in minoranza, i non membri la seguono "
          f"({c.pct(MED_ALTRI['CHN'], 0)}) almeno quanto i membri ({c.pct(F.CHN[NAM].median(), 0)}). "
          f"I non allineati non sono più filocinesi degli altri.")
R.dati["segue"] = {p: float(F[p][NAM].median()) for p in c.POTENZE}
R.dati["segue_non_membri"] = MED_ALTRI
R.dati["segue_india"] = {p: float(F.loc["IND", p]) for p in c.POTENZE}

# ---- test: (1) termine di paragone 'per caso', (2) membri contro non membri, (3) intervalli bootstrap
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / (np.arange(len(p)) + 1)
    a = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(a); out[o] = np.minimum(a, 1); return out
def segue_vs_caso(pot):
    idx = W[pot].notna() & (W[pot] != meta.maj); sub = W.loc[idx]; pv = sub[pot]; righe = {}
    for s in sub.columns:
        if s == pot:
            continue
        ok = sub[s].notna(); n = int(ok.sum())
        if n < 20:
            continue
        a, b = sub.loc[ok, s], pv[ok]
        oss = float((a == b).mean()); att = float(sum((a == k).mean() * (b == k).mean() for k in (1, 0, -1)))
        righe[s] = dict(n=n, oss=oss, att=att, kappa=(oss - att) / (1 - att) if att < 1 else np.nan,
                        p=binomtest(int((a == b).sum()), n, att, alternative="greater").pvalue)
    T = pd.DataFrame(righe).T; T["p_bh"] = bh(T.p.values); return T
R.testo("")
R.voce("TEST 1 · termine di paragone 'per caso'", "accordo atteso se il paese votasse indipendentemente dalla potenza,")
R.testo("                                            con le proprie frequenze di sì/astenuto/no su quelle stesse votazioni;")
R.testo("                                            test binomiale unilaterale per paese, correzione Benjamini-Hochberg")
TEST_CASO = {}
for pot, nm in c.POTENZE.items():
    T = segue_vs_caso(pot); TEST_CASO[pot] = T
    Tm, To = T.reindex(NAM).dropna(subset=["oss"]), T.reindex(ALTRI).dropna(subset=["oss"])
    sig_m = Tm[(Tm.p_bh < .05)].sort_values("kappa", ascending=False)
    R.voce(f"{nm}: membro mediano osservato / atteso per caso", f"{c.pct(Tm.oss.median())} / {c.pct(Tm.att.median())} "
           f"(kappa mediano {Tm.kappa.median():.2f})")
    R.voce(f"   membri che la seguono più del caso (p BH < 0,05)", f"{len(sig_m)} su {len(Tm)}" +
           (": " + ", ".join(c.nome(i) if i in c.NOMI else P.loc[i, "Countryname"].title() for i in sig_m.index[:12]) if len(sig_m) else ""))
    R.voce(f"   non membri che la seguono più del caso", f"{int((To.p_bh < .05).sum())} su {len(To)}")
pd.concat(TEST_CASO, names=["potenza", "iso3"]).to_csv(OUT / "segue_vs_caso.csv")
R.lettura("Il membro tipico segue le potenze in minoranza perfino MENO di quanto farebbe votando a caso con le "
          "proprie abitudini (Cina: 9% osservato contro 15% atteso; kappa vicino a zero o negativo). Solo un gruppo "
          "ristretto di alleati stretti segue la Cina o la Russia più del caso; tra i non membri, 45 su 73 seguono gli USA.")
R.testo("")
R.voce("TEST 2 · membri contro non membri", "Mann-Whitney bilaterale; effetto = correlazione rank-biserial")
R.voce("TEST 3 · intervallo al 95%", "bootstrap sui paesi (5.000 ricampionamenti) della mediana dei membri")
rng_b = np.random.default_rng(0); TEST_MW = {}
for pot, nm in c.POTENZE.items():
    xm = F[pot].reindex(NAM).dropna().values; xo = F[pot].reindex(ALTRI).drop(pot, errors="ignore").dropna().values
    U_, p_ = mannwhitneyu(xm, xo, alternative="two-sided"); rb = 2 * U_ / (len(xm) * len(xo)) - 1
    bs = np.median(rng_b.choice(xm, (5000, len(xm)), replace=True), axis=1)
    TEST_MW[pot] = dict(med_membri=float(np.median(xm)), ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
                        med_altri=float(np.median(xo)), p=float(p_), rb=float(rb))
    R.voce(f"{nm}: membri {c.pct(np.median(xm), 0)} [IC {c.pct(TEST_MW[pot]['ci'][0], 0)}-{c.pct(TEST_MW[pot]['ci'][1], 0)}] vs non membri",
           f"{c.pct(np.median(xo), 0)}; p = {p_:.1e}; rank-biserial {rb:+.2f}")
R.lettura("Con gli USA e la Russia la differenza è netta: i non membri le seguono molto di più. Con la Cina "
          "membri e non membri si comportano in modo simile: i non allineati non sono più filocinesi degli altri.")
R.dati["test_segue"] = dict(mann_whitney=TEST_MW, caso={p: dict(oss=float(T.reindex(NAM).oss.median()),
                            att=float(T.reindex(NAM).att.median()), kappa=float(T.reindex(NAM).kappa.median()),
                            sig_membri=int((T.reindex(NAM).p_bh < .05).sum())) for p, T in TEST_CASO.items()})

mostra = ["IND", "PAK", "EGY", "SAU", "IDN", "DZA", "NGA", "ZAF", "MLI"]
fig, ax = plt.subplots(figsize=(9, 5.3))
righe_plot = ["mediana", "mediana_altri"] + mostra; yy = np.arange(len(righe_plot))[::-1]
val = pd.concat([pd.DataFrame(F.loc[NAM].median()).T.rename(index={0: "mediana"}),
                 pd.DataFrame([MED_ALTRI], index=["mediana_altri"]), F.loc[mostra]])
for y, s in zip(yy, righe_plot):
    ax.plot([val.loc[s].min() * 100, val.loc[s].max() * 100], [y, y], color=c.GRIGIO_CHIARO, lw=2, zorder=1)
att_ = pd.concat([pd.DataFrame({p_: [TEST_CASO[p_].reindex(NAM).att.median()] for p_ in c.POTENZE}, index=["mediana"]),
                  pd.DataFrame({p_: [TEST_CASO[p_].reindex(ALTRI).drop(p_, errors="ignore").att.median()] for p_ in c.POTENZE}, index=["mediana_altri"]),
                  pd.DataFrame({p_: TEST_CASO[p_].reindex(mostra).att for p_ in c.POTENZE})])
for pot in ["USA", "RUS", "CHN"]:
    ax.scatter(att_.loc[righe_plot, pot].astype(float) * 100, yy, s=70, facecolor="none", edgecolor=c.COL_POTENZE[pot],
               linewidth=1.4, zorder=2)
for pot in ["USA", "RUS", "CHN"]:
    _, n = c.segue_in_minoranza(W, meta, pot)
    ax.scatter(val.loc[righe_plot, pot] * 100, yy, s=70, color=c.COL_POTENZE[pot], zorder=3, edgecolor=c.SFONDO,
               linewidth=1.5, label=f"{c.nome(pot)} in minoranza (n={n})")
ax.set_yticks(yy); ax.set_yticklabels(["Membro mediano", "Non membro mediano"] + [c.nome(i) for i in mostra])
ax.get_yticklabels()[0].set_fontweight("bold"); ax.get_yticklabels()[1].set_fontweight("bold")
ax.axhline(yy[1] - .5, color=c.LINEA, lw=1)
ax.set_xlim(0, 60); ax.set_xlabel("% delle votazioni in cui la potenza è in minoranza e il paese vota come lei")
ax.scatter([], [], s=70, facecolor="none", edgecolor=c.TENUE, linewidth=1.4, label="cerchio vuoto: atteso per caso")
ax.legend(frameon=False, loc="lower right"); c.griglia(ax, "x")
ax.set_title("Quando una grande potenza resta in minoranza, quasi nessun membro la segue", fontsize=12)
R.figura(c.salva(fig, OUT, "appendice_seguire_paesi"), "appendice")

W, meta = W_TUTTE, META_TUTTE
# ------------------------------------------- 3. prima e dopo il 2022 (tutte le potenze)
R.sezione("Prima e dopo il 2022: il 'seguire' cambia? (slide 25)")
R.voce("Misura", "la stessa della slide 22 (segue la potenza quando è in minoranza), 2014-21 contro 2022-25")
R.voce("Votazioni 'USA contro tutti'", "escluse, per confrontare i due periodi alla pari")
R.voce("Test", "Wilcoxon a ranghi con segno, appaiato per paese (prima vs dopo)")
Wn_ = W[meta.solo_usa == 0]; mn_ = meta[meta.solo_usa == 0]
MINV = 10
G22 = {}
for pot, nm in c.POTENZE.items():
    cols = {}
    for per, msk in [("prima", mn_.year <= 2021), ("dopo", mn_.year >= 2022)]:
        Wx, mx = Wn_[msk.values], mn_[msk.values]
        idx = Wx[pot].notna() & (Wx[pot] != mx.maj); sub = Wx.loc[idx]
        ok = sub.notna()
        cols[per] = sub.eq(sub[pot], axis=0).where(ok).astype(float).mean().drop(pot)
        cols["n_" + per] = ok.sum().drop(pot)
        R.voce(f"{nm} in minoranza, {per} del 2022", f"{int(idx.sum())} votazioni")
    G = pd.DataFrame(cols).astype(float); G = G[(G.n_prima >= MINV) & (G.n_dopo >= MINV)]; G["d"] = G.dopo - G.prima
    G22[pot] = G
    Gm, Go = G.reindex(NAM).dropna(), G.reindex(ALTRI).dropna()
    w_ = wilcoxon(Gm.prima, Gm.dopo); wo_ = wilcoxon(Go.prima, Go.dopo)
    R.voce(f"   membri ({len(Gm)}): mediana prima -> dopo", f"{c.pct(Gm.prima.median())} -> {c.pct(Gm.dopo.median())}; Wilcoxon p = {w_.pvalue:.1e}")
    R.voce(f"   non membri ({len(Go)}): mediana prima -> dopo", f"{c.pct(Go.prima.median())} -> {c.pct(Go.dopo.median())}; Wilcoxon p = {wo_.pvalue:.1e}")
    R.voce(f"   membri che si avvicinano di più", ", ".join(f"{c.nome(i) if i in c.NOMI else P.loc[i, 'Countryname'].title()} "
           f"{Gm.loc[i, 'prima'] * 100:.0f}->{Gm.loc[i, 'dopo'] * 100:.0f}%" for i in Gm.d.nlargest(5).index))
    R.dati.setdefault("prima_dopo_2022", {})[pot] = dict(membri_prima=float(Gm.prima.median()), membri_dopo=float(Gm.dopo.median()),
        altri_prima=float(Go.prima.median()), altri_dopo=float(Go.dopo.median()), p_membri=float(w_.pvalue), p_altri=float(wo_.pvalue))
pd.concat(G22, names=["potenza", "iso3"]).to_csv(OUT / "prima_dopo_2022.csv")
R.lettura("Dopo il 2022 i membri cambiano poco: restano lontani da USA e Russia e si avvicinano appena alla Cina "
          "(circa 2 punti). A muoversi sono i non membri, che si allontanano da Russia e Cina e si avvicinano agli USA. "
          "Tra i membri si spostano pochi singoli paesi: verso Mosca soprattutto Mali, Niger, Nicaragua.")

fig, axs = plt.subplots(1, 3, figsize=(13, 5.6), sharey=True)
for ax, pot in zip(axs, ["USA", "CHN", "RUS"]):
    G = G22[pot]; Gm = G.reindex(NAM).dropna(); Go = G.reindex(ALTRI).dropna()
    for i, rr in Gm.iterrows():
        ax.plot([0, 1], [rr.prima * 100, rr.dopo * 100], color=c.GRIGIO_CHIARO, lw=.7, zorder=1)
    top = list(Gm.d.nlargest(2).index)
    for i in top:
        rr = Gm.loc[i]
        ax.plot([0, 1], [rr.prima * 100, rr.dopo * 100], color=c.COL_POTENZE[pot], lw=1.6, zorder=2)
        ax.text(1.04, rr.dopo * 100, c.nome(i) if i in c.NOMI else P.loc[i, "Countryname"].title(), fontsize=9, va="center")
    ax.plot([0, 1], [Gm.prima.median() * 100, Gm.dopo.median() * 100], color=c.COL_POTENZE[pot], lw=3.5, zorder=3,
            marker="o", ms=7)
    ax.plot([0, 1], [Go.prima.median() * 100, Go.dopo.median() * 100], color=c.GRIGIO, lw=2.2, ls="--", zorder=3)
    ax.text(-.06, Gm.prima.median() * 100, f"{Gm.prima.median() * 100:.0f}%", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(1.04, Gm.dopo.median() * 100, f"{Gm.dopo.median() * 100:.0f}%", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["2014–21", "2022–25"]); ax.set_xlim(-.35, 1.55)
    ax.set_title(f"{c.nome(pot)} in minoranza", fontsize=12); c.griglia(ax, "y")
axs[0].set_ylabel("% delle votazioni in cui il paese vota come la potenza")
axs[0].set_ylim(0, 85)
from matplotlib.lines import Line2D as _L2
fig.legend([_L2([], [], color=c.INCHIOSTRO, lw=3.5), _L2([], [], color=c.GRIGIO, lw=2.2, ls="--"), _L2([], [], color=c.GRIGIO_CHIARO, lw=1)],
           ["Membro mediano", "Non membro mediano", "Ogni membro"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.02))
fig.subplots_adjust(bottom=.16)
fig.suptitle("Dopo il 2022 i membri cambiano poco: a muoversi sono gli altri", x=.01, ha="left", fontweight="bold", fontsize=13)
R.figura(c.salva(fig, OUT, "prima_dopo_2022"), "25 (la slide riporta le mediane come barre)")

# ---------------------------------------------------- 4. astensione
R.sezione("Astensione: per tema e sull'Ucraina (slide 26)")
Wn = W[meta.solo_usa == 0]; mn = meta[meta.solo_usa == 0]
rows = []
for k, n in list(c.TEMI.items()) + [("tutti", "Tutte le votazioni")]:
    sub = Wn if k == "tutti" else Wn[mn[k] == 1]
    ab = (sub == 0).where(sub.notna()).mean()
    rows.append(dict(tema=n, n=len(sub), membri=ab[NAM].median(), altri=ab[ALTRI].median()))
A = pd.DataFrame(rows).set_index("tema")
for t, rr in A.iterrows():
    R.voce(f"{t} (n = {int(rr.n)})", f"astensione mediana membri {c.pct(rr.membri)} | altri {c.pct(rr.altri)}")
uk = meta.index[meta.session.astype(str) == "11emsp"]
R.testo("")
R.voce("Votazioni sull'Ucraina", f"{len(uk)} (11a sessione speciale d'emergenza)")
lab = {1: "Sì", 2: "Astenuto", 3: "No", 8: "Assente"}
u = df[df.rcid.isin(uk)].copy(); u["membro"] = u.iso3.isin(c.NAM).astype(int)
U = u.groupby(["rcid", "membro"]).vote.value_counts(normalize=True).unstack().fillna(0).rename(columns=lab)
for m, nm in [(1, "membri"), (0, "altri")]:
    d = U.xs(m, level="membro")
    R.voce(f"Ucraina, {nm}: sì / astenuti / no", f"{c.pct(d['Sì'].min(), 0)}-{c.pct(d['Sì'].max(), 0)} / "
           f"{c.pct(d['Astenuto'].min(), 0)}-{c.pct(d['Astenuto'].max(), 0)} / {c.pct(d['No'].min(), 0)}-{c.pct(d['No'].max(), 0)}")
R.lettura("In generale si astengono di più i non membri (l'Occidente sui temi del Sud). Sull'Ucraina "
          "si rovescia: i membri non votano con Mosca, si astengono.")
R.dati["astensione"] = A.to_dict()

order = ["Tutte le votazioni"] + list(c.TEMI.values()); y = np.arange(len(order))[::-1]
fig, ax = plt.subplots(figsize=(8, 4.6))
for yi, t in zip(y, order):
    ax.plot([A.loc[t, "membri"] * 100, A.loc[t, "altri"] * 100], [yi, yi], color=c.GRIGIO_CHIARO, lw=2)
ax.scatter(A.loc[order, "membri"] * 100, y, s=70, color=c.COL_MEMBRI, label="Membri del Movimento (mediana)", zorder=3)
ax.scatter(A.loc[order, "altri"] * 100, y, s=70, color=c.COL_ALTRI, label="Altri paesi (mediana)", zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f"{t} (n={int(A.loc[t, 'n'])})" for t in order]); ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlabel("% di astensioni (votazioni 'USA contro tutti' escluse)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2); c.griglia(ax, "x")
ax.set_title("Di solito si astengono gli altri: l'Occidente sui temi del Sud")
R.figura(c.salva(fig, OUT, "astensione_per_tema"), "appendice")

et = {10791: "Aggressione (mar 2022)", 10792: "Conseguenze umanitarie (mar 2022)", 10793: "Sospensione dal Cons. diritti umani",
      10797: "Integrità territoriale (ott 2022)", 10800: "Riparazioni (nov 2022)", 10881: "Pace giusta (feb 2023)",
      11063: "Pace in Ucraina (feb 2025)", 11064: '"Path to peace" (testo USA, feb 2025)', 11144: "Bambini ucraini (dic 2025)"}
ordine = [r for r in uk]
colv = c.COL_VOTI
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
for ax, m, t in zip(axs, [1, 0], [f"Membri del Movimento ({len(NAM)})", f"Altri paesi ({len(ALTRI)})"]):
    d = U.xs(m, level="membro").reindex(ordine); left = np.zeros(len(d)); yy2 = np.arange(len(d))[::-1]
    for k in colv:
        ax.barh(yy2, d[k] * 100, left=left, color=colv[k], edgecolor=c.SFONDO, linewidth=2, height=.7, label=k)
        for yi, l, v in zip(yy2, left, d[k] * 100):
            if v >= 12:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white" if k in ["Sì", "No"] else c.INCHIOSTRO)
        left += d[k].values * 100
    ax.set_title(t, fontsize=12); ax.set_xlim(0, 100); ax.set_xlabel("%")
axs[0].set_yticks(np.arange(len(ordine))[::-1]); axs[0].set_yticklabels([et.get(r, str(r)) for r in ordine], fontsize=9.5)
axs[1].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12))
fig.suptitle("Sull'Ucraina i membri del Movimento non seguono Mosca: si astengono", x=.01, ha="left", fontweight="bold", fontsize=13)
R.figura(c.salva(fig, OUT, "ucraina_astensione"), "appendice")

# ---- astensione come strategia: test e figura unica
R.testo("")
R.voce("TEST · Ucraina, membri vs altri", "chi-quadro su sì/astenuto/no, voti espressi (assenti esclusi), tutte le 9 votazioni")
uv = u[u.vote.isin([1, 2, 3])]
tab = pd.crosstab(uv.membro, uv.vote).rename(columns=lab)
chi2_u, p_u, _, _ = chi2_contingency(tab); V_u = np.sqrt(chi2_u / (tab.values.sum() * (min(tab.shape) - 1)))
R.voce("   tabella (membri=1)", "; ".join(f"{m}: " + ", ".join(f"{k} {v}" for k, v in tab.loc[m].items()) for m in tab.index))
R.voce("   risultato", f"chi2 = {chi2_u:.0f}, p = {p_u:.1e}, V di Cramér = {V_u:.2f}")
ab_uk = (W.loc[uk] == 0).where(W.loc[uk].notna()).mean()
ab_tutte = (Wn == 0).where(Wn.notna()).mean()
pair = pd.DataFrame({"uk": ab_uk, "tutte": ab_tutte}).astype(float).reindex(NAM).dropna()
pair = pair[W.loc[uk, pair.index].notna().sum() >= 5]
w_ab = wilcoxon(pair.uk, pair.tutte)
R.voce("TEST · membri: astensione sull'Ucraina vs abituale", f"mediana {c.pct(pair.uk.median())} vs {c.pct(pair.tutte.median())} "
       f"({len(pair)} membri con >= 5 voti sull'Ucraina); Wilcoxon appaiato p = {w_ab.pvalue:.1e}")
R.lettura("L'astensione non è un'abitudine di un gruppo ma una scelta: si astiene chi non sente propria la lite. "
          "Sui temi del Sud si astiene l'Occidente; sullo scontro Russia-Occidente si astengono i non allineati, "
          "molto più del loro solito.")
R.dati["astensione_test"] = dict(chi2=float(chi2_u), p=float(p_u), V=float(V_u), membri_uk=float(pair.uk.median()),
                                 membri_abituale=float(pair.tutte.median()), p_wilcoxon=float(w_ab.pvalue))

UKs = {}
for m in [1, 0]:
    d_ = uv[uv.membro == m].groupby("rcid").vote.apply(lambda v: (v == 2).mean())
    UKs[m] = d_.reindex(ordine)
et_b = {10791: "Aggressione (mar 22)", 10792: "Conseguenze umanitarie (mar 22)", 10793: "Sospensione dal CDU (apr 22)",
        10797: "Integrità territoriale (ott 22)", 10800: "Riparazioni (nov 22)", 10881: "Pace giusta (feb 23)",
        11063: "Pace in Ucraina (feb 25)", 11064: "Testo USA (feb 25)", 11144: "Bambini ucraini (dic 25)"}
A2 = A.copy()
A2.loc["Ucraina (9 votazioni)"] = dict(n=len(uk), membri=pair.uk.median(),
                                       altri=ab_uk.reindex(ALTRI)[W.loc[uk].reindex(columns=ALTRI).notna().sum() >= 5].median())
order2 = ["Ucraina (9 votazioni)"] + order; y_2 = np.arange(len(order2))[::-1]
# astensione abituale dei membri misurata come a destra: quota sui voti espressi, Ucraina esclusa
Wnu = Wn.drop(index=[r_ for r_ in uk if r_ in Wn.index])[NAM]
abit_pool = float((Wnu == 0).sum().sum() / Wnu.notna().sum().sum())
R.voce("Astensione abituale dei membri (voti espressi, Ucraina esclusa)", c.pct(abit_pool))
fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True)
ax = axs[0]
for yi, t in zip(y_2, order2):
    ax.plot([A2.loc[t, "membri"] * 100, A2.loc[t, "altri"] * 100], [yi, yi], color=c.GRIGIO_CHIARO, lw=2)
ax.scatter(A2.loc[order2, "membri"] * 100, y_2, s=75, color=c.COL_MEMBRI, zorder=3)
ax.scatter(A2.loc[order2, "altri"] * 100, y_2, s=75, color=c.COL_ALTRI, zorder=3)
for t in ["Ucraina (9 votazioni)", "Tutte le votazioni"]:
    ax.text(A2.loc[t, "membri"] * 100 + 1.8, y_2[order2.index(t)], f"{A2.loc[t, 'membri'] * 100:.0f}%", ha="left",
            va="center", fontsize=10, color=c.ACCENTO_TESTO, fontweight="bold")
ax.axhline(y_2[0] - .5, color=c.LINEA, lw=1)
ax.set_yticks(y_2); ax.set_yticklabels(order2); ax.get_yticklabels()[0].set_fontweight("bold"); ax.get_yticklabels()[1].set_fontweight("bold")
ax.set_title("Per tema e sull'Ucraina", fontsize=12); c.griglia(ax, "x")
ax.set_xlabel("% di astensioni (mediana per paese)")
ax = axs[1]; y2 = np.arange(len(ordine))[::-1]
for yi, r_ in zip(y2, ordine):
    ax.plot([UKs[1][r_] * 100, UKs[0][r_] * 100], [yi, yi], color=c.GRIGIO_CHIARO, lw=2)
ax.scatter(UKs[1].values * 100, y2, s=75, color=c.COL_MEMBRI, zorder=3, label="Membri del Movimento")
ax.scatter(UKs[0].values * 100, y2, s=75, color=c.COL_ALTRI, zorder=3, label="Altri paesi")
ax.axvline(abit_pool * 100, color=c.COL_MEMBRI, ls="--", lw=1.4)
ax.text(abit_pool * 100 + .8, y2[-1] - .55, f"astensione abituale\ndei membri ({abit_pool * 100:.0f}%)", fontsize=9, color=c.COL_MEMBRI, va="top")
ax.set_yticks(y2); ax.set_yticklabels([et_b[r_] for r_ in ordine])
ax.set_ylim(y2[-1] - 1.6, y2[0] + .6)
ax.set_title("Le 9 votazioni sull'Ucraina, una per una", fontsize=12); c.griglia(ax, "x")
ax.set_xlabel("% di astensioni sui voti espressi")
ax.set_xlim(0, 70)
fig.legend([_L2([], [], marker="o", ls="", color=c.COL_MEMBRI, ms=9), _L2([], [], marker="o", ls="", color=c.COL_ALTRI, ms=9)],
           ["Membri del Movimento", "Altri paesi"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.02))
fig.subplots_adjust(bottom=.17, wspace=.55)
fig.suptitle("Ci si astiene quando la lite non è la propria", x=.01, ha="left", fontweight="bold", fontsize=13)
R.figura(c.salva(fig, OUT, "astensione_strategia"), "26 (la slide usa figure_slide/slide26_ukraine.py)")

# ------------------------------------------------------------ 5. modello
R.sezione("Modello: i membri votano secondo la propria storia? (slide 27)")
OTTO = ["IND", "IDN", "ZAF", "EGY", "NGA", "SAU", "DZA", "MYS"]
PAESI_MOD = [i for i in NAM if W[i].notna().mean() >= .75]
R.voce("Paesi", f"tutti i membri del Movimento presenti in almeno il 75% delle votazioni ({len(PAESI_MOD)})")
R.voce("Dettaglio per paese", "8 medie potenze: " + ", ".join(c.nome(i) for i in OTTO))
R.voce("Obiettivo", "prevedere il voto (sì / astenuto / no) su ogni votazione")
R.voce("Divisione", "addestramento 2015-2022, verifica 2023-2025 (temporale: niente informazioni dal futuro)")
blocchi = []
for paese in PAESI_MOD:
    d = pd.DataFrame({"y": W[paese], "USA": W.USA, "CHN": W.CHN, "RUS": W.RUS, "maj": meta.maj, "year": meta.year,
                      "date": meta.date, "titolo": meta.titolo, "solo_usa": meta.solo_usa}).join(meta[list(c.TEMI)])
    d = d.sort_values("date")
    ultimo, lag = {}, []
    for r_ in d.itertuples():
        u_ = ultimo.get(r_.titolo)
        lag.append(u_[1] if u_ is not None and u_[0] < r_.year else np.nan)
        if not np.isnan(r_.y) and (u_ is None or u_[0] < r_.year):
            ultimo[r_.titolo] = (r_.year, r_.y)
    d["voto_prec_stesso_testo"] = lag
    for pot in c.POTENZE:
        ag = (d.y == d[pot]).where(d.y.notna() & d[pot].notna()).groupby(d.year).mean()
        d[f"accordo_{pot}_anno_prec"] = d.year.map(ag.shift(1))
    d["paese"] = paese; blocchi.append(d)
D = pd.concat(blocchi).dropna(subset=["y", "USA", "CHN", "RUS"]); D = D[D.year >= 2015]
D["voto_prec_noto"] = D.voto_prec_stesso_testo.notna().astype(int)
D["voto_prec_stesso_testo"] = D.voto_prec_stesso_testo.fillna(9)
D = pd.get_dummies(D, columns=["paese"], dtype=int)
dummies = [k for k in D if k.startswith("paese_")]
gruppi_var = {
    "solo potenze quel giorno": ["USA", "CHN", "RUS", "solo_usa"] + list(c.TEMI) + dummies,
    "solo storia del paese": ["voto_prec_stesso_testo", "voto_prec_noto"] + [f"accordo_{p}_anno_prec" for p in c.POTENZE] + list(c.TEMI) + dummies,
}
gruppi_var["tutto insieme"] = sorted(set(gruppi_var["solo potenze quel giorno"]) | set(gruppi_var["solo storia del paese"]))
for pot_, nm_ in [("USA", "solo USA quel giorno"), ("CHN", "solo Cina quel giorno"), ("RUS", "solo Russia quel giorno")]:
    gruppi_var[nm_] = [pot_] + list(c.TEMI) + dummies
tr, te = D[D.year <= 2022], D[D.year >= 2023]; ytr, yte = tr.y.astype(int), te.y.astype(int)
R.voce("Osservazioni addestramento / verifica", f"{len(tr)} / {len(te)}")
R.voce("Classi nella verifica", ", ".join(f"{ {1: 'sì', 0: 'astenuto', -1: 'no'}[int(k)]} {v:.1%}" for k, v in te.y.value_counts(normalize=True).items()))
ris = {"sempre sì": (float((yte == 1).mean()), 1 / 3)}
F1_ = {}; PRED = {"sempre sì": np.ones(len(yte), int)}
for nm, vv in gruppi_var.items():
    m = RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(tr[vv], ytr)
    pr = m.predict(te[vv]); ris[nm] = (accuracy_score(yte, pr), balanced_accuracy_score(yte, pr))
    F1_[nm] = f1_score(yte, pr, average="macro"); PRED[nm] = pr
    if nm == "tutto insieme":
        rf, pr_tutto = m, pr
R.voce("Termine di paragone: maggioranza", c.pct((te.y == te.maj).mean()))
# regola semplice: ripeti il voto precedente sullo stesso testo, altrimenti sì
regola = np.where(te.voto_prec_noto == 1, te.voto_prec_stesso_testo, 1).astype(int)
ris["regola: voto precedente"] = (accuracy_score(yte, regola), balanced_accuracy_score(yte, regola))
ris = {k: ris[k] for k in ["sempre sì", "regola: voto precedente"]} | {k: v for k, v in ris.items() if k not in ["sempre sì", "regola: voto precedente"]}
f1 = {"sempre sì": f1_score(yte, np.ones(len(yte), int), average="macro"),
      "regola: voto precedente": f1_score(yte, regola, average="macro")}
f1.update(F1_)
PRED["regola: voto precedente"] = regola
for nm, (a, b) in ris.items():
    R.voce(f"Accuratezza / bilanciata / F1 macro: {nm}", f"{c.pct(a)} / {c.pct(b)} / {f1[nm]:.2f}")
R.voce("Quota indovinata per classe (sì / astenuto / no)", "è la base dell'accuratezza bilanciata (media delle tre)")
for nm, pr_ in PRED.items():
    rc = {k: float((np.asarray(pr_)[yte.values == k] == k).mean()) for k in (1, 0, -1)}
    R.voce(f"   {nm}", f"{c.pct(rc[1])} / {c.pct(rc[0])} / {c.pct(rc[-1])}")
    R.dati.setdefault("modello_per_classe", {})[nm] = rc
R.lettura("La regola 'ripeti il voto dell'anno prima sullo stesso testo, altrimenti sì' fa quanto la random "
          "forest sulla storia del paese: per prevedere un membro basta sapere come ha votato l'ultima volta.")
# robustezza: addestramento ribilanciato (classi pesate) per tutti i modelli
R.testo("")
R.voce("Robustezza: addestramento con classi pesate", "class_weight='balanced' (astensioni e no contano di più)")
for nm, vv in gruppi_var.items():
    mb = RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1, class_weight="balanced").fit(tr[vv], ytr)
    prb = mb.predict(te[vv])
    R.voce(f"   {nm}", f"accuratezza {c.pct(accuracy_score(yte, prb))} / bilanciata {c.pct(balanced_accuracy_score(yte, prb))} / F1 macro {f1_score(yte, prb, average='macro'):.2f}")
    R.dati.setdefault("modello_bilanciato", {})[nm] = [float(accuracy_score(yte, prb)), float(balanced_accuracy_score(yte, prb))]
# importanza per permutazione sulle variabili principali (accuratezza bilanciata)
chiave_var = ["voto_prec_stesso_testo", "voto_prec_noto", "USA", "CHN", "RUS", "solo_usa"] + [f"accordo_{p}_anno_prec" for p in c.POTENZE]
Xte = te[gruppi_var["tutto insieme"]]; base = balanced_accuracy_score(yte, pr_tutto); rng_i = np.random.default_rng(0); imp = {}
for v in chiave_var:
    cali = []
    for _ in range(3):
        Xp = Xte.copy(); Xp[v] = rng_i.permutation(Xp[v].values)
        cali.append(base - balanced_accuracy_score(yte, rf.predict(Xp)))
    imp[v] = float(np.mean(cali))
I = pd.Series(imp).sort_values(ascending=False)
R.voce("Importanza per permutazione (calo di acc. bilanciata)", ", ".join(f"{k} {v:.3f}" for k, v in I.head(6).items()))
te2 = te.assign(ok=pr_tutto == yte, paese=te[dummies].idxmax(axis=1).str[6:])
acc_tutti = te2.groupby("paese").ok.mean()
R.voce("Accuratezza per paese, tutti i membri", f"mediana {c.pct(acc_tutti.median())}, min {c.pct(acc_tutti.min())} "
       f"({P.loc[acc_tutti.idxmin(), 'Countryname'].title()}), max {c.pct(acc_tutti.max())}")
acc_p = acc_tutti.reindex(OTTO).sort_values()
R.voce("Accuratezza per paese, 8 medie potenze", ", ".join(f"{c.nome(k)} {v:.0%}" for k, v in acc_p.items()))
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=0).fit(tr[gruppi_var["tutto insieme"]], ytr)
R.voce("Albero di profondità 3", f"accuratezza {c.pct(accuracy_score(yte, dt.predict(te[gruppi_var['tutto insieme']])))}")
R.testo("      (voto_prec_stesso_testo = 9 indica 'nessun voto precedente sullo stesso testo')")
for riga in export_text(dt, feature_names=gruppi_var["tutto insieme"]).splitlines():
    R.testo("      " + riga)
R.lettura("L'informazione più forte è la storia del paese: il voto sullo stesso testo l'ultima volta pesa molto più "
          "di qualunque voto delle potenze (importanza 0,30 contro al massimo 0,06), e una regola di una riga vale "
          "quanto la foresta. Anche i voti delle potenze portano informazione, soprattutto sulle rare astensioni: "
          "con l'addestramento a classi pesate i due modelli si equivalgono. Non è 'seguire': potenze e membri "
          "reagiscono agli stessi testi controversi. A distinguere le due cose è il test sulle minoranze.")
R.dati["modello"] = {k: list(v) for k, v in ris.items()}
R.dati["modello_per_paese"] = acc_p.to_dict()
R.dati["modello_n_paesi"] = len(PAESI_MOD); R.dati["modello_mediana_paesi"] = float(acc_tutti.median())

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
classifica = [("sempre sì", '"Sempre sì"', c.GRIGIO_CHIARO), ("solo USA quel giorno", "Solo come votano gli USA", c.GRIGIO),
              ("solo Cina quel giorno", "Solo come vota la Cina", c.GRIGIO), ("solo Russia quel giorno", "Solo come vota la Russia", c.GRIGIO),
              ("solo potenze quel giorno", "Come votano tutte e tre", c.GRIGIO_SCURO), ("solo storia del paese", "La storia del paese", c.ACCENTO_CHIARO),
              ("regola: voto precedente", 'Regola: "come l\'ultima volta"', c.CORALLO)]
classifica = sorted(classifica, key=lambda t: ris[t[0]][1])
yy3 = np.arange(len(classifica))
bal_ = [ris[k][1] * 100 for k, _, _ in classifica]
ax.barh(yy3, bal_, height=.62, color=[col for _, _, col in classifica])
for yi, (k, _, _), b_ in zip(yy3, classifica, bal_):
    ax.text(b_ + 1, yi, f"{b_:.0f}%", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(yy3); ax.set_yticklabels([e for _, e, _ in classifica], fontsize=10); ax.set_xlim(0, 100)
ax.set_xlabel("Accuratezza bilanciata: media dei successi su sì, astenuto e no")
c.griglia(ax, "x")
ax.set_title(f"Cosa prevede meglio il voto dei {len(PAESI_MOD)} membri?\n(random forest, addestrata 2015–22, verificata 2023–25)", fontsize=12)
ax = axs[1]
bal_paese = pd.Series({p_: balanced_accuracy_score(yte.values[(te2.paese == p_).values], pr_tutto[(te2.paese == p_).values])
                       for p_ in te2.paese.unique()})
R.voce("Accuratezza bilanciata per paese (modello completo)", f"mediana {c.pct(bal_paese.median())}; " +
       ", ".join(f"{c.nome(k)} {bal_paese[k]:.0%}" for k in OTTO))
bp_ = bal_paese.reindex(OTTO).sort_values()
vals_p = list(bp_.values) + [bal_paese.median()]; nomi_p = [c.nome(k) for k in bp_.index] + [f"Membro mediano ({len(PAESI_MOD)})"]
ax.barh(nomi_p, np.array(vals_p) * 100, color=[c.GRIGIO] * len(bp_) + [c.ACCENTO], height=.6)
for i_, v in enumerate(vals_p):
    ax.text(v * 100 + 1, i_, f"{v * 100:.0f}%", va="center", fontsize=10)
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(0, 100); c.griglia(ax, "x"); ax.set_xlabel("Accuratezza bilanciata sul 2023–25 (modello completo)")
R.dati["modello_bilanciata_paese"] = {k: float(v) for k, v in bal_paese.reindex(OTTO).items()} | {"mediana": float(bal_paese.median())}
ax.set_title("Per paese", fontsize=12)
fig.subplots_adjust(wspace=.5)
R.figura(c.salva(fig, OUT, "modello"), "27 (la slide usa figure_slide/slide27_model.py)")

R.salva(OUT, "report_atto3_livello2_non_allineati.txt")
