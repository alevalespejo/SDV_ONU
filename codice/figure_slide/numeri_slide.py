"""Numeri citati nelle slide 3, 29 e 33 che non vengono da una figura.

Slide 3:  accordo mediano con Cina e USA, con e senza le 282 votazioni "USA contro tutti".
Slide 29: Pakistan e Sudafrica quando una potenza è in minoranza, contro il proprio caso
          (binomiale, Benjamini-Hochberg su tutti i 192 paesi), e i paesi più vicini (kappa).
Slide 33: quante votazioni hanno esattamente la quota mediana del voto più frequente.
"""
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

import base

df = base.carica()
pres = base.presenza(df)


def accordo_mediano(d, potenza):
    v = d[d.Countryname == potenza].set_index("rcid").vote
    x = d[d.Countryname != potenza].copy()
    x["same"] = x.vote.values == v.reindex(x.rcid).values
    x = x[x.rcid.isin(v.index)]
    return x.groupby("Countryname").same.mean()


# slide 3
tutte = df[df.vote.isin([1, 2, 3])]
senza = base.voti_espressi(df)
for nome, d in [("tutte le 1.190", tutte), ("senza le 282", senza)]:
    cina = accordo_mediano(d, "CHINA")
    usa = accordo_mediano(d, "UNITED STATES")
    print(f"slide 3, {nome}: Cina {cina.median():.0%}, USA {usa.median():.0%}, "
          f"paesi sopra l'80% con la Cina {(cina > 0.8).sum()} su {len(cina)}")

# slide 29
maj = base.maggioranza(senza)
righe = []
for P in ["CHINA", "RUSSIAN FEDERATION", "UNITED STATES"]:
    vp, minoranza = base.in_minoranza(senza, maj, P)
    sub = senza[senza.rcid.isin(minoranza) & (senza.Countryname != P)].copy()
    sub["pv"] = vp.reindex(sub.rcid).values
    for c, g in sub.groupby("Countryname"):
        fc = g.vote.value_counts(normalize=True)
        fp = g.pv.value_counts(normalize=True)
        e = sum(fc.get(v, 0) * fp.get(v, 0) for v in [1, 2, 3])
        k = int((g.vote == g.pv).sum())
        righe.append((P, c, k / len(g), e, binomtest(k, len(g), e, alternative="greater").pvalue))
r = pd.DataFrame(righe, columns=["P", "c", "oss", "att", "p"])
r["p_bh"] = r.groupby("P").p.transform(lambda p: false_discovery_control(p.values, method="bh"))
for c in ["PAKISTAN", "SOUTH AFRICA"]:
    for _, x in r[r.c == c].iterrows():
        print(f"slide 29, {c} con {x.P}: {x.oss:.1%} (per caso {x.att:.1%}), p BH = {x.p_bh:.3f}")

kappa = pd.read_csv(base.ANALISI / "atto3_livello1_mondo" / "kappa.csv", index_col=0)
for iso in ["PAK", "ZAF"]:
    print(f"slide 29, più vicini a {iso}:", list(kappa[iso].drop(iso).sort_values(ascending=False).index[:5]))

# slide 33
quota = tutte.groupby("rcid").vote.agg(lambda s: s.value_counts(normalize=True).iloc[0])
uguali = ((quota * 100).round() == round(quota.median() * 100)).sum()
print(f"slide 33: quota mediana {quota.median():.1%}; votazioni con quota {quota.median():.0%} (arrotondata): {uguali}")
