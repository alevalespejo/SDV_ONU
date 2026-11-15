"""Slide 24 - Controllo del test: quando gli USA sono in minoranza, chi vota con loro?

Stessa misura della slide 22 (quota di votazioni in cui lo Stato vota come gli USA,
tra quelle in cui gli USA votano diversamente dalla maggioranza), su tutti gli Stati.
Se il test trova seguaci dove ci sono (Israele, Micronesia), il risultato sui membri non
dipende dalla misura.
Output: figure/slide24_us.png, figure/s24.csv
"""
import numpy as np
import pandas as pd

import base

df = base.carica()
d = base.voti_espressi(df)
maj = base.maggioranza(d)
vus, minoranza = base.in_minoranza(d, maj, "UNITED STATES")
sub = d[d.rcid.isin(minoranza) & (d.Countryname != "UNITED STATES")].copy()
sub["same"] = sub.vote.values == vus.reindex(sub.rcid).values
r = (sub.groupby("Countryname").same.mean() * 100).to_frame("f")
r["mem"] = r.index.isin(base.NAM)
print(r.groupby("mem").f.median().round(1), r.groupby("mem").size())
print(r.sort_values("f", ascending=False).head(8).round(0))
print(r[r.mem].sort_values("f", ascending=False).head(3).round(1))
print("membri sotto il 2%:", (r[r.mem].f < 2).sum())
r.to_csv(base.OUT / "s24.csv")

# figura
plt = base.stile()
NAVY = base.NAVY
rng = np.random.default_rng(1)
fig, ax = base.nuova_figura(plt, 11, 5.2)
righe = {False: (1.0, base.GRIGIO, base.GRIGIO_TESTO, "Non-members\n(73)"),
         True: (0.0, base.CORALLO, base.CORALLO_TESTO, "Movement\nmembers (119)")}
for mem, (y, c, ct, lab) in righe.items():
    s = r[r.mem == mem].f
    jit = pd.Series(rng.uniform(-.22, .22, len(s)), index=s.index)
    r.loc[s.index, "y"] = y + jit
    ax.scatter(s, y + jit, s=70, color=c, alpha=.75, edgecolor=base.SFONDO, lw=.6, zorder=3)
    m = s.median()
    ax.plot([m, m], [y - .34, y + .34], color=NAVY, lw=3, zorder=4)
    ax.text(m + (1.2 if mem else 0), y + .42, f"median {m:.0f}%", ha="left" if mem else "center",
            va="bottom", fontsize=16, color=NAVY, fontweight="bold")
    ax.text(-2, y, lab, ha="right", va="center", fontsize=16, color=ct, fontweight="bold", linespacing=1.15)

for stato, lab, dy in [("ISRAEL", "Israel 91%", .36), ("MICRONESIA (FEDERATED STATES OF)", "Micronesia 74%", -.36),
                       ("CANADA", "Canada 63%", .36), ("UNITED KINGDOM", "UK 56%", -.36)]:
    v, yv = r.loc[stato, "f"], r.loc[stato, "y"]
    ax.annotate(lab, (v, yv), xytext=(v, 1.0 + dy * 1.25), ha="center", va="center", fontsize=14, color=NAVY,
                arrowprops=dict(arrowstyle="-", color=base.GRIGIO_LINEA, lw=1, shrinkB=5))
png = r.loc["PAPUA NEW GUINEA"]
ax.annotate("Papua New Guinea 12%: the highest member", (png.f, png.y), xytext=(15, png.y - 0.3),
            ha="left", va="center", fontsize=14, color=base.CORALLO_TESTO,
            arrowprops=dict(arrowstyle="-", color=base.CORALLO_TESTO, lw=1))

ax.set_xlim(0, 100)
ax.set_ylim(-0.7, 1.75)
base.asse_x_pulito(ax, [0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], size=14)
ax.set_xlabel("When the US votes against the majority: % of times each country votes with it",
              fontsize=14, color=base.GRIGIO_TESTO, labelpad=8)
fig.subplots_adjust(left=.17, right=.97, top=.97, bottom=.13)
base.salva(fig, "slide24_us.png")
