"""Slide 26 - Astensione sulle 9 votazioni sull'Ucraina (sessione speciale ES-11).

Per ogni votazione: quota di astenuti tra i voti espressi, membri contro non membri.
Stampa anche l'astensione mediana per Stato sull'Ucraina, sulle altre votazioni e per tema.
Output: figure/slide26_ukraine.png, figure/s26_votes.csv
"""
import matplotlib.transforms as mtransforms

import base

df = base.carica()
votazioni = df.drop_duplicates("rcid").set_index("rcid")
ucraina = votazioni[votazioni.unres.astype(str).str.contains("ES-11")]
print(ucraina[["date", "unres", "short"]].to_string())

espressi = base.voti_espressi(df).copy()
espressi["mem"] = espressi.Countryname.isin(base.NAM)
espressi["abs"] = espressi.vote == 2
su_ucraina = espressi.rcid.isin(set(ucraina.index))

pv = espressi[su_ucraina].groupby(["rcid", "mem"])["abs"].mean().unstack() * 100
pv["date"] = ucraina.date
pv["res"] = ucraina.unres
pv = pv.sort_values("date")
print(pv.round(0))


def mediana_per_stato(sub):
    return (sub.groupby(["Countryname", "mem"])["abs"].mean() * 100).groupby("mem").median()


print("Ucraina", mediana_per_stato(espressi[su_ucraina]).round(1).to_dict())
print("altre votazioni", mediana_per_stato(espressi[~su_ucraina]).round(1).to_dict())
for tema in ["me", "nu", "di", "hr", "co", "ec"]:
    print(tema, mediana_per_stato(espressi[(espressi[tema] == 1) & ~su_ucraina]).round(1).to_dict())
pv.to_csv(base.OUT / "s26_votes.csv")

# figura
plt = base.stile()
titoli = {"ES-11/1": "Aggression", "ES-11/2": "Humanitarian consequences",
          "ES-11/3": "Russia out of Rights Council", "ES-11/4": "Territorial integrity",
          "ES-11/5": "Reparations", "ES-11/6": "Just and lasting peace", "ES-11/7": "Peace in Ukraine",
          "ES-11/8": "US text: “the path to peace”", "ES-11/9": "Return of Ukrainian children"}
fig, ax = base.nuova_figura(plt, 8.8, 5.0)
n = len(pv)
anni = {}
for i, (_, riga) in enumerate(pv.iterrows()):
    y = n - 1 - i
    membri, altri = riga[True], riga[False]
    ax.plot([altri, membri], [y, y], color="#DCE0E6", lw=3, zorder=1, solid_capstyle="round")
    ax.scatter([altri], [y], s=110, color=base.GRIGIO, zorder=3)
    ax.scatter([membri], [y], s=130, color=base.CORALLO, zorder=3)
    ax.text(membri + 1.8, y, f"{membri:.0f}%", va="center", fontsize=16.5, color=base.CORALLO_TESTO, fontweight="bold")
    codice = "/".join(riga["res"].split("/")[-2:])
    ax.text(-2, y, titoli[codice], ha="right", va="center", fontsize=16.5, color=base.NAVY)
    anni.setdefault(riga["date"][:4], []).append(y)

# anni a sinistra, raggruppati con una linea verticale
tr = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
for anno, ys in anni.items():
    ax.text(0.015, sum(ys) / len(ys), anno, transform=tr, ha="left", va="center", fontsize=16,
            color=base.GRIGIO_TESTO, fontweight="bold")
    ax.plot([0.08, 0.08], [min(ys) - 0.3, max(ys) + 0.3], transform=tr, color="#C5CCDA", lw=1.5, clip_on=False)

ax.axvline(9, color=base.CORALLO_TESTO, lw=1.3, ls=(0, (4, 3)), zorder=0)
ax.text(9.8, n - 0.45, "members usually: 9%", fontsize=14.5, color=base.CORALLO_TESTO, va="bottom")
ax.set_xlim(0, 70)
ax.set_ylim(-0.6, n + 0.05)
base.asse_x_pulito(ax, [0, 20, 40, 60], ["0", "20", "40", "60%"], size=14)
ax.set_xlabel("Abstentions among votes cast", fontsize=14, color=base.GRIGIO_TESTO, labelpad=6)
fig.subplots_adjust(left=.50, right=.95, top=.93, bottom=.14)
base.salva(fig, "slide26_ukraine.png")
