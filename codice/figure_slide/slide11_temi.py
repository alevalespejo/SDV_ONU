"""Slide 11 - Quali temi dividono l'Assemblea.

Per ogni votazione: quota di votanti che esprimono il voto più frequente (100% = unanimità).
Mediana per tema, su tutte le 1.190 votazioni (descrizione dei dati, nessun test).
Una votazione può avere più temi; "No issue code" sono le 141 senza tema.
Output: figure/slide11_temi.png
"""
import base

df = base.carica()
d = df[df.vote.isin([1, 2, 3])]
quota = d.groupby("rcid").vote.agg(lambda s: s.value_counts(normalize=True).iloc[0]) * 100
votazioni = df.drop_duplicates("rcid").set_index("rcid")
TEMI = {"hr": "Human rights", "nu": "Nuclear weapons", "di": "Disarmament", "me": "Middle East",
        "ec": "Economic development", "co": "Colonialism"}

righe = []
for t, nome in TEMI.items():
    r = votazioni.index[votazioni[t] == 1]
    righe.append((nome, len(r), quota[r].median()))
senza = votazioni.index[votazioni[list(TEMI)].sum(axis=1) == 0]
righe.append(("No issue code", len(senza), quota[senza].median()))
righe.sort(key=lambda x: x[2])
tutte = quota.median()
print(f"tutte le votazioni: {tutte:.1f}%")
for nome, n, m in righe:
    print(f"{nome:22s} {n:4d}  {m:.1f}%")

# figura: i due temi più contesi in corallo, il resto in grigio
plt = base.stile()
fig, ax = base.nuova_figura(plt, 8.25, 4.8)
for i, (nome, n, m) in enumerate(righe):
    y = len(righe) - 1 - i
    conteso = i < 2
    ax.barh(y, m, height=.62, color=base.CORALLO if conteso else base.GRIGIO_ASSE, zorder=2)
    ax.text(m + 1, y, f"{m:.0f}%", va="center", fontsize=13, fontweight="bold",
            color=base.CORALLO_TESTO if conteso else base.NAVY)
    ax.text(-1.5, y, f"{nome}  ({n})", va="center", ha="right", fontsize=12.5,
            color=base.CORALLO_TESTO if conteso else base.GRIGIO_TESTO,
            fontweight="bold" if conteso else "normal")
ax.axvline(tutte, color=base.NAVY, lw=1.2, ls=(0, (1, 2)), zorder=3)
ax.text(tutte + .3, len(righe) - .45, f"all votes {tutte:.0f}%", fontsize=11, color=base.NAVY, va="bottom")
ax.set_xlim(0, 108)
ax.set_ylim(-.5, len(righe) - .1)
base.asse_x_pulito(ax, [0, 20, 40, 60, 80, 100], size=12)
ax.set_xlabel("Median share of states casting the most common vote (100% = unanimous)",
              fontsize=11.5, color=base.GRIGIO_TESTO, labelpad=6)
fig.subplots_adjust(left=.31, right=.98, top=.94, bottom=.14)
base.salva(fig, "slide11_temi.png")
