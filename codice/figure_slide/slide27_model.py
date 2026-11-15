"""Slide 27 - Cosa prevede meglio il voto di un membro.

Accuratezza bilanciata sulle votazioni 2023-25 (mai viste in addestramento), 103 membri.
I valori vengono dal modello (random forest e ablazioni) nel report di
atto3_livello2_non_allineati.py; qui si disegna solo il grafico.
Output: figure/slide27_model.png
"""
import base

# (etichetta, accuratezza bilanciata %, colore, in grassetto)
RISULTATI = [("Rule: “same as last time”", 69, "#E76F51", True),
             ("Everything together", 67, "#F2A58F", False),
             ("The country’s own history", 66, "#F2A58F", False),
             ("All three powers", 61, "#6B7485", False),
             ("Only how Russia votes", 59, base.GRIGIO, False),
             ("Only how China votes", 58, base.GRIGIO, False),
             ("Only how the US votes", 51, base.GRIGIO, False),
             ("“Always yes” (floor)", 33, "#DCE0E6", False)]

plt = base.stile()
NAVY = base.NAVY
fig, ax = base.nuova_figura(plt, 11, 5.6)
for y, (lab, v, c, grassetto) in zip(range(len(RISULTATI) - 1, -1, -1), RISULTATI):
    peso = "bold" if grassetto else "normal"
    ax.barh(y, v, height=.66, color=c, zorder=2)
    ax.text(v + 1, y, f"{v}%", va="center", fontsize=16, color=NAVY, fontweight=peso)
    ax.text(-1.5, y, lab, va="center", ha="right", fontsize=16, color=NAVY, fontweight=peso)


def parentesi(y1, y2, testo, colore):
    x = 86
    ax.plot([x, x + 1.2, x + 1.2, x], [y1 + .3, y1 + .3, y2 - .3, y2 - .3], color=colore, lw=1.4, clip_on=False)
    ax.text(x + 2.4, (y1 + y2) / 2, testo, va="center", ha="left", fontsize=14, color=colore, linespacing=1.2)


parentesi(7, 5, "use the\ncountry’s own past", base.CORALLO_TESTO)
parentesi(4, 1, "watch only the\ngreat powers", base.GRIGIO_TESTO)
ax.axvline(33, color=base.GRIGIO_ASSE, lw=1, ls=(0, (3, 3)), zorder=1)
ax.set_xlim(0, 100)
ax.set_ylim(-.6, 7.6)
base.asse_x_pulito(ax, [0, 25, 50, 75, 100])
ax.set_xlabel("Balanced accuracy on 2023–25 votes, never seen in training (%)",
              fontsize=13, color=base.GRIGIO_TESTO, labelpad=8)
fig.subplots_adjust(left=.28, right=.83, top=.98, bottom=.13)
base.salva(fig, "slide27_model.png")
