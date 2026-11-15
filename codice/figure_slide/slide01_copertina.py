"""Slide 1 - Copertina: i 193 Stati ONU, uno per punto; pieni i 119 membri del Movimento.
Output: figure/slide01_copertina.png (sfondo trasparente)
"""
import base

plt = base.stile()
fig, ax = plt.subplots(figsize=(4.72, 2.9), dpi=100)
fig.patch.set_alpha(0)
ax.set_facecolor("none")
PER_RIGA = 18
for i in range(193):
    x, y = i % PER_RIGA, -(i // PER_RIGA)
    if i < 119:
        ax.scatter(x, y, s=150, color=base.CORALLO, lw=0)
    else:
        ax.scatter(x, y, s=130, facecolor="none", edgecolor="#4A5D78", lw=1.6)
ax.set_xlim(-.6, PER_RIGA - .4)
ax.set_ylim(-10.6, .6)
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(base.OUT / "slide01_copertina.png", dpi=200, transparent=True)
