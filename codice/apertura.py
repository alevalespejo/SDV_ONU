"""Parte 0 - Apertura (slide 1-6).

Numeri del dataset e del Movimento, e il gancio: quanto il paese tipico vota come la Cina
e come gli USA (accordo grezzo, tutte le 1.190 votazioni).
Output: output/atto1_apertura/ (report, figure)
"""
import numpy as np
import comune as c

OUT = c.cartella_atto("atto1_apertura")
plt = c.stile()
df, P, W, meta = c.dati()
R = c.Report("REPORT ATTO 1 - APERTURA: CONTESTO, GANCIO E DOMANDA DI RICERCA")

# ------------------------------------------------------------ 1. contesto
R.sezione("Il dataset in numeri (slide 4)")
R.voce("Votazioni registrate", f"{meta.shape[0]:,}".replace(",", "."))
R.voce("Stati", W.shape[1])
R.voce("Periodo", f"{meta.date.min().date()} - {meta.date.max().date()}")
R.voce("Sessioni ordinarie", f"{meta.session.astype(str).str.isdigit().sum()} votazioni in sessioni "
       f"{', '.join(sorted({s for s in meta.session.astype(str) if s.isdigit()}))}")
R.voce("Sessioni speciali d'emergenza", f"{(~meta.session.astype(str).str.isdigit()).sum()} votazioni "
       f"(10a: Palestina; 11a: Ucraina)")
R.voce("Membri Movimento Non Allineati", f"{P.nam.sum()} su {len(P)} Stati")
for reg, lista in [("Africa", c.NAM_AFRICA), ("Asia e Pacifico", c.NAM_ASIA_PACIFICO),
                   ("America Latina e Caraibi", c.NAM_AMERICA_LATINA), ("Europa", c.NAM_EUROPA)]:
    R.voce(f"   di cui {reg}", len(lista))
R.voce("Non membri (osservatori o mai entrati)", "es. Brasile e Messico (osservatori), "
       "Argentina (uscita nel 1991)")
R.lettura("Il Movimento è la maggioranza dell'Assemblea (62% degli Stati): da solo basta a "
          "far passare quasi ogni risoluzione. È il punto da cui nasce l'effetto maggioranza.")
R.dati["contesto"] = dict(votazioni=int(meta.shape[0]), stati=int(W.shape[1]), nam=int(P.nam.sum()))

# --------------------------------------------------------------- 2. gancio
R.sezione("Il gancio: accordo grezzo con le grandi potenze (slide 2-3)")
acc = {p: c.accordo(W, p) for p in c.POTENZE}
for p, nm in c.POTENZE.items():
    a = acc[p]
    R.voce(f"Accordo con {nm}: mediana", c.pct(a.median()))
    R.voce(f"Accordo con {nm}: Stati sopra l'80%", f"{(a > .8).sum()} su {len(a)} ({c.pct((a > .8).mean(), 0)})")
cina = acc["CHN"]
R.lettura(f"Il paese mediano vota come la Cina nel {c.pct(cina.median(), 0)} delle votazioni e come gli "
          f"USA nel {c.pct(acc['USA'].median(), 0)}; il {c.pct((cina > .8).mean(), 0)} dei paesi supera l'80%. "
          f"Sembra un blocco attorno a Pechino. L'Atto 3 (livello 2) mostrerà che è soprattutto "
          f"l'effetto maggioranza.")
R.dati["gancio"] = dict(mediana_cina=cina.median(), quota_sopra80=float((cina > .8).mean()),
                        n_sopra80=int((cina > .8).sum()))

fig, ax = plt.subplots(figsize=(9, 4.3))
bins = np.arange(0.05, 1.0001, 0.025)
cnt, edges = np.histogram(cina.values, bins=bins)
colori = [c.GRIGIO if e >= 0.8 - 1e-9 else c.GRIGIO_CHIARO for e in edges[:-1]]
ax.bar(edges[:-1] * 100, cnt, width=2.5 * 0.85, align="edge", color=colori)
ax.axvline(80, color=c.INCHIOSTRO, lw=1, ls=":")
ax.text(81, cnt.max() * 0.95, f"{(cina > .8).sum()} paesi su {len(cina)}\nsopra l'80%",
        fontsize=10, color=c.INCHIOSTRO, va="top")
ax.annotate("USA e Israele", (cina[["USA", "ISR"]].mean() * 100 + 1, 1.5), xytext=(cina["USA"] * 100 + 5, 8), fontsize=9, color=c.TENUE,
            arrowprops=dict(arrowstyle="-", color=c.GRIGIO, lw=.6))
ax.set_xlim(5, 100); ax.set_ylim(0, cnt.max() * 1.15)
ax.set_xlabel("% di votazioni in cui il paese vota come la Cina")
ax.set_ylabel("Numero di paesi"); c.griglia(ax)
ax.axvline(cina.median() * 100, color=c.EVIDENZA, lw=2)
ax.text(cina.median() * 100 - 1, cnt.max() * 1.03, f"paese mediano: {cina.median() * 100:.0f}%", ha="right", fontsize=10, color=c.INCHIOSTRO)
ax.set_title(f"Il paese tipico vota come la Cina {cina.median() * 100:.0f} volte su 100 (come gli USA: {acc['USA'].median() * 100:.0f})")
f = c.salva(fig, OUT, "gancio_accordo_cina")
R.figura(f, 3)

# ---------------------------------------------- 3. domanda e obiettivo
R.sezione("Domanda di ricerca (slide 5)")
R.testo("Titolo     : Followers or Free Agents? L'autonomia di voto dei paesi non allineati all'ONU, 2014-2025")
R.testo("Domanda    : I paesi non allineati votano in modo autonomo, o seguono una grande potenza?")
R.testo("Obiettivo  : Misurare quanto il voto dei paesi non allineati sia autonomo rispetto a USA,")
R.testo("             Cina e Russia, distinguendo l'accordo reale dall'effetto maggioranza.")

R.sezione("Le cinque parti (slide 6)")
for parte, dom, sl in [("1 - Dati (1.190 votazioni)", "Cosa dicono i voti e cosa lasciano fuori?", "7-11"),
                       ("2 - Mondo (193 Stati)", "Q1: ci sono blocchi? I non allineati sono della Cina?", "12-19"),
                       ("3 - Non Allineati (119)", "Q2-Q5: effetto maggioranza, seguire, 2022, modello", "20-27"),
                       ("4 - Casi", "Q6: chi fa eccezione, e perché?", "28-31"),
                       ("5 - Risposta", "Followers or free agents?", "32-36")]:
    R.voce(parte, f"{dom}  [slide {sl}]", 34)

R.salva(OUT, "report_atto1_apertura.txt")
