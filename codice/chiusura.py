"""Parte 5 - Risposta (slide 32-36).

Legge i risultati.json delle parti precedenti e scrive i numeri chiave, la risposta
in tre frasi, cosa l'analisi stabilisce e cosa no, limiti e sviluppi.
Output: output/atto5_chiusura/report_atto5_chiusura.txt
"""
import json
import comune as c

OUT = c.cartella_atto("atto5_chiusura")


def leggi(cartella):
    return json.load(open(c.OUTPUT / cartella / "risultati.json", encoding="utf-8"))


a1, a2 = leggi("atto1_apertura"), leggi("atto2_dati")
l1, l2, l3 = leggi("atto3_livello1_mondo"), leggi("atto3_livello2_non_allineati"), leggi("atto3_livello3_casi")
R = c.Report("REPORT ATTO 5 - CHIUSURA: RISPOSTA, LIMITI E SVILUPPI")
p = c.pct

R.sezione("Numeri chiave (per controllare le slide)")
R.voce("Votazioni / Stati / membri del Movimento", f"{a2['votazioni']} / {a1['contesto']['stati']} / {a1['contesto']['nam']}")
R.voce("Gancio: accordo mediano con la Cina", p(a1["gancio"]["mediana_cina"], 0))
R.voce("Gancio: paesi sopra l'80% con la Cina", f"{a1['gancio']['n_sopra80']} ({p(a1['gancio']['quota_sopra80'], 0)})")
R.voce("La Cina vota con la maggioranza", p(l2["prima_dopo"]["cina_con_maggioranza"], 0))
R.voce("Correlazione accordo-Cina / accordo-maggioranza", f"r = {l2['prima_dopo']['r']:.2f}")
R.voce("Membro mediano segue Cina / Russia / USA in minoranza",
       f"{p(l2['segue']['CHN'], 0)} / {p(l2['segue']['RUS'], 0)} / {p(l2['segue']['USA'], 0)}")
R.voce("India segue Cina / Russia / USA in minoranza",
       f"{p(l2['segue_india']['CHN'], 0)} / {p(l2['segue_india']['RUS'], 0)} / {p(l2['segue_india']['USA'], 0)}")
R.voce("PCA: varianza PC1", p(l1["pca"]["pc1"], 0))
R.voce("Cluster (k = 4)", ", ".join(f"{k} {v}" for k, v in l1["cluster"].items()))
R.voce("Cluster x Movimento: V di Cramér", f"{l1['cramer_nam']:.2f}")
R.voce("Rete: modularità Q / India con la Cina", f"{l1['rete']['Q']:.2f} / {l1['rete']['india_cina_anni']} anni su 12")
m = l2["modello"]
R.voce("Modello: sempre sì / potenze / storia", f"{p(m['sempre sì'][0], 0)} / {p(m['solo potenze quel giorno'][0], 0)} / {p(m['solo storia del paese'][0], 0)}")
R.voce("Mali: segue la Russia in minoranza", f"{p(l3['MLI']['segue']['RUS_2014-21'], 0)} -> {p(l3['MLI']['segue']['RUS_2022-25'], 0)}")
R.voce("USA con la maggioranza nel 2025", p(l1["allineamento"]["USA"]["2025"], 0))

R.sezione("Risposta alla domanda in tre frasi (slide 34)")
R.testo(f"1. L'accordo alto con la Cina (paese mediano {p(a1['gancio']['mediana_cina'], 0)}) è un effetto maggioranza:")
R.testo(f"   la Cina vota con la maggioranza nel {p(l2['prima_dopo']['cina_con_maggioranza'], 0)} dei casi, e chi vota con la maggioranza")
R.testo("   vota per forza come lei.")
R.testo(f"2. Quando una potenza resta in minoranza, il membro mediano del Movimento la segue solo nel")
R.testo(f"   {p(l2['segue']['CHN'], 0)} dei casi (Cina), {p(l2['segue']['RUS'], 0)} (Russia), {p(l2['segue']['USA'], 0)} (USA): i non allineati votano con la")
R.testo("   maggioranza e secondo la propria storia, non seguendo una potenza.")
R.testo("3. Le eccezioni esistono e sono poche: l'India segue spesso la Cina, il Mali si è avvicinato")
R.testo("   alla Russia dopo il 2022; sull'Ucraina la maggioranza dei membri non segue Mosca, si astiene.")

R.sezione("Cosa l'analisi stabilisce e cosa no (slide 35)")
R.testo("STABILISCE")
for t in ["Una struttura stabile dell'Assemblea 2014-2025: due grandi masse e due piccoli poli,",
          "  ritrovata con quattro metodi (PCA, UMAP, cluster, rete) e robusta ai parametri.",
          "Che i cluster, costruiti solo sui voti, coincidono con l'appartenenza al Movimento",
          f"  (V di Cramér {l1['cramer_nam']:.2f}), variabile mai usata per costruirli.",
          "Che nei casi in cui seguire una potenza e votare con la maggioranza divergono, quasi",
          "  nessun membro segue la potenza."]:
    R.testo("  " + t)
R.testo("NON STABILISCE")
for t in ["Le cause del voto: sono associazioni, non effetti (es. Mali e colpi di stato).",
          "Il comportamento sulle risoluzioni adottate per consenso: il dataset contiene solo le",
          "  votazioni registrate, cioè i casi contesi.",
          "L'allineamento politico in generale: un voto all'ONU non è un'alleanza.",
          "Risultati per tema sulle 141 votazioni senza tema (tra cui 7 delle 9 sull'Ucraina)."]:
    R.testo("  " + t)

R.sezione("Limiti e sviluppi (slide 35)")
for k, v in [("Finestra temporale", "solo 2014-2025: estendere al 1946 per vedere la fine della Guerra fredda"),
             ("Temi", "classificazione di Voeten non esaustiva: colonna 'tema_esteso' con regole dichiarate"),
             ("2025", "le votazioni 'USA contro tutti' cambiano l'agenda: confronti tra anni con cautela"),
             ("Seguire = stesso voto", "include le astensioni uguali; non distingue un 'no' da un'astensione"),
             ("Reti", "nel syllabus ma non trattate a lezione: usate solo come conferma"),
             ("Variabili esterne", "aggiungere alleanze (ATOP), aiuti e commercio per spiegare le eccezioni")]:
    R.voce(k, v)
R.salva(OUT, "report_atto5_chiusura.txt")
