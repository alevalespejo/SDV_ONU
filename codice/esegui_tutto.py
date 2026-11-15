"""Esegue tutta l'analisi in ordine, dalla cartella codice/:  python esegui_tutto.py

Con FIG_EN=1 FIG_SLIDE=1 le figure escono in inglese e senza titolo, come nelle slide.
Circa 5 minuti. Le figure ridisegnate per il deck sono in figure_slide/ (vedi README).
"""
import runpy
import time
from pathlib import Path

SCRIPT = ["atto1_apertura.py", "atto1_mappa.py", "atto2_dati.py", "atto3_livello1_mondo.py",
          "atto3_livello2_non_allineati.py", "atto3_livello3_casi.py", "atto4_dashboard.py",
          "atto5_chiusura.py", "extra_visualizzazioni.py", "figura_rete.py", "figure_aggiuntive.py"]

if __name__ == "__main__":
    qui = Path(__file__).resolve().parent
    for s in SCRIPT:
        t = time.time()
        print(f"\n>>> {s}")
        runpy.run_path(str(qui / s), run_name="__main__")
        print(f"<<< {s}: {time.time() - t:.0f} s")
