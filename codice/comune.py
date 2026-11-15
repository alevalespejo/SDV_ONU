"""Modulo comune del progetto "Followers or Free Agents?".

Percorsi, elenco dei 119 membri del Movimento dei Non Allineati, caricamento dei dati
e matrice votazioni x Stati, misure di accordo (con la maggioranza, seguire una potenza
in minoranza, kappa di Cohen), rete di accordo, colori e stile delle figure, classe
Report che scrive il report testuale di ogni parte.

Cartelle attese:
    dataset/un_votes_clean_2014_2025.txt
    codice/        questi script
    output/        creata dagli script
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- percorsi
RADICE = Path(__file__).resolve().parent.parent
CARTELLA_DATI = RADICE / "dataset"
OUTPUT = RADICE / "output"
CACHE = OUTPUT / "cache"
SEED = 42


def file_dati():
    """Il dataset pulito, in dataset/ o nella cartella dataset/ accanto al progetto."""
    for cartella in [CARTELLA_DATI, RADICE.parent / "dataset"]:
        for nome in ["un_votes_clean_2014_2025.txt", "un_votes_clean_2014_2025.csv"]:
            if (cartella / nome).exists():
                return cartella / nome
    raise FileNotFoundError(f"Dataset non trovato: mettilo in {CARTELLA_DATI}")


def cartella_atto(nome):
    p = OUTPUT / nome
    (p / "figure").mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------- Movimento dei Non Allineati
# 119 Stati membri ONU (Palestina esclusa perché non vota). Fonte: Wikipedia,
# "Non-Aligned Movement", elenco dei membri. Osservatori (es. Brasile, Messico,
# Argentina dal 1991 non più membro) NON inclusi.
NAM_AFRICA = ["DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
              "COM", "COG", "COD", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
              "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
              "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
              "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SDN", "TZA", "TGO", "TUN",
              "UGA", "ZMB", "ZWE"]                      # tutta l'Africa tranne il Sud Sudan
NAM_ASIA_PACIFICO = ["AFG", "BHR", "BGD", "BTN", "BRN", "KHM", "FJI", "IND", "IDN",
                     "IRN", "IRQ", "JOR", "KWT", "LAO", "LBN", "MYS", "MDV", "MNG",
                     "MMR", "NPL", "PRK", "OMN", "PAK", "PNG", "PHL", "QAT", "SAU",
                     "SGP", "LKA", "SYR", "THA", "TLS", "TKM", "ARE", "UZB", "VUT",
                     "VNM", "YEM"]
NAM_AMERICA_LATINA = ["ATG", "BHS", "BRB", "BLZ", "BOL", "CHL", "COL", "CUB", "DMA",
                      "DOM", "ECU", "GRD", "GTM", "GUY", "HTI", "HND", "JAM", "NIC",
                      "PAN", "PER", "KNA", "LCA", "VCT", "SUR", "TTO", "VEN"]
NAM_EUROPA = ["AZE", "BLR"]
NAM = set(NAM_AFRICA + NAM_ASIA_PACIFICO + NAM_AMERICA_LATINA + NAM_EUROPA)
assert len(NAM) == 119, len(NAM)

G20 = ["ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU", "IND", "IDN", "ITA", "JPN",
       "KOR", "MEX", "RUS", "SAU", "ZAF", "TUR", "GBR", "USA"]       # UE esclusa
POTENZE = {"USA": "USA", "CHN": "Cina", "RUS": "Russia"}
TEMI = {"me": "Medio Oriente", "nu": "Armi nucleari", "di": "Disarmo",
        "hr": "Diritti umani", "co": "Colonialismo", "ec": "Sviluppo economico"}
NOMI = {"CAF": "Rep. Centrafricana", "USA": "USA", "CHN": "Cina", "RUS": "Russia", "IND": "India", "ZAF": "Sudafrica",
        "MLI": "Mali", "BRA": "Brasile", "ARG": "Argentina", "PAK": "Pakistan",
        "EGY": "Egitto", "SAU": "Arabia Saudita", "IDN": "Indonesia", "TUR": "Turchia",
        "MEX": "Messico", "NGA": "Nigeria", "DZA": "Algeria", "MYS": "Malesia",
        "ISR": "Israele", "GBR": "Regno Unito", "FRA": "Francia", "PRK": "Corea del Nord",
        "IRN": "Iran", "SYR": "Siria", "BLR": "Bielorussia", "UKR": "Ucraina",
        "JPN": "Giappone", "KOR": "Corea del Sud", "DEU": "Germania", "ITA": "Italia"}


def nome(iso):
    return NOMI.get(iso, iso)


# --------------------------------------------------------------- caricamento
def paesi(df):
    """Tabella ccode -> iso3, continente, NAM, G20."""
    import country_converter as coco
    t = df.drop_duplicates("ccode")[["ccode", "Countryname"]].copy()
    cc = coco.CountryConverter()
    t["iso3"] = cc.convert(t.Countryname.str.title().tolist(), to="ISO3", not_found=None)
    t.loc[t.ccode == 640, "iso3"] = "TUR"            # "TÜRKİYE" non riconosciuto
    t["continente"] = cc.convert(t.iso3.tolist(), src="ISO3", to="continent", not_found=None)
    t["nam"] = t.iso3.isin(NAM).astype(int)
    t["g20"] = t.iso3.isin(G20).astype(int)
    assert t.iso3.str.len().eq(3).all() and not t.iso3.duplicated().any()
    assert t.nam.sum() == 119, t.nam.sum()
    return t.set_index("ccode")


def carica():
    """Restituisce (df lungo, tabella paesi indicizzata per iso3)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    c1, c2 = CACHE / "df.pkl", CACHE / "paesi.pkl"
    if c1.exists() and c2.exists():
        return pd.read_pickle(c1), pd.read_pickle(c2)
    df = pd.read_csv(file_dati(), low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    P = paesi(df)
    df["iso3"] = df.ccode.map(P.iso3)
    P = P.reset_index().set_index("iso3")
    df.to_pickle(c1); P.to_pickle(c2)
    return df, P


def matrice(df):
    """
    W: votazioni (righe, rcid) x Stati (colonne, iso3) con +1 sì, 0 astenuto,
       -1 no, NaN assente.
    meta: una riga per votazione con anno, data, sessione, temi, maggioranza,
          flag 'solo_usa' (USA contrari con al massimo altri 2 contrari).
    """
    W = df.pivot(index="rcid", columns="iso3", values="vote_num").sort_index()
    meta = (df.drop_duplicates("rcid").set_index("rcid")
              [["year", "date", "session", "importantvote", *TEMI, "unres", "short"]]
              .sort_index())
    # maggioranza = voto più frequente tra sì/astenuto/no (assenti esclusi)
    cnt = pd.DataFrame({v: (W == v).sum(axis=1) for v in [1, 0, -1]})
    meta["maj"] = cnt.idxmax(axis=1).astype(float)
    meta["maj_pari"] = cnt.eq(cnt.max(axis=1), axis=0).sum(axis=1) > 1
    meta["quota_maj"] = cnt.max(axis=1) / cnt.sum(axis=1)
    meta["n_no"] = cnt[-1]
    meta["solo_usa"] = ((W["USA"] == -1) & (cnt[-1] <= 3)).astype(int)
    meta["n_temi"] = meta[list(TEMI)].sum(axis=1)
    meta["titolo"] = (meta.short.str.split(" : ").str[0].str.lower()
                      .str.replace(r"[^a-z ]", "", regex=True).str.strip())
    return W, meta


def dati():
    df, P = carica()
    W, meta = matrice(df)
    return df, P, W, meta


# ------------------------------------------------------------------- misure
def accordo(W, a, b=None):
    """Quota di voti uguali tra lo Stato a e ogni altro Stato (o lo Stato b),
    sulle votazioni in cui entrambi hanno votato."""
    ref = W[a]
    if b is not None:
        ok = ref.notna() & W[b].notna()
        return float((ref[ok] == W.loc[ok, b]).mean())
    ok = W.notna() & ref.notna().values[:, None]
    return W.eq(ref, axis=0).where(ok).astype(float).mean().drop(a)


def accordo_maggioranza(W, meta):
    return W.eq(meta.maj, axis=0).where(W.notna()).astype(float).mean()


def segue_in_minoranza(W, meta, potenza):
    """Votazioni in cui la potenza vota diversamente dalla maggioranza:
    quota in cui ogni Stato vota come la potenza. Restituisce (serie, n)."""
    idx = W[potenza].notna() & (W[potenza] != meta.maj)
    sub = W.loc[idx]
    s = sub.eq(sub[potenza], axis=0).where(sub.notna()).astype(float).mean().drop(potenza)
    return s, int(idx.sum())


def kappa(W):
    """Kappa di Cohen per ogni coppia di Stati (3 categorie), sulle votazioni comuni.
    Corregge l'accordo osservato per quello atteso dalla frequenza dei voti."""
    P = W.notna().values.astype(float)
    n = P.T @ P
    oss = np.zeros_like(n); att = np.zeros_like(n)
    for c in [1, 0, -1]:
        M = (W.values == c).astype(float)
        oss += M.T @ M
        Mp = M.T @ P
        att += Mp * Mp.T
    with np.errstate(all="ignore"):
        po, pe = oss / n, att / n ** 2
        k = (po - pe) / (1 - pe)
    return (pd.DataFrame(k, index=W.columns, columns=W.columns),
            pd.DataFrame(n, index=W.columns, columns=W.columns))


def rete(K, N, soglia, min_comuni=20):
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(K.index)
    a = K.values; i, j = np.triu_indices(len(K), 1)
    m = (N.values[i, j] >= min_comuni) & (a[i, j] >= soglia)
    G.add_weighted_edges_from(zip(K.index[i[m]], K.index[j[m]], a[i, j][m]))
    return G


def dividi_votazioni(meta, seed=SEED):
    """Metà A (costruire i cluster) e metà B (descriverli), stratificate per anno."""
    rng = np.random.default_rng(seed); A, B = [], []
    for _, g in meta.groupby("year"):
        idx = g.index.values.copy(); rng.shuffle(idx); h = len(idx) // 2
        A += list(idx[:h]); B += list(idx[h:])
    return A, B


# -------------------------------------------------------------------- colori
# Ogni colore ha un solo significato in tutte le figure:
#   corallo  = membri del Movimento e Sud globale, e ciò che va guardato quando non è un gruppo
#   navy     = testo, e seconda evidenza quando il corallo è già usato
#   ocra     = Cina e blocco Cina-Russia (Russia in ocra scuro)
#   ardesia  = USA e blocco USA-Israele;  viola = Europa e Occidente;  lampone = America Latina (solo rete)
#   voti     = sì blu, astenuto grigio chiaro, no navy, assente grigio chiarissimo
SFONDO, SCHEDA = "#F7F7F5", "#FFFFFF"
INCHIOSTRO, TENUE, ASSE = "#14213D", "#5F6A7A", "#9AA2AF"
GRIGIO, GRIGIO_CHIARO, GRIGLIA, LINEA = "#B4BAC4", "#D2D6DD", "#E6E9EE", "#DCE0E6"
GRIGIO_SCURO, GRIGIO_MEDIO = "#8B93A1", "#5E6778"
CORALLO, ACCENTO_CHIARO, ACCENTO_TESTO = "#E76F51", "#F4B8A8", "#B8452C"
ACCENTO = EVIDENZA = COL_MEMBRI = CORALLO
COL_ALTRI = CONFRONTO = GRIGIO_SCURO
SECONDA = INCHIOSTRO
COL_CINA, COL_RUSSIA, COL_USA = "#E9B949", "#9C7414", "#33415E"
TESTO_OCRA = COL_RUSSIA
NERO = COL_USA
COL_POTENZE = {"USA": COL_USA, "CHN": COL_CINA, "RUS": COL_RUSSIA}
GCOL = {"Maggioranza del Sud globale": CORALLO, "Europa e alleati": "#7B6BA8",
        "Blocco Cina–Russia": COL_CINA, "USA, Israele e alleati stretti": COL_USA}
CCOL = {"Sud globale": CORALLO, "Occidente, USA e Israele": "#7B6BA8",
        "Cina, Russia, India, Iran…": COL_CINA, "America Latina e Pacifico": "#C44E6E"}
BLU_SI, PRUGNA_NO = "#4A6FA5", INCHIOSTRO
COL_VOTI = {"Sì": BLU_SI, "Astenuto": "#CDD3DB", "No": INCHIOSTRO, "Assente": "#EEF0F3"}
SCALA_CINA = ["#ECEDEF", "#F6E3B4", "#EDC765", "#C99A2E", "#7A5A0E"]   # mappa: dal grigio all'ocra
DIVERGENTE = ["#6F8FAF", "#F7F7F5", "#E76F51"]                          # heatmap: sotto / sulla / sopra la media


def T(it, en):
    """Testo nella lingua della figura: inglese se FIG_EN è impostata."""
    import os
    return en if os.environ.get("FIG_EN") else it


def stile():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in list((RADICE / "fonts").glob("*.ttf")) + list((RADICE / "fonts").glob("*.otf")):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Inter", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": ASSE, "text.color": INCHIOSTRO, "axes.labelcolor": TENUE,
        "xtick.color": TENUE, "ytick.color": TENUE,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "figure.facecolor": SFONDO, "axes.facecolor": SFONDO, "savefig.facecolor": SFONDO})
    return plt


SCALA_TESTO = 1.3     # tutti i testi delle figure ingranditi del 30%, per la proiezione


def salva(fig, cartella, nome_file, rect_layout=None):
    """Salva la figura in figure/ con i testi ingranditi di SCALA_TESTO.
    FIG_SLIDE=1: versione per le slide, senza titolo (lo dà la slide), in figure_slide/.
    FIG_EN=1: testi tradotti con traduzioni.py, nella cartella con suffisso _en."""
    import os
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    sottocartella = "figure"
    if os.environ.get("FIG_SLIDE"):
        sottocartella = "figure_slide"
        if fig._suptitle is not None:
            fig._suptitle.set_text("")
        assi = [a for a in fig.axes if a.get_label() != "<colorbar>"]
        if len(assi) == 1:
            assi[0].set_title("")
    if os.environ.get("FIG_EN"):
        from traduzioni import traduci
        fig.canvas.draw()                                   # servono le etichette degli assi già create
        for ax_ in fig.axes:
            for asse_ in ("x", "y"):
                testi = [e.get_text() for e in getattr(ax_, f"get_{asse_}ticklabels")()]
                nuovi = [traduci(t_) if t_ else t_ for t_ in testi]
                if nuovi != testi:
                    getattr(ax_, f"set_{asse_}ticks")(getattr(ax_, f"get_{asse_}ticks")())
                    getattr(ax_, f"set_{asse_}ticklabels")(nuovi)
        for t in fig.findobj(Text):
            if t.get_text():
                t.set_text(traduci(t.get_text()))
        sottocartella += "_en"
    (Path(cartella) / sottocartella).mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize() * SCALA_TESTO)
    if rect_layout is not None:                             # figure a più pannelli: spazi ricalcolati dopo l'ingrandimento
        fig.tight_layout(rect=rect_layout, h_pad=2.2, w_pad=2.5)
    p = Path(cartella) / sottocartella / f"{nome_file}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p.name


def griglia(ax, asse="y"):
    getattr(ax, f"{asse}axis").grid(True, color=GRIGLIA); ax.set_axisbelow(True)


# ------------------------------------------------------------------- report
class Report:
    """Report testuale di una parte dell'analisi, più risultati.json con i numeri principali."""
    L = 88

    def __init__(self, titolo):
        self.righe = ["=" * self.L, f" {titolo}", "=" * self.L]
        self.n = 0
        self.dati = {}

    def sezione(self, titolo):
        self.n += 1
        self.righe += ["", f"SEZIONE {self.n}: {titolo.upper()}"]

    def voce(self, chiave, valore, larghezza=38):
        self.righe.append(f"  - {chiave:<{larghezza}}: {valore}")

    def testo(self, t, rientro=2):
        self.righe.append(" " * rientro + t)

    def lettura(self, t):
        self.righe.append(f"   -> LETTURA: {t}")

    def figura(self, nome_file, slide):
        self.righe.append(f"   -> FIGURA: figure/{nome_file}   (slide {slide})")

    def attenzione(self, t):
        self.righe.append(f"   [ATTENZIONE] {t}")

    def salva(self, cartella, nome_file):
        self.righe += ["", "=" * self.L]
        p = Path(cartella) / nome_file
        p.write_text("\n".join(self.righe), encoding="utf-8")
        with open(Path(cartella) / "risultati.json", "w", encoding="utf-8") as f:
            json.dump(self.dati, f, ensure_ascii=False, indent=2, default=float)
        print("\n".join(self.righe))
        return p


def pct(x, dec=1):
    return f"{100 * x:.{dec}f}%"
