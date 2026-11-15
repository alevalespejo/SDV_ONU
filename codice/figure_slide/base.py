"""Parti comuni agli script delle figure delle slide 14-27.

Dati, filtri usati in tutto il progetto, elenco dei membri del Movimento e colori.
"""
from pathlib import Path

import pandas as pd

QUI = Path(__file__).resolve().parent
NOME_DATI = "un_votes_clean_2014_2025.txt"
OUT = QUI / "figure"
OUT.mkdir(exist_ok=True)
# risultati della pipeline principale (codice/esegui_tutto.py), letti da slide 17 e 29
ANALISI = QUI.parent.parent / "output"

# colori del deck
SFONDO = "#F7F7F5"
NAVY = "#14213D"
CORALLO, CORALLO_TESTO = "#E76F51", "#B8452C"
GRIGIO, GRIGIO_TESTO = "#9AA1AD", "#5F6A7A"
GRIGIO_LINEA, GRIGIO_ASSE, GRIGIO_GRIGLIA = "#8B93A1", "#B4BAC4", "#E4E7EB"

# 119 Stati membri ONU del Movimento dei Non Allineati (nomi come nel dataset)
NAM = [
    # Africa
    "ALGERIA", "ANGOLA", "BENIN", "BOTSWANA", "BURKINA FASO", "BURUNDI", "CABO VERDE",
    "CAMEROON", "CENTRAL AFRICAN REPUBLIC", "CHAD", "COMOROS", "CONGO",
    "DEMOCRATIC REPUBLIC OF THE CONGO", "COTE D'IVOIRE", "DJIBOUTI", "EGYPT",
    "EQUATORIAL GUINEA", "ERITREA", "ESWATINI", "ETHIOPIA", "GABON", "GAMBIA", "GHANA",
    "GUINEA", "GUINEA-BISSAU", "KENYA", "LESOTHO", "LIBERIA", "LIBYA", "MADAGASCAR",
    "MALAWI", "MALI", "MAURITANIA", "MAURITIUS", "MOROCCO", "MOZAMBIQUE", "NAMIBIA",
    "NIGER", "NIGERIA", "RWANDA", "SAO TOME AND PRINCIPE", "SENEGAL", "SEYCHELLES",
    "SIERRA LEONE", "SOMALIA", "SOUTH AFRICA", "SUDAN", "UNITED REPUBLIC OF TANZANIA",
    "TOGO", "TUNISIA", "UGANDA", "ZAMBIA", "ZIMBABWE",
    # Asia
    "AFGHANISTAN", "BAHRAIN", "BANGLADESH", "BHUTAN", "BRUNEI DARUSSALAM", "CAMBODIA",
    "INDIA", "INDONESIA", "IRAN (ISLAMIC REPUBLIC OF)", "IRAQ", "JORDAN", "KUWAIT",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC", "LEBANON", "MALAYSIA", "MALDIVES", "MONGOLIA",
    "MYANMAR", "NEPAL", "DEMOCRATIC PEOPLE'S REPUBLIC OF KOREA", "OMAN", "PAKISTAN",
    "PHILIPPINES", "QATAR", "SAUDI ARABIA", "SINGAPORE", "SRI LANKA",
    "SYRIAN ARAB REPUBLIC", "THAILAND", "TIMOR-LESTE", "TURKMENISTAN",
    "UNITED ARAB EMIRATES", "UZBEKISTAN", "VIET NAM", "YEMEN", "AZERBAIJAN",
    # America Latina e Caraibi
    "ANTIGUA AND BARBUDA", "BAHAMAS", "BARBADOS", "BELIZE",
    "BOLIVIA (PLURINATIONAL STATE OF)", "CHILE", "COLOMBIA", "CUBA", "DOMINICA",
    "DOMINICAN REPUBLIC", "ECUADOR", "GRENADA", "GUATEMALA", "GUYANA", "HAITI",
    "HONDURAS", "JAMAICA", "NICARAGUA", "PANAMA", "PERU", "SAINT KITTS AND NEVIS",
    "SAINT LUCIA", "SAINT VINCENT AND THE GRENADINES", "SURINAME", "TRINIDAD AND TOBAGO",
    "VENEZUELA (BOLIVARIAN REPUBLIC OF)",
    # Europa e Pacifico
    "BELARUS", "FIJI", "PAPUA NEW GUINEA", "VANUATU",
]


def file_dati():
    for cartella in [QUI, QUI.parent.parent / "dataset", QUI.parent.parent.parent / "dataset"]:
        if (cartella / NOME_DATI).exists():
            return cartella / NOME_DATI
    raise FileNotFoundError(f"{NOME_DATI} non trovato: mettilo in dataset/")


def carica():
    df = pd.read_csv(file_dati(), low_memory=False)
    assert set(NAM) <= set(df.Countryname) and len(set(NAM)) == 119
    return df


def usa_contro_tutti(df):
    """Votazioni in cui gli USA votano no e al massimo altri 2 Stati votano no (282)."""
    usa = df[df.ccode == 2].set_index("rcid").vote
    n_no = df[df.vote == 3].groupby("rcid").size()
    return [r for r in usa.index if usa[r] == 3 and n_no.get(r, 0) <= 3]


def presenza(df):
    """Quota di votazioni in cui ogni Stato è presente (tutte le 1.190)."""
    return df.groupby("Countryname").is_absent.apply(lambda s: 1 - s.mean())


def voti_espressi(df):
    """Sì, astenuto, no (1, 2, 3) senza le votazioni "USA contro tutti"."""
    return df[~df.rcid.isin(usa_contro_tutti(df)) & df.vote.isin([1, 2, 3])]


def maggioranza(d):
    """Voto più frequente di ogni votazione."""
    return d.groupby("rcid").vote.agg(lambda s: s.value_counts().idxmax())


def in_minoranza(d, maj, stato):
    """Voti dello Stato e votazioni in cui vota diversamente dalla maggioranza."""
    v = d[d.Countryname == stato].set_index("rcid").vote
    return v, v[v != maj.reindex(v.index)].index


def stile():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in list((QUI.parent.parent / "fonts").glob("*.otf")) + fm.findSystemFonts():
        if "inter/Inter" in str(f) or Path(f).name.startswith("Inter-"):
            fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Inter"})
    return plt


def nuova_figura(plt, w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(SFONDO)
    ax.set_facecolor(SFONDO)
    return fig, ax


def asse_x_pulito(ax, ticks, etichette=None, size=13):
    """Solo l'asse x, con griglia verticale leggera."""
    ax.set_xticks(ticks)
    if etichette:
        ax.set_xticklabels(etichette)
    ax.tick_params(axis="x", labelsize=size, colors=GRIGIO_TESTO, length=0)
    ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRIGIO_ASSE)
    ax.grid(axis="x", color=GRIGIO_GRIGLIA, lw=1)
    ax.set_axisbelow(True)


def salva(fig, nome):
    fig.savefig(OUT / nome, dpi=200, facecolor=SFONDO)
    print("salvata", OUT / nome)
