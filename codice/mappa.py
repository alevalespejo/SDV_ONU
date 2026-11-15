"""Slide 4 - Planisfero dei membri del Movimento dei Non Allineati (in corallo).

I piccoli Stati insulari non sono visibili a questa scala: il numero viene stampato.
Output: output/atto1_apertura/figure_slide_en/mappa_membri.png
"""
import geopandas as gpd
import comune as c

OUT = c.cartella_atto("atto1_apertura")
plt = c.stile()
df, P, W, meta = c.dati()
DEST = OUT / "figure_slide_en"
DEST.mkdir(exist_ok=True)
NAM_COL, ALTRI, BORDO = c.ACCENTO, c.GRIGIO_CHIARO, c.SFONDO

mondo = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
mondo = mondo[mondo.continent != "Antarctica"].copy()
mondo.loc[mondo.name == "France", "iso_a3"] = "FRA"
mondo.loc[mondo.name == "Norway", "iso_a3"] = "NOR"
mondo.loc[mondo.name == "Kosovo", "iso_a3"] = "XKX"
mondo.loc[mondo.name == "Somaliland", "iso_a3"] = "SOM"
mondo["nam"] = mondo.iso_a3.isin(c.NAM) | (mondo.iso_a3 == "PSE")
visibili = set(mondo.iso_a3) & c.NAM
mondo = mondo.to_crs("ESRI:54030")
fig, ax = plt.subplots(figsize=(16, 7.6))
mondo[~mondo.nam].plot(ax=ax, color=ALTRI, edgecolor=BORDO, linewidth=.5)
mondo[mondo.nam].plot(ax=ax, color=NAM_COL, edgecolor=BORDO, linewidth=.5)
ax.set_axis_off(); fig.patch.set_facecolor(BORDO); ax.set_facecolor(BORDO)
fig.tight_layout(pad=0)
fig.savefig(DEST / "mappa_membri.png", dpi=200, facecolor=BORDO, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
invis = sorted(c.NAM - visibili)
print("membri visibili sulla mappa:", len(visibili), "su", len(c.NAM), "| troppo piccoli:", len(invis), [c.nome(i) if i in c.NOMI else i for i in invis])
