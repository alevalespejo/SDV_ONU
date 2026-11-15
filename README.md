# Followers or Free Agents? Voting autonomy of non-aligned countries in the UN General Assembly, 2014–2025

## Overview

### Data
Recorded votes of the UN General Assembly from the dataset of Voeten, Strezhnev and Bailey ([doi:10.7910/DVN/LEJUQZ](https://doi.org/10.7910/DVN/LEJUQZ)), sessions held in **2014–2025**: **1,190 votes** cast by **193 states**, 229,670 rows after cleaning. The 119 UN members of the Non-Aligned Movement are listed by hand from the Movement's membership (Palestine is excluded because it does not vote; Brazil and Mexico are observers).

Five decisions are applied everywhere:
- **Absences** (19,805) are missing data, never counted as yes, abstain or no.
- **"US against all" votes** (282, 114 of them in 2025: the US votes no with at most two other states) describe the data but are left out of every test and comparison.
- **Votes with no issue code** (141, including Ukraine) are left out of the issue analysis; the nine Ukraine votes are studied on their own.
- **Majority** = the most frequent vote among yes, abstain and no.
- **Following a power** = voting like it when it votes against the majority, always compared with the agreement expected **by chance** from the country's own mix of yes, abstain and no.

PCA, UMAP, clustering and the network use the **172 states present in at least 75% of the votes**. Groups are built on one half of the votes and described on the other half.

### Questions
1. Are there voting blocs, and are the non-aligned China's?
2. Is the 78% agreement with China real?
3. When a power stands alone, who follows it?
4. Did 2022 change anything?
5. What best predicts a member's vote?
6. Who breaks the pattern, and why?

### Main results
1. **Four stable blocs**, found with four methods. PCA puts 60% of the variance on one axis (Europe and allies vs the Global South); UMAP draws the same divide in 9 runs out of 9; k-means with k = 4 gives Global South (107), Europe and allies (47), China–Russia (11), US–Israel (7), stable (bootstrap ARI 0.87) and close to Ward (0.95). The groups match the Movement without ever having seen it (Cramér's V 0.83, at most 0.33 with shuffled labels). In the agreement network (Cohen's κ ≥ 0.4, Louvain, Q = 0.51) there are **0 strong ties** between the West and the Global South; the only bridge (66 ties) runs through China–Russia.
2. **Mostly a majority effect.** China votes with the majority 67–83% of the time, so the crowd resembles China: across 192 countries, voting with the majority and voting like China correlate at r = 0.92 (r² = 0.85).
3. **Almost no member follows a power that stands alone.** The median member votes with China, Russia and the US 9%, 4% and 1% of the time, below its own chance level (16%, 7%, 3%); 75%, 76% and 88% of members sit below chance (Wilcoxon p < 0.001). The test does find followers where there are some: Israel votes with the US 91% of the time; 8 members follow China and 9 follow Russia above chance after Benjamini–Hochberg correction.
4. **2022 moved the others, not the members.** Median members change by at most 2 points (China 8% → 10%); non-members move more (Russia 24% → 17%). On Ukraine, members switch to abstaining: 33% vs 9% on all other votes.
5. **A country's own past predicts it better than any power.** On 2023–25 votes never seen in training (103 members, balanced accuracy), "same vote as last time on the same text" reaches 69%, the country's history 66%, all three powers 61%, a single power 51–59%, "always yes" 33%.
6. **A few exceptions.** Pakistan follows China (53% vs 41% by chance); South Africa follows no one; Mali moves towards Moscow after the coups (54% in 2023, chance 34%).

### Limitations and future work
- **Associations, not causes.** The data say how countries vote, not why.
- **Most resolutions pass by consensus** and leave no record: the study covers only the votes that were fought over.
- **A UN vote is not an alliance**, and abstaining on Ukraine is neutral in the vote, not necessarily in its effects.
- **141 votes have no issue code**, and "importantvote" is a proxy computed from the data, not the official list.
- **Next steps**: extend the period back to 1946 to see the end of the Cold War; code the votes with no issue; add alliances, aid and trade to explain the exceptions.

---

## How to run it

### Folder contents

| Item | What it is |
|---|---|
| `dataset/un_votes_clean_2014_2025.txt` | Cleaned dataset (CSV format), input of every script. |
| `dataset/un_pipeline.py` | Cleaning of the raw Voeten file: writes the cleaned dataset and `report_pulizia.txt`. |
| `codice/comune.py` | Shared module: paths, the 119 members, vote matrix, agreement with the majority, following in the minority, Cohen's κ, network, colours, `Report` class. |
| `codice/atto1_apertura.py`, `atto1_mappa.py` | Part 0, opening: dataset in numbers, hook (agreement with China and the US), map of the members. |
| `codice/atto2_dati.py` | Part 1, data: structure, cleaning decisions, exploratory analysis. |
| `codice/atto3_livello1_mondo.py` | Part 2, world (Q1): agreement with the majority, PCA, UMAP, k-means and Ward, validation with the Movement, Kruskal–Wallis profiles, network. |
| `codice/atto3_livello2_non_allineati.py` | Part 3, non-aligned (Q2–Q5): majority effect, following in the minority vs chance, before/after 2022, abstention and Ukraine, random forest. |
| `codice/atto3_livello3_casi.py` | Part 4, cases (Q6): country profiles of India, South Africa and Mali. |
| `codice/atto4_dashboard.py` | Data of the interactive dashboard (172 country profiles). |
| `codice/atto5_chiusura.py` | Part 5, answer: key numbers, what the analysis establishes and what it does not. |
| `codice/extra_visualizzazioni.py`, `figura_rete.py`, `figure_aggiuntive.py` | Map of who follows China, typical vote, UMAP map, clusters-to-network bridge, appendix charts. |
| `codice/esegui_tutto.py` | Runs all of the above in order. |
| `codice/figure_slide/` | The charts redrawn for the deck, one script per slide, plus `base.py` and `numeri_slide.py` (numbers quoted on slides 3, 29 and 33). |
| `fonts/` | Inter font used in the charts (SIL Open Font License). |
| `output/` | Created by the scripts: one folder per part with a text report, `risultati.json`, tables and figures. |
| `requirements.txt` | Package versions. |
| `presentation.pptx` | The slides presenting the results. |

### Where each chart of the deck comes from

| Slide | Chart | File |
|---|---|---|
| 1 | 193 dots, 119 members | `figure_slide/slide01_copertina.py` |
| 3 | Agreement with China, histogram | `atto1_apertura.py` → `gancio_accordo_cina.png` |
| 4 | Map of the members | `atto1_mappa.py` → `mappa_membri.png` |
| 10 | Votes per year; how contested | `atto2_dati.py` → `distribuzione_voti.png`, `quanto_sono_contese.png` |
| 11 | Contested issues | `figure_slide/slide11_temi.py` |
| 13 | Agreement with the majority | `atto3_livello1_mondo.py` → `allineamento_maggioranza.png` |
| 14 | PCA | `figure_slide/slide14_pca.py` |
| 15 | UMAP | `figure_aggiuntive.py` → `umap.png` |
| 16 | Choice of k | `atto3_livello1_mondo.py` → `scelta_k.png` |
| 17 | Cluster profiles | `figure_slide/slide17_profili.py` |
| 18 | Network | `figure_slide/slide18_network.py` |
| 19 | Clusters vs network | `figure_aggiuntive.py` → `ponte_cluster_rete.png` |
| 21 | Majority effect | `atto3_livello2_non_allineati.py` → `effetto_maggioranza.png` |
| 22 | Following vs chance | `figure_slide/slide22_follow.py` |
| 23 | Map: who follows China | `extra_visualizzazioni.py` → `mappa_segue_cina.png` |
| 24 | Who follows the US | `figure_slide/slide24_us.py` |
| 25 | Before/after 2022 (numbers) | `figure_slide/slide25_2022.py` |
| 26 | Abstention on Ukraine | `figure_slide/slide26_ukraine.py` |
| 27 | Model | `figure_slide/slide27_model.py` (scores from the `atto3_livello2` report) |
| 29 | Pakistan and South Africa (numbers) | `figure_slide/numeri_slide.py` |
| 30 | Mali year by year | `figure_slide/slide30_mali.py` |
| 33 | A typical vote | `extra_visualizzazioni.py` → `votazione_tipica.png` |

### Steps

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd codice
FIG_EN=1 FIG_SLIDE=1 python esegui_tutto.py      # about 5 minutes
cd figure_slide
for f in *.py; do python "$f"; done              # base.py only defines shared functions
```

Put `un_votes_clean_2014_2025.txt` in `dataset/` (the scripts also find it in a `dataset/` folder next to the project). Run from the `codice` folder, since paths are relative to it. Run `esegui_tutto.py` first: slides 17, 29 and the case studies read its results. `FIG_EN=1` writes the charts in English and `FIG_SLIDE=1` without their title, as in the deck; the charts land in `output/<part>/figure_slide_en/` and `codice/figure_slide/figure/`. All random steps use fixed seeds (42 for the analysis, 0–2 for UMAP, 0–49 for Louvain), so the numbers are reproducible.

To rebuild the cleaned dataset from the raw file, download it from the DOI above and run `python dataset/un_pipeline.py --input data_raw/UNVotes.csv`.
