"""
Traduzione in inglese dei testi delle figure.
Usata da comune.salva() quando la variabile d'ambiente FIG_EN=1:
ogni testo disegnato viene cercato in ESATTI; se non c'è, si applicano le
sostituzioni parziali di PARZIALI (nomi di paesi, parole ricorrenti).
"""
import re

ESATTI = {
    '"Path to peace" (testo USA, feb 2025)': '"Path to peace" (US text, Feb 2025)',
    '"Sempre sì"': '"Always yes"',
    "% dei voti": "% of votes",
    "% delle votazioni in cui la potenza è in minoranza e il paese vota come lei": "% of votes where the power is in the minority and the country votes with it",
    "% di Stati votanti che esprimono il voto più frequente": "% of voting states casting the most common vote",
    "% di astensioni (votazioni 'USA contro tutti' escluse)": "% abstentions ('US against all' votes excluded)",
    "% di sì": "% yes",
    "% di votazioni in cui il paese vota come la Cina": "% of votes in which the country votes like China",
    "% di votazioni in cui lo Stato è assente": "% of votes in which the state is absent",
    "% di voti uguali a quelli della Cina": "% of votes identical to China's",
    "% di voti uguali a quelli della Russia": "% of votes identical to Russia's",
    "% di voti uguali alla Cina": "% of votes identical to China's",
    "% di voti uguali alla maggioranza": "% of votes identical to the majority",
    "% varianza spiegata": "% variance explained",
    "% votazioni": "% of votes",
    "% voti uguali": "% identical votes",
    "15 assenti": "15 absent",
    "176 sì": "176 yes",
    "Accordo con Ward (ARI)": "Agreement with Ward (ARI)",
    "Accordo per anno": "Agreement by year",
    "Accuratezza": "Accuracy",
    "Accuratezza bilanciata (3 classi)": "Balanced accuracy (3 classes)",
    "Accuratezza sul 2023–25 (modello completo)": "Accuracy on 2023–25 (full model)",
    "Aggressione (mar 2022)": "Aggression (Mar 2022)",
    "Altre votazioni": "Other votes",
    "Altri paesi (mediana)": "Other countries (median)",
    "Assente": "Absent",
    "Astenuto": "Abstain",
    "Bambini ucraini (dic 2025)": "Ukrainian children (Dec 2025)",
    "Componente": "Component",
    "Conseguenze umanitarie (mar 2022)": "Humanitarian consequences (Mar 2022)",
    "Crimea": "Crimea",
    "Integrità territoriale (ott 2022)": "Territorial integrity (Oct 2022)",
    "Invasione\ndell'Ucraina": "Invasion\nof Ukraine",
    "India con\nla Cina": "India with\nChina",
    "Maggioranza": "Majority",
    "Membro mediano": "Median member",
    "Modularità Q": "Modularity Q",
    "No": "No",
    "Numero di paesi": "Number of countries",
    "Pace giusta (feb 2023)": "Just peace (Feb 2023)",
    "Pace in Ucraina (feb 2025)": "Peace in Ukraine (Feb 2025)",
    "Per paese": "By country",
    "Profilo per tema": "Profile by issue",
    "Riparazioni (nov 2022)": "Reparations (Nov 2022)",
    "Scree plot": "Scree plot",
    "Segue la potenza quando è in minoranza (linea punteggiata: membro mediano 2014-21)": "Follows the power when it is in the minority (dotted line: median member 2014–21)",
    "Silhouette": "Silhouette",
    "Solo come votano USA,\nCina e Russia quel giorno": "Only how the US, China\nand Russia vote that day",
    "Solo la storia del paese\n(voto precedente sullo stesso testo)": "Only the country's history\n(previous vote on the same text)",
    "Sospensione dal Cons. diritti umani": "Suspension from Human Rights Council",
    "Stabilità bootstrap (ARI mediano)": "Bootstrap stability (median ARI)",
    "Sì": "Yes",
    "Tutto insieme": "Everything together",
    "USA contro (quasi) tutti: ≤ 3 voti contrari": "US against (almost) all: ≤ 3 votes against",
    "Un punto per Stato · Convenzione sui diritti delle persone con disabilità, 17 dicembre 2025": "One dot per state · Convention on the Rights of Persons with Disabilities, 17 December 2025",
    "Votazioni": "Votes",
    "Votazioni registrate": "Recorded votes",
    "altri G20": "other G20",
    "altri paesi": "other countries",
    "k = 4 scelto": "k = 4 chosen",
    "membri del Movimento": "Movement members",
    "membro mediano": "median member",
    "meno del 5%": "under 5%",
    "nessun dato": "no data",
    "ocra = stessa comunità Louvain (soglia κ 0,4); grigio = comunità diversa": "ochre = same Louvain community (κ threshold 0.4); grey = different community",
    "oltre il 40%": "over 40%",
    "periodo": "period",
    "soglia di 'importantvote'": "'importantvote' threshold",
    "votazioni in cui gli USA votano no\ncon al massimo altri 2 paesi": "votes in which the US votes no\nwith at most 2 other countries",
    "z-score per riga": "z-score by row",
    "κ di Cohen": "Cohen's κ",
    "Quando la Cina vota diversamente dalla maggioranza, il paese vota come lei:": "When China votes differently from the majority, the country votes with it:",
    "I temi si sovrappongono:\n372 votazioni ne hanno più di uno.\nBarre = votazioni per anno.": "Issues overlap:\n372 votes have more than one.\nBars = votes per year.",
    "Nastri: coppie di paesi di gruppi diversi con accordo forte (κ di Cohen ≥ 0,4).\nLegami interni ai gruppi (non disegnati): Sud globale 1.748, Europa e alleati 1.122, Cina–Russia 34, USA–Israele 12.":
        "Ribbons: pairs of countries in different groups with strong agreement (Cohen's κ ≥ 0.4).\nTies within groups (not drawn): Global South 1,748, Europe and allies 1,122, China–Russia 34, US–Israel 12.",
    "Senza legami forti: Argentina, Costa Rica": "No strong ties: Argentina, Costa Rica",
    "PRIMA: tutti sembrano vicini alla Cina": "BEFORE: everyone looks close to China",
    "DOPO: è quasi solo l'accordo con la maggioranza": "AFTER: it is almost only agreement with the majority",
    "PC1 (58%) · pesano soprattutto diritti umani, nucleare, disarmo": "PC1 (58%) · driven by human rights, nuclear weapons, disarmament",
    "PC2 (8%) · Medio Oriente": "PC2 (8%) · Middle East",
    "Prevedere il voto di 8 membri del Movimento\n(random forest, addestrata 2015–22, verificata 2023–25)": "Predicting the vote of 8 Movement members\n(random forest, trained 2015–22, tested 2023–25)",
    "Profili dei cluster su votazioni NON usate per costruirli\n(mediane; Kruskal–Wallis, p corretti BH < 0,001)": "Cluster profiles on votes NOT used to build them\n(medians; Kruskal–Wallis, BH-adjusted p < 0.001)",
    "Quante volte succede, anno per anno\n(USA contrari con al massimo altri 2 paesi)": "How often it happens, year by year\n(US against with at most 2 other countries)",
    # analisi aggiunte (idee di Alec)
    "Non membro mediano": "Median non-member",
    "PC1 è nucleare, diritti umani e disarmo; PC2 è il Medio Oriente": "PC1 is nuclear weapons, human rights and disarmament; PC2 is the Middle East",
    "PC1 (58% della varianza)": "PC1 (58% of variance)",
    "Peso medio delle votazioni del tema sulla componente (|loading| × 1000)": "Average weight of the issue’s votes on the component (|loading| × 1000)",
    "Centralità nella rete di accordo, anno per anno": "Centrality in the agreement network, year by year",
    "Centralità (forza normalizzata)": "Centrality (normalised strength)",
    'Regola: "ripeti il voto\nprecedente, altrimenti sì"': 'Rule: “repeat the previous\nvote, otherwise yes”',
    "% delle votazioni in cui il paese vota come la potenza": "% of votes in which the country votes like the power",
    "Ogni membro": "Each member",
    "Sui temi del Sud si astengono gli altri": "On the South’s issues, the others abstain",
    "Sull'Ucraina si astengono i non allineati": "On Ukraine, the non-aligned abstain",
    "% di astensioni (mediana per paese)": "% abstentions (median per country)",
    "% di astensioni sui voti espressi": "% abstentions among votes cast",
    "astensione abituale\ndei membri": "members’ usual\nabstention",
    "Aggressione (mar 22)": "Aggression (Mar 22)",
    "Conseguenze umanitarie (mar 22)": "Humanitarian consequences (Mar 22)",
    "Sospensione dal CDU (apr 22)": "Suspension from HRC (Apr 22)",
    "Integrità territoriale (ott 22)": "Territorial integrity (Oct 22)",
    "Riparazioni (nov 22)": "Reparations (Nov 22)",
    "Pace giusta (feb 23)": "Just peace (Feb 23)",
    "Pace in Ucraina (feb 25)": "Peace in Ukraine (Feb 25)",
    "Testo USA (feb 25)": "US text (Feb 25)",
    "Bambini ucraini (dic 25)": "Ukrainian children (Dec 25)",
    "Ci si astiene quando la lite non è la propria": "You abstain when it is not your fight",
    "Dopo il 2022 i membri cambiano poco: a muoversi sono gli altri": "After 2022 the members barely move: it is the others who shift",
    "Prevedere il voto dei 103 membri del Movimento\n(random forest, addestrata 2015–22, verificata 2023–25)": "Predicting the vote of the 103 Movement members\n(random forest, trained 2015–22, tested 2023–25)",
    "Membro mediano (103)": "Median member (103)",
    "Quando gli USA votano contro la maggioranza: % di volte in cui il paese vota come loro": "When the US votes against the majority: % of times the country votes with it",
    "% delle votazioni con gli USA in minoranza in cui il paese vota come gli USA": "% of votes with the US in the minority in which the country votes like the US",
    "Quando gli USA restano in minoranza, i loro alleati li seguono; i non allineati no": "When the US stands alone, its allies follow it; the non-aligned do not",
    "Isole Marshall": "Marshall Islands",
    "Lente: i membri tra 0 e 10%": "Zoom: members between 0 and 10%",
    "Solo come votano gli USA": "Only how the US votes",
    "Solo come vota la Cina": "Only how China votes",
    "Solo come vota la Russia": "Only how Russia votes",
    "Come votano tutte e tre": "How all three vote",
    "La storia del paese": "The country’s own history",
    'Regola: "come l\'ultima volta"': "Rule: “same as last time”",
    "Accuratezza bilanciata: media dei successi su sì, astenuto e no": "Balanced accuracy: average success on yes, abstain and no",
    "Cosa prevede meglio il voto dei 103 membri?\n(random forest, addestrata 2015–22, verificata 2023–25)": "What best predicts the vote of the 103 members?\n(random forest, trained 2015–22, tested 2023–25)",
    "Cina:\nuno dei 176": "China:\none of the 176",
    "La Cina non guida la folla: ne fa parte": "China does not lead the crowd: it is part of it",
}

GRUPPI = [
    ("Maggioranza del Sud globale", "Global South majority"),
    ("USA, Israele e alleati stretti", "US, Israel and close allies"),
    ("Occidente, USA e Israele", "West, US and Israel"),
    ("Cina, Russia, India, Iran…", "China, Russia, India, Iran…"),
    ("America Latina e Pacifico", "Latin America and Pacific"),
    ("Blocco Cina–Russia", "China–Russia bloc"),
    ("Europa e alleati", "Europe and allies"),
    ("Sud globale", "Global South"),
    ("USA–Israele", "US–Israel"),
    ("USA e Israele", "US and Israel"),
    ("Membri del Movimento", "Movement members"),
    ("Altri paesi", "Other countries"),
    ("Tutte le votazioni", "All votes"),
    ("Nessun tema", "No issue"),
]
TEMI = [("Armi nucleari", "Nuclear weapons"), ("Diritti umani", "Human rights"), ("Sviluppo economico", "Economic development"),
        ("Medio Oriente", "Middle East"), ("Colonialismo", "Colonialism"), ("Disarmo", "Disarmament")]
PAESI = [("Francia", "France"), ("Rep. Centrafricana", "Central African Rep."), ("Arabia Saudita", "Saudi Arabia"), ("Corea del Nord", "North Korea"), ("Regno Unito", "United Kingdom"),
         ("United Republic Of Tanzania", "Tanzania"), ("Viet Nam", "Vietnam"), ("Bielorussia", "Belarus"),
         ("Sudafrica", "South Africa"), ("Brasile", "Brazil"), ("Egitto", "Egypt"), ("Giappone", "Japan"),
         ("Israele", "Israel"), ("Malesia", "Malaysia"), ("Messico", "Mexico"), ("Siria", "Syria"),
         ("Turchia", "Türkiye"), ("Ucraina", "Ukraine"), ("Cina", "China"), ("USA", "US")]
PAROLE = [("(accuratezza ", "(accuracy "), (" membri su ", " members out of "), (" sotto il 2%; il più alto, ", " below 2%; the highest, "), ("il più alto: ", "highest: "), ("Non membri\n(", "Non-members\n("), ("Membri del\nMovimento (", "Movement\nmembers ("), ("Israele ", "Israel "), (" paesi)", " countries)"), ("I 5 paesi che votano in modo più simile · cluster: ", "The 5 countries that vote most alike · cluster: "),
          ("Accordo con", "Agreement with"), (" Stati", " states"), ("Sì · ", "Yes · "), ("Astenuto · ", "Abstain · "),
          ("senza colori", "no colours"), (" in minoranza", " in minority")]

# Titoli (compaiono solo nelle figure con titolo; le slide usano figure_slide senza titolo)
TITOLI = {
    "Il 2025 non è un errore: 114 votazioni chieste dagli USA su testi prima adottati per consenso": "2025 is not an error: 114 votes on texts previously adopted by consensus",
    "Tre voti su quattro sono sì: da qui nasce l'effetto maggioranza": "Three votes out of four are yes: the source of the majority effect",
}
TITOLI.update({
    "114 su 192": "114 of 192",
    "51 paesi su 192\nsopra l'80%": "51 of 192 countries\nabove 80%",
    "La divisione in blocchi è stabile anno per anno; l'India sta quasi sempre con la Cina": "The bloc structure is stable year after year; India is almost always with China",
    "PCA: una frattura principale, Occidente contro il resto": "PCA: one main divide, the West against the rest",
    "Tra i blocchi ci sono solo due ponti": "Only two bridges between the blocs",
    "paese mediano: 78%": "median country: 78%",
})
ESATTI.update(TITOLI)


def traduci(t):
    if t in ESATTI:
        return ESATTI[t]
    s = t
    for a, b in GRUPPI + TEMI + PAROLE + PAESI:
        s = s.replace(a, b)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)                 # 0,4 -> 0.4
    s = s.replace("mediana", "median").replace("membri", "members").replace("altri", "others")
    return s


FONTE_EN = "Source: Voeten et al. / UN Digital Library, recorded UN General Assembly votes 2014–2025."


# ---------------------------------------------------------------- versione finale: testi nuovi
ESATTI.update({
    "Ogni Stato con ogni Stato (172 × 172)": "Every state with every state (172 × 172)",
    "κ di Cohen": "Cohen’s κ",
    "0,4 soglia": "0.4 threshold",
    "Righe e colonne: gli Stati, ordinati per comunità (Louvain) e, dentro ogni comunità, per PC1.\n"
    "Blocchi scuri sulla diagonale = comunità; macchie fuori dalla diagonale = legami tra comunità.":
        "Rows and columns: the states, ordered by community (Louvain) and, within each community, by PC1.\n"
        "Dark blocks on the diagonal = communities; marks off the diagonal = ties between communities.",
    "Numero nel nodo = Stati · numero sull'arco = coppie di Stati\ncon κ ≥ 0,4 tra le due comunità (Occidente–Sud globale: nessuna)":
        "Number in the node = states · number on the edge = pairs of states\nwith κ ≥ 0.4 across the two communities (West–Global South: none)",
    "Stabilità bootstrap\n(ARI mediano)": "Bootstrap stability\n(median ARI)",
    "Accordo con Ward\n(ARI)": "Agreement with Ward\n(ARI)",
    "k = 4 scelto": "k = 4 chosen",
    "k (numero di gruppi)": "k (number of groups)",
    "cerchio vuoto: atteso per caso": "hollow circle: expected by chance",
    "Ucraina (9 votazioni)": "Ukraine (9 votes)",
    "Sui temi del Sud si astengono gli altri; sull'Ucraina i membri": "On the South’s issues the others abstain; on Ukraine, the members",
    "Le 9 votazioni sull'Ucraina, una per una": "The 9 Ukraine votes, one by one",
    "% di astensioni (mediana per paese)": "% abstentions (median per country)",
    "% di astensioni sui voti espressi": "% abstentions among votes cast",
    "Accuratezza bilanciata sul 2023–25 (modello completo)": "Balanced accuracy on 2023–25 (full model)",
    "La Cina non guida la folla: ne fa parte": "China does not lead the crowd: it is part of it",
})
PAROLE[:0] = [("senza legami forti (", "no strong ties ("), ("La stessa rete, un nodo per comunità", "The same network, one node per community"),
              ("legami interni: ", "internal ties: "), ("il prezzo: accordo con Ward", "the cost: agreement with Ward"),
              ("astensione abituale\ndei membri", "members’ usual\nabstention"), ("Cina:\nuno dei ", "China:\none of the "),
              (" astenuti · ", " abstain · "), (" assenti", " absent"),
              ("Un punto per Stato · Insediamenti israeliani nei Territori palestinesi occupati, ", "One dot per state · Israeli settlements in the Occupied Palestinian Territory, "),
              (" dicembre ", " December "), ("Votazione tipica: il voto più frequente raccoglie il ", "A typical vote: the most common vote gets "),
              ("% dei votanti (mediana di tutte le votazioni: ", "% of those voting (median of all votes: ")]
PAROLE.append((" sì", " yes"))
ESATTI.update({"Per tema e sull'Ucraina": "By issue and on Ukraine", "Stabilità bootstrap\n(ARI mediano)": "Bootstrap stability\n(median ARI)",
               "Accordo con Ward (ARI)": "Agreement with Ward (ARI)", "Silhouette": "Silhouette"})
PAROLE[:0] = [("il prezzo: accordo\ncon Ward", "the cost: agreement\nwith Ward")]
ESATTI.update({"Segue la potenza in minoranza\n(punteggiato: membro mediano 2014-21)": "Follows the power in the minority\n(dotted: median member 2014–21)"})
ESATTI.update({
    "La rete: ogni punto è uno Stato, colorato per comunità": "The network: each dot is a state, coloured by community",
    "legame tra comunità diverse (ponte)": "tie between different communities (a bridge)",
    "legame dentro la stessa comunità": "tie within the same community",
    "In sintesi: un cerchio per comunità": "In short: one circle per community",
    "numero sulla linea = coppie di Stati con accordo forte tra i due gruppi": "number on the line = pairs of states with strong agreement across the two groups",
})
PAROLE[:0] = [(" Stati senza legami forti (non disegnati)", " states with no strong ties (not drawn)"),
              ("Arco se κ di Cohen ≥ ", "Edge if Cohen’s κ ≥ "), (" (accordo corretto per il caso) · comunità con Louvain, Q = ", " (agreement corrected for chance) · communities by Louvain, Q = "),
              (" · i gruppi sono disposti separati apposta; le distanze non misurano nulla: contano i gruppi e le linee.", " · the groups are placed apart on purpose; distances measure nothing: what matters is the groups and the lines.")]
