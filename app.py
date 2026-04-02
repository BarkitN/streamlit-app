
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MAX PRO New Grid Project V2", layout="wide")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR

SOURCE_FILES = {
    "BeSoccer": DATA_DIR / "besoccer_newgrid.csv",
    "Bookmaker": DATA_DIR / "bookmaker_newgrid.csv",
    "Opta": DATA_DIR / "opta_newgrid.csv",
    "Windrawwin": DATA_DIR / "windrawwin_newgrid.csv",
    "Forebet": DATA_DIR / "forebet_newgrid.csv",
    "EduardoLosilla": DATA_DIR / "eduardolosilla_newgrid.csv",
    "WebPronostici": DATA_DIR / "webpronostici_newgrid.csv",
    "SoccerVista": DATA_DIR / "soccervista_newgrid.csv",
    "WhoScored": DATA_DIR / "whoscored_newgrid.csv",
    "TicketLocal": DATA_DIR / "ticketlocal_newgrid.csv",
}

TEAM_MAPS = {
    "Spain": {"Atletico Madrid":"Ath Madrid","Real Mallorca":"Mallorca","Real Betis":"Betis","Espanyol":"Espanol","Barcelona":"Barcelona","Real Madrid":"Real Madrid"},
    "Germany": {"Stuttgart":"Stuttgart","Borussia Dortmund":"Dortmund","Freiburg":"Freiburg","Bayern Munich":"Bayern Munich","Werder Bremen":"Werder Bremen","RB Leipzig":"RB Leipzig","Hamburg":"Hamburger SV","Augsburg":"Augsburg","Hoffenheim":"Hoffenheim","Mainz":"Mainz"},
    "France": {"Stade Rennais":"Rennes","Strasbourg":"Strasbourg","Nice":"Nice","Brest":"Brest","Lille":"Lille","Lens":"Lens"},
    "Italy": {"Lazio":"Lazio","Parma":"Parma","Verona":"Verona","Fiorentina":"Fiorentina"},
}

@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data(show_spinner=False)
def load_inputs():
    web = read_csv_safe(DATA_DIR / "web_database_import.csv")
    weights = read_csv_safe(DATA_DIR / "recalibrated_weights.csv")
    sources = {name: read_csv_safe(path) for name, path in SOURCE_FILES.items()}
    hist = {
        "Spain": read_csv_safe(DATA_DIR / "SP1.csv"),
        "Germany": read_csv_safe(DATA_DIR / "D1.csv"),
        "France": read_csv_safe(DATA_DIR / "F1.csv"),
        "Italy": read_csv_safe(DATA_DIR / "I1.csv"),
        "ALL": read_csv_safe(DATA_DIR / "ALL_HISTORICAL_LEAGUES.csv"),
    }
    return web, weights, sources, hist

def save_source(name: str, df: pd.DataFrame):
    df.to_csv(SOURCE_FILES[name], index=False, encoding="utf-8-sig")
    st.cache_data.clear()

def safe_float(v):
    try:
        if v is None or str(v).strip() == "":
            return float("nan")
        return float(str(v).replace("%","").replace(",","."))
    except Exception:
        return float("nan")

def derive_pick(row):
    pick = str(row.get("Pick","")).strip().upper()
    if pick in {"1","X","2","1X","12","X2","1X2"}:
        return pick
    dbl = str(row.get("Double","")).strip().upper()
    p1, px, p2 = safe_float(row.get("Prob_1")), safe_float(row.get("Prob_X")), safe_float(row.get("Prob_2"))
    vals = pd.Series({"1":p1,"X":px,"2":p2}).dropna().sort_values(ascending=False)
    if len(vals) == 0:
        return dbl if dbl in {"1X","12","X2"} else ""
    if len(vals) == 1:
        return vals.index[0]
    if vals.iloc[0] - vals.iloc[1] >= 8:
        return vals.index[0]
    return "".join(sorted(vals.index[:2].tolist(), key=lambda x: {"1":0,"X":1,"2":2}[x]))

def map_team_guess(df, team, league):
    mapped = TEAM_MAPS.get(league, {}).get(team, team)
    universe = set(df.get("HomeTeam", pd.Series(dtype=str))).union(set(df.get("AwayTeam", pd.Series(dtype=str))))
    if mapped in universe:
        return mapped
    base = str(team).lower().replace("stade ", "").replace("real ", "").replace("fc ", "")
    for cand in sorted(universe):
        cn = str(cand).lower().replace("stade ", "").replace("real ", "").replace("fc ", "")
        if base in cn or cn in base:
            return cand
    return mapped

def league_table(df):
    if df.empty:
        return pd.DataFrame()
    teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
    rows = []
    for team in teams:
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]
        gp = len(home) + len(away)
        w = int((home["FTR"] == "H").sum() + (away["FTR"] == "A").sum())
        d = int((home["FTR"] == "D").sum() + (away["FTR"] == "D").sum())
        l = gp - w - d
        gf = int(home["FTHG"].sum() + away["FTAG"].sum())
        ga = int(home["FTAG"].sum() + away["FTHG"].sum())
        pts = 3*w + d
        rows.append({"Team":team,"GP":gp,"W":w,"D":d,"L":l,"GF":gf,"GA":ga,"GD":gf-ga,"Pts":pts})
    tab = pd.DataFrame(rows).sort_values(["Pts","GD","GF"], ascending=False).reset_index(drop=True)
    tab["Rank"] = range(1, len(tab)+1)
    return tab[["Rank","Team","GP","W","D","L","GF","GA","GD","Pts"]]

def recent_form(df, team, n=5, venue=None):
    if df.empty:
        return {"Pts":None,"Form":"","GF":None,"GA":None}
    if venue == "home":
        matches = df[df["HomeTeam"] == team].tail(n)
    elif venue == "away":
        matches = df[df["AwayTeam"] == team].tail(n)
    else:
        matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if matches.empty:
        return {"Pts":None,"Form":"","GF":None,"GA":None}
    pts = gf = ga = 0
    form = []
    for _, r in matches.iterrows():
        if r["HomeTeam"] == team:
            gf += int(r["FTHG"]); ga += int(r["FTAG"])
            if r["FTR"] == "H": pts += 3; form.append("W")
            elif r["FTR"] == "D": pts += 1; form.append("D")
            else: form.append("L")
        else:
            gf += int(r["FTAG"]); ga += int(r["FTHG"])
            if r["FTR"] == "A": pts += 3; form.append("W")
            elif r["FTR"] == "D": pts += 1; form.append("D")
            else: form.append("L")
    return {"Pts":pts,"Form":"".join(form),"GF":gf,"GA":ga}

def h2h_summary(df, home, away, league):
    if df.empty:
        return {"Games":0,"HomeWins":0,"Draws":0,"AwayWins":0,"Trail":""}
    home_m = map_team_guess(df, home, league)
    away_m = map_team_guess(df, away, league)
    subset = df[((df["HomeTeam"] == home_m) & (df["AwayTeam"] == away_m)) | ((df["HomeTeam"] == away_m) & (df["AwayTeam"] == home_m))]
    if subset.empty:
        return {"Games":0,"HomeWins":0,"Draws":0,"AwayWins":0,"Trail":""}
    hw = dr = aw = 0
    trail = []
    for _, r in subset.tail(5).iterrows():
        if r["FTR"] == "D":
            dr += 1; trail.append("D")
        elif (r["HomeTeam"] == home_m and r["FTR"] == "H") or (r["AwayTeam"] == home_m and r["FTR"] == "A"):
            hw += 1; trail.append("H")
        else:
            aw += 1; trail.append("A")
    return {"Games":len(subset),"HomeWins":hw,"Draws":dr,"AwayWins":aw,"Trail":"".join(trail)}

def historical_analysis(web_df, hist_dict):
    league_tables = {lg: league_table(df) for lg, df in hist_dict.items() if lg != "ALL"}
    rows = []
    for _, r in web_df.iterrows():
        lg = r["League"]
        hist = hist_dict.get(lg, pd.DataFrame())
        if hist.empty:
            rows.append({"N°":int(r["N°"]),"Match":r["Match"],"League":lg,"Historical_Status":"Historique indisponible","Hist_Pick":"","Hist_Double":"","Hist_Confidence":None})
            continue
        home = map_team_guess(hist, r["Equipe1"], lg)
        away = map_team_guess(hist, r["Equipe2"], lg)
        tab = league_tables[lg]
        hr = tab.loc[tab["Team"] == home]
        ar = tab.loc[tab["Team"] == away]
        hrank = int(hr.iloc[0]["Rank"]) if not hr.empty else None
        arank = int(ar.iloc[0]["Rank"]) if not ar.empty else None
        hpts = int(hr.iloc[0]["Pts"]) if not hr.empty else 0
        apts = int(ar.iloc[0]["Pts"]) if not ar.empty else 0
        hgd = int(hr.iloc[0]["GD"]) if not hr.empty else 0
        agd = int(ar.iloc[0]["GD"]) if not ar.empty else 0
        rf_home = recent_form(hist, home, 5)
        rf_away = recent_form(hist, away, 5)
        hf_home = recent_form(hist, home, 5, "home")
        af_away = recent_form(hist, away, 5, "away")
        h2h = h2h_summary(hist, r["Equipe1"], r["Equipe2"], lg)

        sh = sa = 0.0
        if hrank and arank:
            sh += (arank - hrank) * 0.16
            sa += (hrank - arank) * 0.16
        sh += (hpts - apts) * 0.04; sa += (apts - hpts) * 0.04
        sh += ((rf_home.get("Pts") or 0) - (rf_away.get("Pts") or 0)) * 0.10
        sa += ((rf_away.get("Pts") or 0) - (rf_home.get("Pts") or 0)) * 0.10
        sh += ((hf_home.get("Pts") or 0) - (af_away.get("Pts") or 0)) * 0.12
        sa += ((af_away.get("Pts") or 0) - (hf_home.get("Pts") or 0)) * 0.12
        sh += (hgd - agd) * 0.02; sa += (agd - hgd) * 0.02
        sh += (h2h["HomeWins"] - h2h["AwayWins"]) * 0.12
        sa += (h2h["AwayWins"] - h2h["HomeWins"]) * 0.12
        diff = sh - sa

        if abs(diff) >= 0.85:
            pick = "1" if diff > 0 else "2"; dbl = pick; conf = round(min(0.80, 0.52 + abs(diff)/4), 3)
        elif abs(diff) >= 0.35:
            pick = "1" if diff > 0 else "2"; dbl = "1X" if diff > 0 else "X2"; conf = round(min(0.70, 0.44 + abs(diff)/5), 3)
        else:
            pick = "X"; dbl = "1X" if diff >= 0 else "X2"; conf = round(0.38 + (0.35-abs(diff))/3, 3)

        rows.append({
            "N°":int(r["N°"]), "Match":r["Match"], "League":lg, "Historical_Status":"OK",
            "Hist_Pick":pick, "Hist_Double":dbl, "Hist_Confidence":conf,
            "Home_Rank":hrank, "Away_Rank":arank, "Home_Pts":hpts, "Away_Pts":apts,
            "Home_Form5":rf_home.get("Form",""), "Away_Form5":rf_away.get("Form",""),
            "Home_Home5":hf_home.get("Form",""), "Away_Away5":af_away.get("Form",""),
            "H2H":f"{h2h['HomeWins']}-{h2h['Draws']}-{h2h['AwayWins']} | {h2h['Trail']}"
        })
    return pd.DataFrame(rows), league_tables

def compute_consensus(web_df, weights_df, source_frames, hist_df):
    weights = dict(zip(weights_df["Source"], weights_df["Weight"]))
    hist_lookup = hist_df.set_index("N°").to_dict("index")
    rows = []
    for _, g in web_df.iterrows():
        n = int(g["N°"])
        votes = {"1":0.0,"X":0.0,"2":0.0}
        traces = []
        for src, df in source_frames.items():
            row = df.loc[df["N°"] == n]
            if row.empty: continue
            pred = derive_pick(row.iloc[0]); w = float(weights.get(src, 1.0))
            if pred in {"1","X","2"}:
                votes[pred] += w; traces.append(f"{src}:{pred}@{w:.2f}")
            elif pred in {"1X","12","X2"}:
                for ch in pred: votes[ch] += w*0.55
                traces.append(f"{src}:{pred}@{w:.2f}")
            elif pred == "1X2":
                for ch in ["1","X","2"]: votes[ch] += w*0.34
                traces.append(f"{src}:{pred}@{w:.2f}")
        h = hist_lookup.get(n, {})
        hw = float(weights.get("Historical", 1.0))
        hp = str(h.get("Hist_Pick","")); hd = str(h.get("Hist_Double",""))
        if hp in {"1","X","2"}:
            votes[hp] += hw; traces.append(f"Historical:{hp}@{hw:.2f}")
        if hd in {"1X","12","X2"} and hd != hp:
            for ch in hd: votes[ch] += hw*0.25
            traces.append(f"HistoricalCover:{hd}@{hw:.2f}")

        ordered = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        total = sum(votes.values()); top, second = ordered[0], ordered[1]
        confidence = top[1]/total if total else 0
        gap = top[1]-second[1]
        if total == 0:
            ticket=""; level="No signal"; consensus=""
        elif confidence >= 0.52 and gap >= 0.80:
            ticket=top[0]; level="BASE"; consensus=top[0]
        elif confidence >= 0.40:
            ticket="".join(sorted({top[0], second[0]}, key=lambda x: {"1":0,"X":1,"2":2}[x])); level="DOUBLE"; consensus=top[0]
        else:
            ticket="1X2"; level="TRIPLE"; consensus="".join(sorted({k for k,v in votes.items() if v > 0}, key=lambda x: {"1":0,"X":1,"2":2}[x]))
        rows.append({
            "N°":n, "Match":g["Match"], "League":g["League"], "Vote_1":round(votes["1"],2), "Vote_X":round(votes["X"],2), "Vote_2":round(votes["2"],2),
            "Consensus":consensus, "Ticket":ticket, "Level":level, "Confidence":round(confidence,3),
            "Historical_Pick":h.get("Hist_Pick",""), "Historical_Double":h.get("Hist_Double",""), "Trace":" | ".join(traces)
        })
    return pd.DataFrame(rows)

def to_excel_bytes(sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    output.seek(0)
    return output.getvalue()

web_df, weights_df, source_frames, hist_dict = load_inputs()
hist_df, league_tables = historical_analysis(web_df, hist_dict)
consensus_df = compute_consensus(web_df, weights_df, source_frames, hist_df)

st.markdown("""
<style>
.block-container {padding-top:1rem; max-width:96rem;}
.hero {padding:16px 20px; border-radius:18px; color:white; background:linear-gradient(135deg,#0f172a,#1d4ed8 85%); margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>MAX PRO - Nouvelle grille régénérée V2</h1><p>Tous les historiques joints sont intégrés dans la base CSV et les anciens onglets de la version MAX PRO sont conservés.</p></div>', unsafe_allow_html=True)

tabs = st.tabs(["WEB DB import","Historiques ligues","Historique profond","Saisie sources","Consensus MAX PRO","Ticket final","Export Excel"])

with tabs[0]:
    st.subheader("Données importées depuis WEB_DATABASE.xlsx")
    st.dataframe(web_df, use_container_width=True, height=520, hide_index=True)

with tabs[1]:
    st.subheader("Tous les historiques joints")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Espagne", f"{len(hist_dict['Spain'])} lignes")
    c2.metric("Allemagne", f"{len(hist_dict['Germany'])} lignes")
    c3.metric("France", f"{len(hist_dict['France'])} lignes")
    c4.metric("Italie", f"{len(hist_dict['Italy'])} lignes")
    lg = st.selectbox("Afficher l'historique d'une league", ["ALL","Spain","Germany","France","Italy"])
    st.dataframe(hist_dict[lg], use_container_width=True, height=420, hide_index=True)

with tabs[2]:
    st.subheader("Analyse historique profonde par match")
    st.dataframe(hist_df, use_container_width=True, height=520, hide_index=True)
    st.markdown("### Classements dérivés")
    lg2 = st.selectbox("League table", ["Spain","Germany","France","Italy"])
    st.dataframe(league_tables[lg2], use_container_width=True, height=380, hide_index=True)

with tabs[3]:
    st.subheader("Édition des bases CSV alimentées par le fichier Excel")
    src = st.selectbox("Choisir la source", list(SOURCE_FILES.keys()))
    edited = st.data_editor(source_frames[src], use_container_width=True, height=560, num_rows="fixed")
    c1, c2 = st.columns(2)
    if c1.button(f"Enregistrer {src}"):
        save_source(src, edited); st.success(f"{src} enregistré."); st.rerun()
    if c2.button("Recharger les données"):
        st.cache_data.clear(); st.rerun()

with tabs[4]:
    st.subheader("Consensus recalibré enrichi de l'historique")
    st.dataframe(consensus_df, use_container_width=True, height=560, hide_index=True)

with tabs[5]:
    st.subheader("Ticket final MAX PRO")
    show = consensus_df[["N°","Match","League","Ticket","Level","Confidence","Historical_Pick","Historical_Double"]].copy()
    show["Recommendation"] = show.apply(lambda r: "Jouer en base" if r["Level"]=="BASE" else (f"Couvrir en {r['Ticket']}" if r["Level"]=="DOUBLE" else "Match très piégeux"), axis=1)
    st.dataframe(show, use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Télécharger le projet complet")
    payload = {
        "WEB_DB_Import": web_df,
        "Weights": weights_df,
        "Historical_All": hist_dict["ALL"],
        "Historical_Analysis": hist_df,
        "Consensus": consensus_df,
        "Final_Ticket": show,
        "Spain_Table": league_tables.get("Spain", pd.DataFrame()),
        "Germany_Table": league_tables.get("Germany", pd.DataFrame()),
        "France_Table": league_tables.get("France", pd.DataFrame()),
        "Italy_Table": league_tables.get("Italy", pd.DataFrame()),
    }
    for src, df in source_frames.items():
        payload[src] = df
    st.download_button("Télécharger l'Excel MAX PRO V2", data=to_excel_bytes(payload), file_name="MAX_PRO_NEW_GRID_PROJECT_V2.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
