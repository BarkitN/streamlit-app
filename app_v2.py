from __future__ import annotations
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MAX PRO V3 AI PREDICTOR FULL ELITE", layout="wide")
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
    "Spain": {
        "Atletico Madrid": "Ath Madrid",
        "Real Mallorca": "Mallorca",
        "Real Betis": "Betis",
        "Espanyol": "Espanol",
        "Barcelona": "Barcelona",
        "Real Madrid": "Real Madrid",
    },
    "Germany": {
        "Stuttgart": "Stuttgart",
        "Borussia Dortmund": "Dortmund",
        "Freiburg": "Freiburg",
        "Bayern Munich": "Bayern Munich",
        "Werder Bremen": "Werder Bremen",
        "RB Leipzig": "RB Leipzig",
        "Hamburg": "Hamburger SV",
        "Augsburg": "Augsburg",
        "Hoffenheim": "Hoffenheim",
        "Mainz": "Mainz",
    },
    "France": {
        "Stade Rennais": "Rennes",
        "Strasbourg": "Strasbourg",
        "Nice": "Nice",
        "Brest": "Brest",
        "Lille": "Lille",
        "Lens": "Lens",
    },
    "Italy": {
        "Lazio": "Lazio",
        "Parma": "Parma",
        "Verona": "Verona",
        "Fiorentina": "Fiorentina",
    },
}

LEAGUE_ORDER = ["Spain", "Germany", "France", "Italy"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def safe_int(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        return int(float(str(v).replace(",", ".")))
    except Exception:
        return default

def safe_float(v):
    try:
        if v is None or str(v).strip() == "":
            return float("nan")
        return float(str(v).replace("%", "").replace(",", "."))
    except Exception:
        return float("nan")

def ordered_combo(symbols):
    symbols = [s for s in symbols if s in {"1", "X", "2"}]
    return "".join(sorted(set(symbols), key=lambda x: {"1": 0, "X": 1, "2": 2}[x]))

def guess_league_from_team(team: str) -> str:
    t = str(team).strip()
    for lg, mapping in TEAM_MAPS.items():
        if t in mapping:
            return lg
    return "ALL"

def classify_risk(confidence: float, gap: float, historical_ok: bool) -> str:
    if confidence >= 0.72 and gap >= 1.20 and historical_ok:
        return "LOW"
    if confidence >= 0.50 and gap >= 0.55:
        return "MEDIUM"
    return "HIGH"

@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1:
                return normalize_columns(df)
        except Exception:
            pass
    return pd.DataFrame()

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

def enrich_web_df(web_df: pd.DataFrame) -> pd.DataFrame:
    web_df = normalize_columns(web_df)
    if web_df is None or web_df.empty:
        return pd.DataFrame()
    if "Match" not in web_df.columns and {"Equipe1", "Equipe2"}.issubset(web_df.columns):
        web_df["Match"] = web_df["Equipe1"].astype(str) + " vs " + web_df["Equipe2"].astype(str)
    if "League" not in web_df.columns and "Equipe1" in web_df.columns:
        web_df["League"] = web_df["Equipe1"].apply(guess_league_from_team)
    if "N°" not in web_df.columns:
        alt_n = find_first_column(web_df, ["No", "N", "Match_No", "Match No", "match_no"])
        if alt_n:
            web_df["N°"] = web_df[alt_n]
        else:
            web_df["N°"] = range(1, len(web_df) + 1)
    return web_df

def save_source(name: str, df: pd.DataFrame):
    df.to_csv(SOURCE_FILES[name], index=False, encoding="utf-8-sig", sep=";")
    st.cache_data.clear()

def derive_pick(row):
    pick = str(row.get("Pick", "")).strip().upper()
    if pick in {"1", "X", "2", "1X", "12", "X2", "1X2"}:
        return pick
    dbl = str(row.get("Double", "")).strip().upper()
    p1 = safe_float(row.get("Prob_1"))
    px = safe_float(row.get("Prob_X"))
    p2 = safe_float(row.get("Prob_2"))
    vals = pd.Series({"1": p1, "X": px, "2": p2}).dropna().sort_values(ascending=False)
    if len(vals) == 0:
        return dbl if dbl in {"1X", "12", "X2"} else ""
    if len(vals) == 1:
        return vals.index[0]
    if vals.iloc[0] - vals.iloc[1] >= 8:
        return vals.index[0]
    return ordered_combo(vals.index[:2].tolist())

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
    df = normalize_columns(df)
    required = {"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
    if df.empty or not required.issubset(df.columns):
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
        pts = 3 * w + d
        rows.append({"Team": team, "GP": gp, "W": w, "D": d, "L": l, "GF": gf, "GA": ga, "GD": gf - ga, "Pts": pts})
    tab = pd.DataFrame(rows).sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    tab["Rank"] = range(1, len(tab) + 1)
    return tab[["Rank", "Team", "GP", "W", "D", "L", "GF", "GA", "GD", "Pts"]]

def recent_form(df, team, n=5, venue=None):
    df = normalize_columns(df)
    required = {"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
    if df.empty or not required.issubset(df.columns):
        return {"Pts": None, "Form": "", "GF": None, "GA": None}
    if venue == "home":
        matches = df[df["HomeTeam"] == team].tail(n)
    elif venue == "away":
        matches = df[df["AwayTeam"] == team].tail(n)
    else:
        matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(n)
    if matches.empty:
        return {"Pts": None, "Form": "", "GF": None, "GA": None}
    pts = gf = ga = 0
    form = []
    for _, r in matches.iterrows():
        if r["HomeTeam"] == team:
            gf += int(r["FTHG"]); ga += int(r["FTAG"])
            if r["FTR"] == "H":
                pts += 3; form.append("W")
            elif r["FTR"] == "D":
                pts += 1; form.append("D")
            else:
                form.append("L")
        else:
            gf += int(r["FTAG"]); ga += int(r["FTHG"])
            if r["FTR"] == "A":
                pts += 3; form.append("W")
            elif r["FTR"] == "D":
                pts += 1; form.append("D")
            else:
                form.append("L")
    return {"Pts": pts, "Form": "".join(form), "GF": gf, "GA": ga}

def h2h_summary(df, home, away, league):
    df = normalize_columns(df)
    if df.empty or not {"HomeTeam", "AwayTeam", "FTR"}.issubset(df.columns):
        return {"Games": 0, "HomeWins": 0, "Draws": 0, "AwayWins": 0, "Trail": ""}
    home_m = map_team_guess(df, home, league)
    away_m = map_team_guess(df, away, league)
    subset = df[((df["HomeTeam"] == home_m) & (df["AwayTeam"] == away_m)) | ((df["HomeTeam"] == away_m) & (df["AwayTeam"] == home_m))]
    if subset.empty:
        return {"Games": 0, "HomeWins": 0, "Draws": 0, "AwayWins": 0, "Trail": ""}
    hw = dr = aw = 0
    trail = []
    for _, r in subset.tail(5).iterrows():
        if r["FTR"] == "D":
            dr += 1; trail.append("D")
        elif (r["HomeTeam"] == home_m and r["FTR"] == "H") or (r["AwayTeam"] == home_m and r["FTR"] == "A"):
            hw += 1; trail.append("H")
        else:
            aw += 1; trail.append("A")
    return {"Games": len(subset), "HomeWins": hw, "Draws": dr, "AwayWins": aw, "Trail": "".join(trail)}

def historical_analysis(web_df, hist_dict):
    web_df = enrich_web_df(web_df)
    home_col = find_first_column(web_df, ["Equipe1", "Home", "HomeTeam"])
    away_col = find_first_column(web_df, ["Equipe2", "Away", "AwayTeam"])
    if web_df.empty or not all(["N°", "Match", "League", home_col, away_col]):
        return pd.DataFrame(), {lg: pd.DataFrame() for lg in LEAGUE_ORDER}
    league_tables = {lg: league_table(df) for lg, df in hist_dict.items() if lg != "ALL"}
    rows = []
    for _, r in web_df.iterrows():
        lg = r["League"]
        hist = hist_dict.get(lg, pd.DataFrame())
        if hist.empty:
            rows.append({"N°": safe_int(r["N°"], None), "Match": r["Match"], "League": lg, "Historical_Status": "Historique indisponible", "Hist_Pick": "", "Hist_Double": "", "Hist_Confidence": None})
            continue
        home = map_team_guess(hist, r[home_col], lg)
        away = map_team_guess(hist, r[away_col], lg)
        tab = league_tables.get(lg, pd.DataFrame())
        hr = tab.loc[tab["Team"] == home] if not tab.empty else pd.DataFrame()
        ar = tab.loc[tab["Team"] == away] if not tab.empty else pd.DataFrame()
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
        h2h = h2h_summary(hist, r[home_col], r[away_col], lg)
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
            pick = "1" if diff > 0 else "2"; dbl = pick; conf = round(min(0.80, 0.52 + abs(diff) / 4), 3)
        elif abs(diff) >= 0.35:
            pick = "1" if diff > 0 else "2"; dbl = "1X" if diff > 0 else "X2"; conf = round(min(0.70, 0.44 + abs(diff) / 5), 3)
        else:
            pick = "X"; dbl = "1X" if diff >= 0 else "X2"; conf = round(0.38 + (0.35 - abs(diff)) / 3, 3)
        rows.append({
            "N°": safe_int(r["N°"], None), "Match": r["Match"], "League": lg, "Historical_Status": "OK",
            "Hist_Pick": pick, "Hist_Double": dbl, "Hist_Confidence": conf,
            "Home_Rank": hrank, "Away_Rank": arank, "Home_Pts": hpts, "Away_Pts": apts,
            "Home_Form5": rf_home.get("Form", ""), "Away_Form5": rf_away.get("Form", ""),
            "Home_Home5": hf_home.get("Form", ""), "Away_Away5": af_away.get("Form", ""),
            "H2H": f"{h2h['HomeWins']}-{h2h['Draws']}-{h2h['AwayWins']} | {h2h['Trail']}",
        })
    return pd.DataFrame(rows), league_tables

def compute_consensus(web_df, weights_df, source_frames, hist_df):
    web_df = enrich_web_df(web_df)
    weights_df = normalize_columns(weights_df)
    hist_df = normalize_columns(hist_df)
    if web_df.empty:
        return pd.DataFrame()
    if weights_df is None or weights_df.empty:
        weights = {}
    else:
        source_col = find_first_column(weights_df, ["Source", "source"])
        weight_col = find_first_column(weights_df, ["Weight", "weight", "Poids", "poids"])
        weights = dict(zip(weights_df[source_col], weights_df[weight_col])) if source_col and weight_col else {}
    hist_lookup = hist_df.set_index("N°").to_dict("index") if hist_df is not None and not hist_df.empty and "N°" in hist_df.columns else {}
    rows = []
    for _, g in web_df.iterrows():
        n = safe_int(g.get("N°"), None)
        if n is None:
            continue
        votes = {"1": 0.0, "X": 0.0, "2": 0.0}
        traces = []
        source_count = 0
        for src, df in source_frames.items():
            df = normalize_columns(df)
            if df is None or df.empty:
                continue
            src_n_col = find_first_column(df, ["N°", "No", "N", "Match_No", "Match No", "match_no"])
            if src_n_col is None:
                continue
            matched = df[df[src_n_col].apply(lambda x: safe_int(x, None)) == n]
            if matched.empty:
                continue
            pred = derive_pick(matched.iloc[0])
            w = float(weights.get(src, 1.0))
            source_count += 1
            if pred in {"1", "X", "2"}:
                votes[pred] += w; traces.append(f"{src}:{pred}@{w:.2f}")
            elif pred in {"1X", "12", "X2"}:
                for ch in pred:
                    votes[ch] += w * 0.55
                traces.append(f"{src}:{pred}@{w:.2f}")
            elif pred == "1X2":
                for ch in ["1", "X", "2"]:
                    votes[ch] += w * 0.34
                traces.append(f"{src}:{pred}@{w:.2f}")
        h = hist_lookup.get(n, {})
        hw = float(weights.get("Historical", 1.0))
        hp = str(h.get("Hist_Pick", "")).strip()
        hd = str(h.get("Hist_Double", "")).strip()
        historical_ok = False
        if hp in {"1", "X", "2"}:
            votes[hp] += hw; historical_ok = True; traces.append(f"Historical:{hp}@{hw:.2f}")
        if hd in {"1X", "12", "X2"} and hd != hp:
            for ch in hd:
                votes[ch] += hw * 0.25
            historical_ok = True; traces.append(f"HistoricalCover:{hd}@{hw:.2f}")
        ordered = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        total = sum(votes.values())
        top, second = ordered[0], ordered[1]
        confidence = top[1] / total if total else 0
        gap = top[1] - second[1]
        if total == 0:
            ticket = ""; level = "NO SIGNAL"; consensus = ""
        elif confidence >= 0.62 and gap >= 1.00:
            ticket = top[0]; level = "BASE"; consensus = top[0]
        elif confidence >= 0.42:
            ticket = ordered_combo([top[0], second[0]]); level = "DOUBLE"; consensus = top[0]
        else:
            ticket = "1X2"; level = "TRIPLE"; consensus = ordered_combo([k for k, v in votes.items() if v > 0])
        risk = classify_risk(confidence, gap, historical_ok)
        recommendation = "Jouer en base" if level == "BASE" else f"Couvrir en {ticket}" if level == "DOUBLE" else "Match très piégeux"
        rows.append({
            "N°": n, "Match": g.get("Match", ""), "League": g.get("League", ""),
            "Vote_1": round(votes["1"], 2), "Vote_X": round(votes["X"], 2), "Vote_2": round(votes["2"], 2),
            "Consensus": consensus, "Ticket": ticket, "Level": level, "Risk": risk, "Confidence": round(confidence, 3),
            "Historical_Pick": h.get("Hist_Pick", ""), "Historical_Double": h.get("Hist_Double", ""),
            "Recommendation": recommendation, "Top_Source_Count": source_count, "Trace": " | ".join(traces),
        })
    return pd.DataFrame(rows)

def to_excel_bytes(sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, index=False, sheet_name=name[:31])
    output.seek(0)
    return output.getvalue()

web_df, weights_df, source_frames, hist_dict = load_inputs()
web_df = enrich_web_df(web_df)
hist_df, league_tables = historical_analysis(web_df, hist_dict)
consensus_df = compute_consensus(web_df, weights_df, source_frames, hist_df)

st.markdown("""
<style>
.block-container {padding-top:0.8rem; max-width:96rem;}
.hero {padding:16px 20px; border-radius:20px; color:white; background:linear-gradient(135deg,#0f172a,#1d4ed8 85%); margin-bottom:12px;}
.small-note {font-size:0.92rem; opacity:0.9;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>MAX PRO - V3 AI PREDICTOR FULL ELITE</h1><p class="small-note">Version renforcée : auto-détection CSV, ligues inférées automatiquement, consensus enrichi, ticket final, niveau de risque et export Excel.</p></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("WEB Matches", len(web_df))
c2.metric("Consensus Rows", len(consensus_df))
c3.metric("Spain Hist", len(hist_dict.get("Spain", pd.DataFrame())))
c4.metric("Germany Hist", len(hist_dict.get("Germany", pd.DataFrame())))

tabs = st.tabs(["WEB DB import", "Historiques ligues", "Historique profond", "Saisie sources", "Consensus MAX PRO", "Ticket final", "Debug V3", "Export Excel"])

with tabs[0]:
    st.subheader("Données importées")
    st.dataframe(web_df, use_container_width=True, height=520, hide_index=True)

with tabs[1]:
    st.subheader("Tous les historiques joints")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Espagne", f"{len(hist_dict.get('Spain', pd.DataFrame()))} lignes")
    h2.metric("Allemagne", f"{len(hist_dict.get('Germany', pd.DataFrame()))} lignes")
    h3.metric("France", f"{len(hist_dict.get('France', pd.DataFrame()))} lignes")
    h4.metric("Italie", f"{len(hist_dict.get('Italy', pd.DataFrame()))} lignes")
    lg = st.selectbox("Afficher l'historique d'une league", ["ALL"] + LEAGUE_ORDER)
    st.dataframe(hist_dict.get(lg, pd.DataFrame()), use_container_width=True, height=420, hide_index=True)

with tabs[2]:
    st.subheader("Analyse historique profonde par match")
    st.dataframe(hist_df, use_container_width=True, height=520, hide_index=True)
    st.markdown("### Classements dérivés")
    lg2 = st.selectbox("League table", LEAGUE_ORDER)
    st.dataframe(league_tables.get(lg2, pd.DataFrame()), use_container_width=True, height=380, hide_index=True)

with tabs[3]:
    st.subheader("Édition des bases CSV")
    src = st.selectbox("Choisir la source", list(SOURCE_FILES.keys()))
    edited = st.data_editor(source_frames.get(src, pd.DataFrame()), use_container_width=True, height=560, num_rows="fixed")
    e1, e2 = st.columns(2)
    if e1.button(f"Enregistrer {src}"):
        save_source(src, edited)
        st.success(f"{src} enregistré.")
        st.rerun()
    if e2.button("Recharger les données"):
        st.cache_data.clear()
        st.rerun()

with tabs[4]:
    st.subheader("Consensus recalibré enrichi")
    st.dataframe(consensus_df, use_container_width=True, height=560, hide_index=True)

with tabs[5]:
    st.subheader("Ticket final MAX PRO V3")
    show = consensus_df[["N°", "Match", "League", "Ticket", "Level", "Risk", "Confidence", "Historical_Pick", "Historical_Double", "Recommendation"]].copy() if not consensus_df.empty else pd.DataFrame(columns=["N°", "Match", "League", "Ticket", "Level", "Risk", "Confidence", "Historical_Pick", "Historical_Double", "Recommendation"])
    st.dataframe(show, use_container_width=True, height=560, hide_index=True)

with tabs[6]:
    st.subheader("Debug V3")
    dbg_rows = []
    dbg_rows.append({"Item": "web_database_import.csv", "Rows": len(web_df), "Columns": ", ".join(map(str, web_df.columns.tolist())) if not web_df.empty else "EMPTY"})
    dbg_rows.append({"Item": "recalibrated_weights.csv", "Rows": len(weights_df), "Columns": ", ".join(map(str, weights_df.columns.tolist())) if not weights_df.empty else "EMPTY"})
    for key in LEAGUE_ORDER:
        df = hist_dict.get(key, pd.DataFrame())
        dbg_rows.append({"Item": f"{key} history", "Rows": len(df), "Columns": ", ".join(map(str, df.columns.tolist())) if not df.empty else "EMPTY"})
    for src, df in source_frames.items():
        dbg_rows.append({"Item": src, "Rows": len(df), "Columns": ", ".join(map(str, df.columns.tolist())) if not df.empty else "EMPTY"})
    st.dataframe(pd.DataFrame(dbg_rows), use_container_width=True, height=560, hide_index=True)

with tabs[7]:
    st.subheader("Télécharger le projet complet")
    payload = {
        "WEB_DB_Import": web_df,
        "Weights": weights_df,
        "Historical_All": hist_dict.get("ALL", pd.DataFrame()),
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
    st.download_button("Télécharger l'Excel MAX PRO V3", data=to_excel_bytes(payload), file_name="MAX_PRO_V3_AI_PREDICTOR_FULL_ELITE.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
