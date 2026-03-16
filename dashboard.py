from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import copy
import config

LOG_PATH        = "logs/run_log.csv"
BINARY_LOG_PATH = "logs/feature_freq_log.csv"

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

#Responsive Meta-Tag + globales CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GA Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body { margin: 0; overflow: hidden; }

            /* Responsive Schriftgrößen */
            @media (max-width: 1400px) {
                .param-label { font-size: 11px !important; }
                .param-badge { font-size: 11px !important; }
                .kpi-value   { font-size: 20px !important; }
            }
            @media (max-width: 1100px) {
                .param-label { font-size: 10px !important; }
                .param-badge { font-size: 10px !important; }
                .kpi-value   { font-size: 16px !important; }
                .card-title  { font-size: 13px !important; }
            }

            /* Karten sauber halten */
            .dash-graph { height: 100% !important; }
            .js-plotly-plot, .plot-container { height: 100% !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

ISLAND_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
START_TIME    = datetime.now()

# ── 1. RUN-LOG LADEN ─────────────────────────────────────────────────────────
try:
    FULL_DF = pd.read_csv(LOG_PATH)
    FULL_DF = FULL_DF.drop_duplicates(
        subset=['island_id', 'generation'], keep='first'
    ).reset_index(drop=True)
    FULL_DF["timestamp"] = pd.to_datetime(FULL_DF["timestamp"], errors='coerce')
    print(f"Run-Log: {len(FULL_DF)} Zeilen")
except Exception as e:
    print(f"Run-Log Fehler: {e}")
    FULL_DF = pd.DataFrame()

# ── 2. FEATURE-FREQ-LOG LADEN ────────────────────────────────────────────────
try:
    with open(BINARY_LOG_PATH, 'r') as f:
        n_header_cols = len(f.readline().split(','))
    print(f"   Header-Spalten: {n_header_cols}")

    BINARY_DF = pd.read_csv(
        BINARY_LOG_PATH,
        on_bad_lines='skip',
        usecols=range(n_header_cols)
    )
    print(f"Feature-Freq-Log geladen: {len(BINARY_DF)} Zeilen")
    print(f"   Alle Spalten ({len(BINARY_DF.columns)}): {list(BINARY_DF.columns[:15])} ...")

    FEATURE_COLS = [
        c for c in BINARY_DF.columns
        if c.startswith('f') and c[1:].isdigit() and len(c) > 1
    ]
    if len(FEATURE_COLS) == 0:
        FEATURE_COLS = [
            c for c in BINARY_DF.columns
            if c.lower().startswith('feature') or
               (c.startswith('F') and c[1:].isdigit())
        ]
        print(f"Fallback: {len(FEATURE_COLS)} Spalten gefunden")

    if len(FEATURE_COLS) == 0:
        print(f"KEINE Feature-Spalten! Spaltennamen: {list(BINARY_DF.columns)}")
    else:
        print(f"{len(FEATURE_COLS)} Feature-Spalten: {FEATURE_COLS[:5]} ... {FEATURE_COLS[-3:]}")

    FEATURE_COLS = FEATURE_COLS[:-1]
    print(f"Nach Entfernen der letzten Spalte: {len(FEATURE_COLS)} Features")

    BINARY_DF[FEATURE_COLS] = BINARY_DF[FEATURE_COLS].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)

except Exception as e:
    print(f"Feature-Freq-Log Fehler: {e}")
    import traceback; traceback.print_exc()
    BINARY_DF    = pd.DataFrame()
    FEATURE_COLS = []

# ── 3. FEATURE-NAMEN AUS RESULTS-DATEI LADEN ─────────────────────────────────
try:
    results_file   = Path("results_data/IB Seed 9 No NR/size300_run1/results_run1_2026-02-18.xlsx")
    df_feat_header = pd.read_excel(results_file, sheet_name='full_result_IB_AVG', header=0)
    raw_names      = list(df_feat_header.columns[4:-1])
    if len(raw_names) == len(FEATURE_COLS):
        FEATURE_NAMES = raw_names
        print(f"Feature-Namen geladen: {FEATURE_NAMES[0]} ... {FEATURE_NAMES[-1]}")
    else:
        FEATURE_NAMES = FEATURE_COLS
        print(f"Feature-Namen Länge passt nicht ({len(raw_names)} vs {len(FEATURE_COLS)}) → Fallback")
except Exception as e:
    print(f"Feature-Namen Fehler (nicht kritisch): {e}")
    FEATURE_NAMES = FEATURE_COLS

# ── 4. CUMULATIVE SUMS VORBERECHNEN ──────────────────────────────────────────
if not BINARY_DF.empty and len(FEATURE_COLS) > 0:
    try:
        feature_vals = BINARY_DF[FEATURE_COLS].values.astype(np.int32)
        island_ids   = BINARY_DF["island_id"].values
        CUMSUM_ALL   = feature_vals.cumsum(axis=0)
        CUMSUM_ISLAND = {}
        for iid in range(4):
            running = np.zeros(len(FEATURE_COLS), dtype=np.int32)
            result  = np.zeros_like(feature_vals)
            for i in range(len(feature_vals)):
                if island_ids[i] == iid:
                    running = running + feature_vals[i]
                result[i] = running
            CUMSUM_ISLAND[iid] = result
        print(f"Cumulative Sums vorberechnet: {CUMSUM_ALL.shape}")
    except Exception as e:
        print(f"Cumulative Sums Fehler: {e}")
        CUMSUM_ALL    = np.array([])
        CUMSUM_ISLAND = {}
else:
    print(f"CUMSUM übersprungen: BINARY_DF leer={BINARY_DF.empty}, FEATURE_COLS={len(FEATURE_COLS)}")
    CUMSUM_ALL    = np.array([])
    CUMSUM_ISLAND = {}

# ── 5. GENE SELECTION FREQUENCY VORBERECHNEN ─────────────────────────────────
if not BINARY_DF.empty and len(FEATURE_COLS) > 0 and 'generation' in BINARY_DF.columns:
    try:
        def _gene_freq(df_sub):
            grp = df_sub.groupby('generation')[FEATURE_COLS].sum()
            return grp.values.astype(np.float32), grp.index.tolist()

        GENE_FREQ_ALL, GENE_FREQ_GENS = _gene_freq(BINARY_DF)
        GENE_FREQ_ISLAND = {}
        for iid in range(4):
            df_isl = BINARY_DF[BINARY_DF['island_id'] == iid]
            GENE_FREQ_ISLAND[iid] = _gene_freq(df_isl) if not df_isl.empty else (np.array([]), [])
        print(f"Gene-Freq vorberechnet: {GENE_FREQ_ALL.shape}")
    except Exception as e:
        print(f"Gene-Freq Fehler: {e}")
        GENE_FREQ_ALL    = np.array([])
        GENE_FREQ_GENS   = []
        GENE_FREQ_ISLAND = {}
else:
    print(f"Gene-Freq übersprungen")
    GENE_FREQ_ALL    = np.array([])
    GENE_FREQ_GENS   = []
    GENE_FREQ_ISLAND = {}

# ── 6. PLAYBACK-SPEED ────────────────────────────────────────────────────────
PLAYBACK_SPEED = max(1, len(FULL_DF) // 300) if not FULL_DF.empty else 1
print(f"Playback-Speed: {PLAYBACK_SPEED}x")

# ── 7. GRUPPENREGELN LADEN ───────────────────────────────────────────────────
try:
    data_path   = Path(getattr(config, 'DATA_BASE_PATH', getattr(config, 'data_base_path', 'data')))
    GROUP_RULES = pd.read_excel(
        data_path / "group_rules_A.xlsx",
        usecols=[0, 1, 2], header=0, sheet_name=2
    )
    GROUP_RULES.columns = ["Group-Name", "start_index", "end_index"]
    GROUP_RULES["start_index"] = pd.to_numeric(GROUP_RULES["start_index"], errors="coerce")
    GROUP_RULES["end_index"]   = pd.to_numeric(GROUP_RULES["end_index"],   errors="coerce")
    GROUP_RULES = GROUP_RULES.dropna(
        subset=["start_index", "end_index"]
    ).astype({"start_index": int, "end_index": int})
    print(f"Gruppenregeln: {len(GROUP_RULES)} Gruppen")
except Exception as e:
    print(f"Gruppenregeln nicht geladen (optional): {e}")
    GROUP_RULES = pd.DataFrame()

try:
    lab_path   = (Path(config.data_base_path) / Path(config.lab_dir)).resolve()
    lab_df     = pd.read_excel(lab_path) if lab_path.suffix in ['.xlsx', '.xls'] else pd.read_csv(lab_path)
    N_SAMPLES  = len(lab_df)
    print(f"Datensatzgröße: {N_SAMPLES} Zeilen ({lab_path.name})")
except Exception as e:
    print(f"Datensatzgröße nicht geladen: {e}")
    N_SAMPLES = getattr(config, 'n_samples', getattr(config, 'data_shape', '–'))

# ── 8. BASE FEATURE FIGURE ───────────────────────────────────────────────────
def make_base_feat_fig():
    fig = go.Figure()
    if not GROUP_RULES.empty:
        group_colors = px.colors.qualitative.Pastel
        for i, grow in GROUP_RULES.iterrows():
            fig.add_vrect(
                x0=grow["start_index"] - 0.5, x1=grow["end_index"] + 0.5,
                fillcolor=group_colors[i % len(group_colors)],
                opacity=0.35, layer="below", line_width=0,
            )
    return fig

BASE_FEAT_FIG = make_base_feat_fig()


# ── 9. HILFSFUNKTIONEN ───────────────────────────────────────────────────────
def resp_graph(gid, widgets=False):
    return dcc.Graph(
        id=gid, figure={},
        responsive=True,
        style={"flex": "1", "minHeight": "0"},
        config={
            "displayModeBar": widgets,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"] if widgets else [],
            "scrollZoom": widgets,
        }
    )

def flex_card(*children, height="38vh", mb=True):
    return html.Div(list(children), className="card p-2", style={
        "height": height,
        "minHeight": "120px",
        "display": "flex",
        "flexDirection": "column",
        "marginBottom": "8px" if mb else "0",
        "overflow": "hidden",
    })

def make_param_card():
    total_gen  = config.migration_interval * config.iterations
    train_size = getattr(config, 'train_size', 0.8)
    n_samples  = N_SAMPLES
    params = [
        ("Populationsgröße",    str(config.pop_size)),
        ("Inseln",              str(config.num_islands)),
        ("Iterationen",         f"{config.iterations}  →  {total_gen} Gen."),
        ("Migration Interval",  str(config.migration_interval)),
        ("Migrationsstrategie", config.migration_strategy),
        ("Topologie",           config.topology),
        ("Kreuzungswahrsch.",   str(config.cxpb)),
        ("Mutationswahrsch.",   str(config.mutpb)),
        ("Regeltyp",            config.rule_type),
        ("Datensatzgröße",      str(n_samples)),
        ("Train / Test Split",  f"{train_size} / {round(1 - train_size, 2)}"),
    ]
    rows = []
    for label, value in params:
        rows.append(html.Div([
            html.Span(label, className="param-label", style={
                "color": "#6c757d", "fontSize": "12px", "fontWeight": "500",
                "whiteSpace": "nowrap", "overflow": "hidden",
                "textOverflow": "ellipsis", "maxWidth": "55%",
            }),
            html.Span(value, className="param-badge", style={
                "backgroundColor": "#e8f0fa", "color": "#1a3a5c",
                "fontWeight": "bold", "fontSize": "12px",
                "padding": "2px 10px", "borderRadius": "12px",
                "whiteSpace": "nowrap", "minWidth": "110px",
                "textAlign": "center", "display": "inline-block",
            }),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "3px 2px",
            "borderBottom": "1px solid #f0f0f0",
        }))
    return html.Div(rows, style={
        "display": "flex", "flexDirection": "column",
        "justifyContent": "space-evenly",
        "flex": "1", "overflowY": "auto",
    })

def fitness_xaxis(title="Generation"):
    return dict(
        title=title, type="linear",
        showspikes=True, spikemode="across",
        spikesnap="cursor", spikecolor="#aaaaaa", spikethickness=1,
    )


# ── 10. LAYOUT ────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[
    dcc.Interval(id="interval", interval=600, n_intervals=0),

    # OBERE KACHELN  (10vh)
    dbc.Row([
        dbc.Col(html.Div([
            html.H5("Runtime",     className="card-title", style={"marginBottom": "4px", "fontSize": "14px"}),
            html.Div(id="runtime", children="--", className="kpi-value",
                     style={"fontSize": "22px", "fontWeight": "bold"})
        ], className="card p-3 h-100"), width=3),
        dbc.Col(html.Div([
            html.H5("Generationen", className="card-title", style={"marginBottom": "4px", "fontSize": "14px"}),
            html.Div(id="gen",      children="--", className="kpi-value",
                     style={"fontSize": "22px", "fontWeight": "bold"})
        ], className="card p-3 h-100"), width=3),
        dbc.Col(html.Div([
            html.H5("#Lösungen",   className="card-title", style={"marginBottom": "4px", "fontSize": "14px"}),
            html.Div(id="prog",    children="--", className="kpi-value",
                     style={"fontSize": "22px", "fontWeight": "bold"})
        ], className="card p-3 h-100"), width=3),
        dbc.Col(html.Div([
            html.H5("Score",       className="card-title", style={"marginBottom": "4px", "fontSize": "14px"}),
            html.Div(id="score",   children="--", className="kpi-value",
                     style={"fontSize": "22px", "fontWeight": "bold"})
        ], className="card p-3 h-100"), width=3),
    ], className="mb-2 g-2", style={"height": "10vh"}),

    # MITTLERER BEREICH  (80vh)
    dbc.Row([
        # LINKS: Bar-Chart + Heatmap
        dbc.Col([
            flex_card(
                html.H5("Feature-Häufigkeit (kumulativ)",
                        className="card-title", style={"marginBottom": "4px", "fontSize": "13px"}),
                dcc.Dropdown(
                    id="island-filter",
                    options=[{'label': 'Alle Inseln', 'value': -1}] +
                            [{'label': f'Insel {i+1}', 'value': i} for i in range(4)],
                    value=-1, clearable=False, className="mb-1",
                    style={"fontSize": "13px"},
                ),
                resp_graph("matrix_chart"),
                height="38vh",
            ),
            flex_card(
                html.H5("Gene Selection Frequency",
                        className="card-title", style={"marginBottom": "4px", "fontSize": "13px"}),
                resp_graph("heatmap_chart"),
                height="38vh", mb=False,
            ),
        ], width=4),

        # MITTE: Best + Avg Fitness
        dbc.Col([
            flex_card(
                html.H5("Best Fitness je Insel",
                        className="card-title", style={"marginBottom": "4px", "fontSize": "13px"}),
                resp_graph("best_chart", widgets=True),
                height="38vh",
            ),
            flex_card(
                html.H5("Avg Fitness je Insel",
                        className="card-title", style={"marginBottom": "4px", "fontSize": "13px"}),
                resp_graph("avg_chart", widgets=True),
                height="38vh", mb=False,
            ),
        ], width=4),

        # RECHTS: GA Parameter (38vh) + Diversity (38vh)
        dbc.Col([
            html.Div([
                html.H5("GA Parameter",
                        className="card-title", style={"marginBottom": "4px", "fontSize": "13px"}),
                html.Hr(style={"margin": "6px 0"}),
                make_param_card(),
            ], className="card p-3", style={
                "height": "38vh",
                "minHeight": "120px",
                "display": "flex",
                "flexDirection": "column",
                "marginBottom": "8px",
                "overflow": "hidden",
            }),
            flex_card(
                html.H5("Diversity je Insel",
                        className="card-title", style={"marginBottom": "4px", "fontSize": "13px"}),
                resp_graph("div_chart"),
                height="38vh",
                mb=False,
            ),
        ], width=4),
    ], className="g-2", style={"height": "80vh"}),

    # UNTEN: Status  (5vh)
    dbc.Row([
        dbc.Col(html.Div(
            html.Span(id="footer-status", style={"fontSize": "13px", "color": "#6c757d"}),
            className="card p-2", style={"textAlign": "center"}
        ), width=12),
    ], className="mt-2", style={"height": "5vh"}),

], style={
    "padding": "10px",
    "height": "100vh",
    "overflow": "hidden",
    "minWidth": "900px",
})


# ── 11. HAUPT-CALLBACK ───────────────────────────────────────────────────────
@app.callback(
    Output("runtime",       "children"),
    Output("gen",           "children"),
    Output("prog",          "children"),
    Output("score",         "children"),
    Output("best_chart",    "figure"),
    Output("avg_chart",     "figure"),
    Output("div_chart",     "figure"),
    Output("matrix_chart",  "figure"),
    Output("heatmap_chart", "figure"),
    Output("footer-status", "children"),
    Input("interval",       "n_intervals"),
    Input("island-filter",  "value")
)
def update_all(n, iid):
    empty_fig = go.Figure()
    empty = "--", "--", "--", "--", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, ""

    try:
        if FULL_DF.empty:
            return empty

        # ── Cursor ────────────────────────────────────────────────────────────
        idx     = min(n * PLAYBACK_SPEED, len(FULL_DF) - 1)
        step    = max(1, idx // 200)
        df_plot = FULL_DF.iloc[:idx+1:step]
        row     = FULL_DF.iloc[idx]
        bin_idx = min(n * PLAYBACK_SPEED, len(BINARY_DF) - 1) if not BINARY_DF.empty else 0

        # ── Runtime ───────────────────────────────────────────────────────────
        elapsed = (datetime.now() - START_TIME).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        runtime = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        gen    = str(int(row["generation"]))
        prog   = f"{idx}/{len(FULL_DF)}"
        score  = f"{FULL_DF.iloc[:idx+1]['best_fitness'].min():.3f}"
        footer = f"Zuletzt aktualisiert: {datetime.now().strftime('%H:%M:%S')}"

        def base_layout(widgets=False):
            return dict(
                margin=dict(l=35, r=8, t=8 if not widgets else 28, b=28),
                autosize=True,
                showlegend=widgets,
                legend=dict(orientation="h", y=1.08, x=1, xanchor="right",
                            font=dict(size=11)) if widgets else {},
            )

        # ── Fitness & Diversity ───────────────────────────────────────────────
        fig_best = go.Figure()
        fig_avg  = go.Figure()
        fig_div  = go.Figure()
        for i in range(4):
            d = df_plot[df_plot["island_id"] == i]
            if not d.empty:
                fig_best.add_trace(go.Scatter(
                    x=d["generation"], y=d["best_fitness"],
                    mode='lines', name=f'Insel {i+1}',
                    line=dict(color=ISLAND_COLORS[i], width=2)
                ))
                fig_avg.add_trace(go.Scatter(
                    x=d["generation"], y=d["avg_fitness"],
                    mode='lines', name=f'Insel {i+1}',
                    line=dict(color=ISLAND_COLORS[i], width=2)
                ))
                fig_div.add_trace(go.Scatter(
                    x=d["generation"], y=d["normalized_diversity"],
                    mode='lines', name=f'Insel {i+1}',
                    line=dict(color=ISLAND_COLORS[i], width=2)
                ))

        fig_best.update_layout(**base_layout(widgets=True),
            xaxis=fitness_xaxis("Generation"),
            yaxis=dict(title="Best Fitness", title_font=dict(size=11)),
        )
        fig_avg.update_layout(**base_layout(widgets=True),
            xaxis=fitness_xaxis("Generation"),
            yaxis=dict(title="Avg Fitness", title_font=dict(size=11)),
        )
        fig_div.update_layout(
            margin=dict(l=35, r=8, t=28, b=28),
            autosize=True,
            showlegend=True,
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=11)),
            xaxis=dict(title="Generation", title_font=dict(size=11)),
            yaxis=dict(title="Diversity",  title_font=dict(size=11)),
        )

        # ── Feature Bar Chart ─────────────────────────────────────────────────
        if CUMSUM_ALL.size == 0 or len(FEATURE_COLS) == 0:
            return runtime, gen, prog, score, fig_best, fig_avg, fig_div, empty_fig, empty_fig, footer

        counts = (
            CUMSUM_ALL[bin_idx]
            if (iid is None or iid < 0)
            else CUMSUM_ISLAND.get(iid, CUMSUM_ALL)[bin_idx]
        )
        x_vals   = list(range(len(FEATURE_COLS)))
        fig_feat = copy.deepcopy(BASE_FEAT_FIG)
        fig_feat.add_trace(go.Bar(
            x=x_vals, y=counts,
            customdata=FEATURE_NAMES,
            marker_color='#1f77b4', opacity=0.75, showlegend=False,
            hovertemplate="<b>%{customdata}</b><br>Selektionen: %{y}<extra></extra>",
        ))
        island_label = f"Insel {iid+1}" if (iid is not None and iid >= 0) else "Alle Inseln"
        fig_feat.update_layout(
            autosize=True,
            margin=dict(l=35, r=8, t=28, b=28),
            title=dict(text=f"Gen {gen} – {island_label}", font=dict(size=11), x=0.5),
            xaxis=dict(showticklabels=False, title="Produktionsgruppen", title_font=dict(size=11)),
            yaxis=dict(title="Kum. Selektionen", title_font=dict(size=11)),
            bargap=0.1,
        )

        # ── Gene Selection Heatmap ────────────────────────────────────────────
        fig_heat = go.Figure()
        if GENE_FREQ_ALL.size > 0:
            if iid is None or iid < 0:
                matrix, gens = GENE_FREQ_ALL, GENE_FREQ_GENS
            else:
                matrix, gens = GENE_FREQ_ISLAND.get(iid, (GENE_FREQ_ALL, GENE_FREQ_GENS))

            if matrix.size > 0:
                current_gen_val = int(row["generation"])
                mask       = [g <= current_gen_val for g in gens]
                matrix_cut = matrix[mask]
                gens_cut   = [g for g, m in zip(gens, mask) if m]

                if len(gens_cut) > 0:
                    feat_names_2d = np.tile(
                        np.array(FEATURE_NAMES)[:, np.newaxis], (1, len(gens_cut))
                    )
                    fig_heat.add_trace(go.Heatmap(
                        z=matrix_cut.T,
                        x=gens_cut,
                        y=list(range(len(FEATURE_COLS))),
                        colorscale='BuPu',
                        showscale=True,
                        customdata=feat_names_2d,
                        hovertemplate=(
                            "Gen: %{x}<br>"
                            "Feature: <b>%{customdata}</b><br>"
                            "Freq: %{z:.0f}<extra></extra>"
                        ),
                        colorbar=dict(
                            thickness=10, len=0.9,
                            title=dict(text="Freq", side="right", font=dict(size=10))
                        ),
                    ))
            fig_heat.update_layout(
                autosize=True,
                margin=dict(l=35, r=55, t=8, b=28),
                xaxis=dict(title="Generation", title_font=dict(size=11)),
                yaxis=dict(title="Feature Index", showticklabels=False, title_font=dict(size=11)),
            )

        return runtime, gen, prog, score, fig_best, fig_avg, fig_div, fig_feat, fig_heat, footer

    except Exception as e:
        print(f"Callback Fehler Tick {n}: {e}")
        import traceback; traceback.print_exc()
        return "--", "--", "--", "--", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), ""


# ── 12. START ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=8051)
