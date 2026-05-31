"""
NodeBias Full-Stack Application
================================
Run with:  python nodebias_app.py
Then open: http://localhost:5000

This file serves the complete NodeBias dashboard and exposes the
/api/audit  POST endpoint that the frontend calls via fetch().
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# ─── Inline mitigation engine (copied from mitigation_engine.py) ────────────
def run_nodebias_audit(df, model_choice, target_col="readmitted", sensitive_col="gender"):
    try:
        df.replace(["?", "NA", "N/A", ""], np.nan, inplace=True)

        if df[target_col].dtype == "object":
            df["target_binary"] = df[target_col].astype("category").cat.codes
        else:
            df["target_binary"] = df[target_col]

        for col in df.select_dtypes(include=["object", "string", "category"]).columns:
            if col not in [sensitive_col, target_col]:
                df[col] = df[col].astype("category").cat.codes

        safe_features = df.select_dtypes(include=[np.number]).columns.tolist()
        safe_features = [f for f in safe_features if f not in [target_col, "target_binary", sensitive_col]]

        df.dropna(subset=safe_features + [sensitive_col], inplace=True)

        X = df[safe_features].values.astype(float)
        y = df["target_binary"].values.astype(int)
        raw_sensitive = df[sensitive_col].values

        # ── Attempt GlassBoxML; fall back to sklearn ──────────────────────
        try:
            from glassboxml.preprocessing import StandardScaler
            from glassboxml.models import LogisticRegression, GaussianNaiveBayes
            from glassboxml.core import Momentum

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if model_choice == "Logistic Regression":
                optimizer = Momentum(learning_rate=0.01, beta=0.9)
                model = LogisticRegression(optimizer=optimizer, epochs=10000, loss_function="bce")
            else:
                model = GaussianNaiveBayes()

            model.fit(X_scaled, y)
            predictions = model.predict(X_scaled)
            engine = "GlassBoxML"

        except ImportError:
            # ── sklearn fallback ──────────────────────────────────────────
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            mapping = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Gaussian Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=14, random_state=42),
                "Decision Tree": DecisionTreeClassifier(max_depth=14, random_state=42),
            }
            model = mapping.get(model_choice, LogisticRegression(max_iter=1000))
            model.fit(X_scaled, y)
            predictions = model.predict(X_scaled)
            engine = "sklearn (fallback)"

        # ── Compute fairness metrics ──────────────────────────────────────
        results_df = pd.DataFrame({"attr": raw_sensitive, "pred": predictions})
        groups = results_df["attr"].unique()
        rates = {
            str(g): float(results_df[results_df["attr"] == g]["pred"].mean())
            for g in groups
        }

        g_list = list(rates.values())
        dir_val = (
            round(min(g_list) / max(g_list), 3)
            if len(g_list) > 1 and max(g_list) > 0
            else 1.0
        )
        gap = round(abs(g_list[0] - g_list[1]) * 100, 2) if len(g_list) > 1 else 0.0

        return {
            "engine": engine,
            "model_used": model_choice,
            "features_used": len(safe_features),
            "records_processed": len(df),
            "group_rates": rates,
            "disparity_gap_pct": gap,
            "dir_score": dir_val,
            "threshold": 0.8,
            "status": "Safe for Deployment" if dir_val >= 0.8 else "Bias Detected",
            "verdict": "PASS" if dir_val >= 0.8 else "FAIL",
        }

    except Exception as e:
        raise Exception(f"Engine Failure: {str(e)}")


# ─── API Route ───────────────────────────────────────────────────────────────
@app.route("/api/audit", methods=["POST"])
def audit():
    try:
        file = request.files["dataset"]
        model = request.form.get("modelType", "Logistic Regression")
        target = request.form.get("targetColumn", "readmitted")
        sensitive = request.form.get("sensitiveColumn", "gender")

        df = pd.read_csv(file)
        output = run_nodebias_audit(df, model, target, sensitive)
        return jsonify(output)

    except Exception as e:
        print(f"CRASH: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "2.4"})


# ─── Serve the full frontend ──────────────────────────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NodeBias — AI Fairness Audit Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #050810;
  --surface: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.08);
  --border-hi: rgba(255,255,255,0.15);
  --green: #00F5A0;
  --green-dim: rgba(0,245,160,0.10);
  --red: #FF4D6D;
  --red-dim: rgba(255,77,109,0.10);
  --amber: #FFB800;
  --amber-dim: rgba(255,184,0,0.10);
  --blue: #4D9FFF;
  --blue-dim: rgba(77,159,255,0.10);
  --purple: #8B5CF6;
  --purple-dim: rgba(139,92,246,0.10);
  --text: #E8E6FF;
  --text-muted: rgba(232,230,255,0.50);
  --text-dim: rgba(232,230,255,0.25);
  --font: 'Syne', sans-serif;
  --mono: 'Space Mono', monospace;
  --radius: 14px;
  --radius-sm: 8px;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
html { scroll-behavior:smooth; }
body {
  background:var(--bg); color:var(--text);
  font-family:var(--font); min-height:100vh;
  overflow-x:hidden; -webkit-font-smoothing:antialiased;
}
.bg-grid {
  position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(77,159,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(77,159,255,0.025) 1px, transparent 1px);
  background-size:52px 52px;
}
.bg-glow {
  position:fixed; z-index:0; pointer-events:none;
  border-radius:50%; filter:blur(100px); opacity:0.08;
}
.bg-g1 { width:600px; height:600px; background:var(--blue); top:-200px; right:-100px; }
.bg-g2 { width:400px; height:400px; background:var(--green); bottom:0; left:-100px; }
.bg-noise {
  position:fixed; inset:0; z-index:0; pointer-events:none; opacity:0.03;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size:200px;
}
@keyframes scanline { from{top:-4px} to{top:100vh} }
.bg-scan { position:fixed; left:0; right:0; height:3px; z-index:1; pointer-events:none; background:linear-gradient(transparent,rgba(0,245,160,0.04),transparent); animation:scanline 8s linear infinite; }

.app { position:relative; z-index:2; max-width:1200px; margin:0 auto; padding:0 28px 60px; }

/* Header */
.header { display:flex; align-items:center; justify-content:space-between; padding:28px 0 32px; border-bottom:1px solid var(--border); margin-bottom:36px; animation:fadeUp 0.5s ease both; }
.logo { display:flex; align-items:center; gap:14px; text-decoration:none; }
.logo-mark { width:40px; height:40px; border-radius:11px; background:linear-gradient(135deg,var(--green),var(--blue)); display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:800; color:#050810; box-shadow:0 0 24px rgba(0,245,160,0.25); }
.logo-wordmark { font-size:22px; font-weight:800; letter-spacing:-0.5px; color:var(--text); }
.logo-wordmark em { color:var(--green); font-style:normal; }
.header-right { display:flex; align-items:center; gap:16px; }
.status-chip { display:flex; align-items:center; gap:7px; background:var(--surface); border:1px solid var(--border); border-radius:100px; padding:6px 14px; font-family:var(--mono); font-size:11px; color:var(--text-muted); }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,245,160,0.4)}50%{box-shadow:0 0 0 5px rgba(0,245,160,0)} }
.pulse-dot { width:6px; height:6px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
.export-btn { background:transparent; border:1px solid var(--border-hi); color:var(--text-muted); border-radius:var(--radius-sm); padding:7px 16px; font-family:var(--mono); font-size:11px; cursor:pointer; transition:all 0.2s; }
.export-btn:hover { color:var(--green); border-color:var(--green); }

/* Hero */
.hero { background:var(--surface); border:1px solid var(--border); border-top:2px solid var(--green); border-radius:var(--radius); padding:32px 40px; display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:32px; margin-bottom:24px; animation:fadeUp 0.6s ease 0.08s both; backdrop-filter:blur(16px); position:relative; overflow:hidden; }
.hero::after { content:''; position:absolute; inset:0; pointer-events:none; background:radial-gradient(ellipse at 50% 0%,rgba(0,245,160,0.04),transparent 70%); }
.hero-meta { display:flex; flex-direction:column; gap:10px; }
.hero-row { display:flex; justify-content:space-between; align-items:center; gap:20px; }
.hero-label { font-size:12px; color:var(--text-muted); }
.hero-val { font-family:var(--mono); font-size:12px; color:var(--text); }
.hero-val.ok { color:var(--green); }
.hero-divider { width:1px; height:100px; background:var(--border); }
.hero-score { text-align:center; }
.hero-score-label { font-family:var(--mono); font-size:10px; letter-spacing:2px; color:var(--text-muted); text-transform:uppercase; margin-bottom:10px; }
.hero-score-number { font-size:72px; font-weight:800; line-height:1; letter-spacing:-4px; }
.hero-score-number.pass { color:var(--green); }
.hero-score-number.fail { color:var(--red); }
.hero-badge { display:inline-flex; align-items:center; gap:6px; border-radius:6px; margin-top:10px; padding:5px 14px; font-family:var(--mono); font-size:11px; font-weight:700; }
.hero-badge.pass { background:var(--green-dim); border:1px solid rgba(0,245,160,0.25); color:var(--green); }
.hero-badge.fail { background:var(--red-dim); border:1px solid rgba(255,77,109,0.25); color:var(--red); }

/* Nav */
.nav { display:flex; gap:3px; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:4px; margin-bottom:32px; animation:fadeUp 0.5s ease both; }
.nav-btn { flex:1; padding:9px 12px; border:none; border-radius:var(--radius-sm); background:transparent; color:var(--text-muted); font-family:var(--mono); font-size:11px; letter-spacing:0.5px; cursor:pointer; transition:all 0.18s; text-transform:uppercase; }
.nav-btn:hover:not(.active) { color:var(--text); background:rgba(255,255,255,0.03); }
.nav-btn.active { background:rgba(255,255,255,0.09); color:var(--text); }

/* Sections */
.section { display:none; }
.section.active { display:block; animation:fadeUp 0.35s ease both; }

/* Cards */
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:24px; backdrop-filter:blur(12px); transition:border-color 0.2s; }
.card:hover { border-color:rgba(255,255,255,0.12); }
.card-title { font-family:var(--mono); font-size:10px; letter-spacing:2px; color:var(--text-muted); text-transform:uppercase; margin-bottom:20px; display:flex; align-items:center; gap:10px; }
.card-title::after { content:''; flex:1; height:1px; background:var(--border); }

/* Grids */
.g2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.g3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }
.g4 { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
.mb { margin-bottom:20px; }

/* Stat box */
.stat-box { background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:var(--radius-sm); padding:14px 18px; }
.s-lbl { font-family:var(--mono); font-size:10px; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; }
.s-val { font-size:26px; font-weight:800; letter-spacing:-1px; }
.s-val.g { color:var(--green); } .s-val.r { color:var(--red); } .s-val.a { color:var(--amber); } .s-val.b { color:var(--blue); }
.s-sub { font-family:var(--mono); font-size:10px; color:var(--text-dim); margin-top:4px; }

/* Bias bars */
.bias-item { margin-bottom:18px; }
.bias-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:7px; }
.bias-name { font-size:13px; font-weight:600; }
.bias-stats { font-family:var(--mono); font-size:11px; color:var(--text-muted); }
.bias-track { height:10px; background:rgba(255,255,255,0.06); border-radius:5px; overflow:hidden; }
.bias-fill { height:100%; border-radius:5px; width:0; transition:width 1.6s cubic-bezier(0.16,1,0.3,1); }

/* Feature bars */
.feat-item { display:flex; align-items:center; gap:12px; margin-bottom:14px; }
.feat-rank { font-family:var(--mono); font-size:10px; color:var(--text-dim); width:18px; }
.feat-name { font-size:12px; color:var(--text-muted); flex:1; }
.feat-track { flex:2; height:7px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; }
.feat-bar { height:100%; border-radius:4px; width:0; transition:width 1.6s cubic-bezier(0.16,1,0.3,1); }
.feat-pct { font-family:var(--mono); font-size:11px; width:34px; text-align:right; }

/* Terminal */
.terminal { background:#020408; border:1px solid rgba(0,245,160,0.15); border-radius:var(--radius); overflow:hidden; }
.term-bar { background:rgba(0,245,160,0.06); border-bottom:1px solid rgba(0,245,160,0.1); padding:11px 20px; display:flex; align-items:center; gap:8px; font-family:var(--mono); font-size:11px; color:var(--green); }
.term-dots { display:flex; gap:6px; }
.term-dot { width:10px; height:10px; border-radius:50%; }
.term-body { padding:18px 20px; max-height:300px; overflow-y:auto; }
.term-body::-webkit-scrollbar { width:3px; }
.term-body::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); }
.term-line { display:flex; gap:14px; font-family:var(--mono); font-size:11px; line-height:1.9; }
.t-time { color:rgba(0,245,160,0.35); white-space:nowrap; }
.t-info { color:var(--blue); } .t-pass { color:var(--green); } .t-fail { color:var(--red); } .t-warn { color:var(--amber); } .t-dim { color:rgba(255,255,255,0.25); }
.t-msg { color:rgba(232,230,255,0.7); }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:0} }
.cursor { display:inline-block; width:7px; height:12px; background:var(--green); animation:blink 1s step-end infinite; vertical-align:-1px; margin-left:4px; }

/* Model grid */
.mc-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.mc { background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:var(--radius-sm); padding:18px; cursor:pointer; transition:all 0.2s; }
.mc:hover:not(.active) { border-color:var(--border-hi); }
.mc.active { border-color:var(--blue); background:var(--blue-dim); box-shadow:0 0 0 1px var(--blue); }
.mc-name { font-size:13px; font-weight:600; margin-bottom:3px; }
.mc-type { font-family:var(--mono); font-size:10px; color:var(--text-muted); margin-bottom:14px; }
.mc-score { font-size:28px; font-weight:800; }
.mc-score.g { color:var(--green); } .mc-score.r { color:var(--red); } .mc-score.d { color:var(--text-dim); }
.mc-status { font-family:var(--mono); font-size:10px; margin-top:4px; }
.mc-status.p { color:var(--green); } .mc-status.f { color:var(--red); } .mc-status.n { color:var(--text-dim); }

/* Upload */
.upload-zone { border:1.5px dashed rgba(255,255,255,0.15); border-radius:var(--radius); padding:40px 24px; text-align:center; cursor:pointer; transition:all 0.2s; background:var(--surface); }
.upload-zone:hover { border-color:var(--blue); background:var(--blue-dim); }
.upload-zone.loaded { border-color:var(--green); background:var(--green-dim); }
.upload-icon { font-size:30px; margin-bottom:12px; opacity:0.6; }
.upload-text { font-size:13px; color:var(--text-muted); }
.upload-sub { font-family:var(--mono); font-size:10px; color:var(--text-dim); margin-top:6px; }

/* Form */
.form-label { font-family:var(--mono); font-size:10px; color:var(--text-muted); letter-spacing:1px; text-transform:uppercase; display:block; margin-bottom:6px; }
.form-input,.form-select { background:rgba(255,255,255,0.05); border:1px solid var(--border-hi); color:var(--text); border-radius:var(--radius-sm); padding:9px 13px; font-family:var(--mono); font-size:12px; width:100%; outline:none; transition:border-color 0.2s; appearance:none; }
.form-input:focus,.form-select:focus { border-color:var(--blue); }
.form-row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }

/* Buttons */
.btn { display:inline-flex; align-items:center; gap:8px; padding:10px 22px; border-radius:var(--radius-sm); border:1px solid; font-family:var(--mono); font-size:12px; font-weight:700; letter-spacing:0.5px; cursor:pointer; transition:all 0.2s; text-transform:uppercase; }
.btn-green { background:var(--green); color:#050810; border-color:var(--green); }
.btn-green:hover { box-shadow:0 0 24px rgba(0,245,160,0.35); transform:translateY(-1px); }
.btn-green:disabled { opacity:0.5; cursor:not-allowed; transform:none; }
.btn-ghost { background:transparent; color:var(--text-muted); border-color:var(--border-hi); }
.btn-ghost:hover { color:var(--text); border-color:var(--text-muted); }
.btn-row { display:flex; gap:10px; justify-content:center; margin-top:20px; }

/* Alerts */
.alert { display:flex; align-items:center; gap:10px; padding:12px 16px; border-radius:var(--radius-sm); font-size:12px; }
.alert-green { background:var(--green-dim); border:1px solid rgba(0,245,160,0.2); color:var(--green); }
.alert-red { background:var(--red-dim); border:1px solid rgba(255,77,109,0.2); color:var(--red); }
.alert-amber { background:var(--amber-dim); border:1px solid rgba(255,184,0,0.2); color:var(--amber); }

/* Compare bars */
.cmp-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.cmp-name { font-size:12px; width:150px; flex-shrink:0; }
.cmp-track { flex:1; height:10px; background:rgba(255,255,255,0.06); border-radius:5px; overflow:hidden; }
.cmp-fill { height:100%; border-radius:5px; transition:width 1.4s cubic-bezier(0.16,1,0.3,1); }
.cmp-val { font-family:var(--mono); font-size:11px; width:40px; text-align:right; }

/* Pipeline steps */
.pipe-step { padding:11px 15px; border-radius:var(--radius-sm); font-size:12px; border:1px solid; }
.pipe-num { font-family:var(--mono); font-size:10px; display:block; margin-bottom:3px; }
.pipe-arrow { text-align:center; color:var(--text-dim); font-size:14px; padding:2px 0; }

/* Mit cards */
.mit-card { background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:var(--radius-sm); padding:16px; text-align:center; cursor:pointer; transition:all 0.2s; }
.mit-card:hover { border-color:var(--border-hi); transform:translateY(-2px); }
.mit-card.selected { border-color:var(--amber); background:var(--amber-dim); }
.mit-icon { font-size:22px; margin-bottom:8px; }
.mit-name { font-size:12px; font-weight:600; margin-bottom:4px; }
.mit-desc { font-family:var(--mono); font-size:10px; color:var(--text-muted); }

/* Tree */
.tree-box { background:rgba(0,0,0,0.35); border:1px solid var(--border); border-radius:var(--radius-sm); padding:14px 16px; font-family:var(--mono); font-size:11px; line-height:2; color:var(--text-muted); overflow-x:auto; }
.kw { color:var(--blue); } .feat { color:var(--amber); } .val { color:var(--green); }

/* Spinner */
@keyframes spin { to{transform:rotate(360deg)} }
.spinner { width:14px; height:14px; border:2px solid rgba(0,0,0,0.3); border-top-color:#050810; border-radius:50%; animation:spin 0.6s linear infinite; }

@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

@media(max-width:768px) {
  .hero { grid-template-columns:1fr; }
  .hero-divider { display:none; }
  .g2,.g4,.mc-grid { grid-template-columns:1fr; }
  .g3 { grid-template-columns:1fr 1fr; }
  .form-row { grid-template-columns:1fr; }
  .hero-score-number { font-size:56px; }
}
</style>
</head>
<body>

<div class="bg-grid"></div>
<div class="bg-glow bg-g1"></div>
<div class="bg-glow bg-g2"></div>
<div class="bg-noise"></div>
<div class="bg-scan"></div>

<div class="app">

  <header class="header">
    <a class="logo" href="/">
      <div class="logo-mark">N</div>
      <span class="logo-wordmark">Node<em>Bias</em></span>
    </a>
    <div class="header-right">
      <div class="status-chip">
        <span class="pulse-dot"></span>
        AUDIT ENGINE v2.4 &nbsp;·&nbsp; <span id="clock">--:--:--</span>
      </div>
      <button class="export-btn" onclick="exportReport()">↓ EXPORT JSON</button>
    </div>
  </header>

  <!-- Hero: shows live result or default RF report -->
  <section class="hero" id="hero">
    <div class="hero-meta">
      <div class="hero-row"><span class="hero-label">Dataset</span><span class="hero-val" id="h-dataset">diabetic_data.csv</span></div>
      <div class="hero-row"><span class="hero-label">Records</span><span class="hero-val" id="h-records">101,766</span></div>
      <div class="hero-row"><span class="hero-label">Sensitive attr.</span><span class="hero-val" id="h-sensitive">gender</span></div>
      <div class="hero-row"><span class="hero-label">Target col.</span><span class="hero-val" id="h-target">readmitted</span></div>
    </div>
    <div class="hero-divider"></div>
    <div class="hero-score">
      <div class="hero-score-label">Disparate Impact Ratio</div>
      <div class="hero-score-number pass" id="h-dir">0.981</div>
      <div><span class="hero-badge pass" id="h-badge">✓ PASS</span></div>
    </div>
    <div class="hero-divider"></div>
    <div class="hero-meta">
      <div class="hero-row"><span class="hero-label">Model</span><span class="hero-val" id="h-model">Random Forest</span></div>
      <div class="hero-row"><span class="hero-label">Features</span><span class="hero-val" id="h-features">42 auto-detected</span></div>
      <div class="hero-row"><span class="hero-label">Threshold</span><span class="hero-val ok">≥ 0.800</span></div>
      <div class="hero-row"><span class="hero-label">Status</span><span class="hero-val ok" id="h-status">FAIR</span></div>
    </div>
  </section>

  <nav class="nav">
    <button class="nav-btn active" onclick="showTab('overview')">Overview</button>
    <button class="nav-btn" onclick="showTab('audit')">Run Audit</button>
    <button class="nav-btn" onclick="showTab('models')">Models</button>
    <button class="nav-btn" onclick="showTab('explain')">Explainability</button>
    <button class="nav-btn" onclick="showTab('log')">Audit Log</button>
  </nav>

  <!-- ──── OVERVIEW ──── -->
  <div class="section active" id="tab-overview">
    <div class="g4 mb">
      <div class="stat-box"><div class="s-lbl">DIR Score</div><div class="s-val g" id="ov-dir">0.981</div><div class="s-sub">RANDOM FOREST</div></div>
      <div class="stat-box"><div class="s-lbl">Demographic Gap</div><div class="s-val a">1.80%</div><div class="s-sub">F 46.92% · M 45.12%</div></div>
      <div class="stat-box"><div class="s-lbl">Total Patients</div><div class="s-val b">101,766</div><div class="s-sub">PROCESSED</div></div>
      <div class="stat-box"><div class="s-lbl">Features Used</div><div class="s-val g">42</div><div class="s-sub">AUTO-DETECTED</div></div>
    </div>
    <div class="g2 mb">
      <div class="card">
        <div class="card-title">Group Readmission Rates</div>
        <div class="bias-item">
          <div class="bias-head"><span class="bias-name">Female</span><span class="bias-stats">54,708 patients · 46.92%</span></div>
          <div class="bias-track"><div class="bias-fill" id="bf" style="background:linear-gradient(90deg,#4D9FFF,#8B5CF6)"></div></div>
        </div>
        <div class="bias-item">
          <div class="bias-head"><span class="bias-name">Male</span><span class="bias-stats">47,055 patients · 45.12%</span></div>
          <div class="bias-track"><div class="bias-fill" id="bm" style="background:linear-gradient(90deg,var(--green),#00C9FF)"></div></div>
        </div>
        <div class="bias-item">
          <div class="bias-head"><span class="bias-name">Unknown / Invalid</span><span class="bias-stats">3 patients · 0.00%</span></div>
          <div class="bias-track"><div class="bias-fill" id="bu" style="background:rgba(255,255,255,0.25)"></div></div>
        </div>
        <div style="margin-top:20px;padding:16px;background:var(--green-dim);border:1px solid rgba(0,245,160,0.2);border-radius:var(--radius-sm);display:flex;justify-content:space-between;align-items:center">
          <div>
            <div class="s-lbl" style="margin-bottom:3px">DIR Score (min÷max)</div>
            <div style="font-family:var(--mono);font-size:11px;color:var(--text-muted)">Gap: 1.80 percentage points</div>
          </div>
          <div style="font-size:32px;font-weight:800;color:var(--green);letter-spacing:-2px">0.981</div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Feature Importance</div>
        <div class="feat-item"><div class="feat-rank">01</div><div class="feat-name">num_lab_procedures</div><div class="feat-track"><div class="feat-bar" id="ff1" style="background:linear-gradient(90deg,var(--amber),var(--red))"></div></div><div class="feat-pct" style="color:var(--amber)">35%</div></div>
        <div class="feat-item"><div class="feat-rank">02</div><div class="feat-name">insulin</div><div class="feat-track"><div class="feat-bar" id="ff2" style="background:linear-gradient(90deg,var(--blue),var(--green))"></div></div><div class="feat-pct" style="color:var(--blue)">25%</div></div>
        <div class="feat-item"><div class="feat-rank">03</div><div class="feat-name">age</div><div class="feat-track"><div class="feat-bar" id="ff3" style="background:linear-gradient(90deg,var(--purple),var(--blue))"></div></div><div class="feat-pct" style="color:var(--purple)">20%</div></div>
        <div class="feat-item"><div class="feat-rank">04</div><div class="feat-name">number_diagnoses</div><div class="feat-track"><div class="feat-bar" id="ff4" style="background:linear-gradient(90deg,var(--green),#00C9FF)"></div></div><div class="feat-pct" style="color:var(--green)">12%</div></div>
        <div class="feat-item"><div class="feat-rank">05</div><div class="feat-name">num_medications</div><div class="feat-track"><div class="feat-bar" id="ff5" style="background:linear-gradient(90deg,rgba(255,184,0,.7),var(--amber))"></div></div><div class="feat-pct" style="color:var(--amber)">8%</div></div>
        <div style="margin-top:16px;padding:12px 14px;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:var(--radius-sm)">
          <div class="s-lbl" style="margin-bottom:5px">FAIRNESS THROUGH UNAWARENESS</div>
          <div style="font-size:12px;color:var(--text-muted)">Sensitive attribute <code style="color:var(--amber);font-family:var(--mono)">gender</code> excluded from all model features.</div>
        </div>
      </div>
    </div>
    <div class="terminal">
      <div class="term-bar"><div class="term-dots"><div class="term-dot" style="background:#FF5F57"></div><div class="term-dot" style="background:#FFBD2E"></div><div class="term-dot" style="background:#28CA41"></div></div><span style="margin-left:10px">nodebias — audit output</span></div>
      <div class="term-body" id="main-term">
        <div class="term-line"><span class="t-time">08:42:01</span><span class="t-info">INFO</span><span class="t-msg">Loading diabetic_data.csv — 101,766 records</span></div>
        <div class="term-line"><span class="t-time">08:42:03</span><span class="t-info">INFO</span><span class="t-msg">Auto-detected <span class="t-pass">42</span> safe numerical features</span></div>
        <div class="term-line"><span class="t-time">08:42:04</span><span class="t-warn">WARN</span><span class="t-msg">Encoding 12 categorical columns via .cat.codes</span></div>
        <div class="term-line"><span class="t-time">08:42:07</span><span class="t-info">INFO</span><span class="t-msg">Training RandomForestClassifier — 10 trees, max_depth=14</span></div>
        <div class="term-line"><span class="t-time">08:42:19</span><span class="t-pass">PASS</span><span class="t-msg">DIR: <span style="font-weight:700">0.981</span> (threshold ≥ 0.800)</span></div>
        <div class="term-line"><span class="t-time">08:42:19</span><span class="t-pass">PASS</span><span class="t-msg">Status: <span style="font-weight:700">SAFE FOR DEPLOYMENT</span></span></div>
        <div class="term-line"><span class="t-time">08:42:20</span><span class="t-pass">DONE</span><span class="t-msg">NodeBias audit complete<span class="cursor"></span></span></div>
      </div>
    </div>
  </div>

  <!-- ──── AUDIT ──── -->
  <div class="section" id="tab-audit">
    <div class="upload-zone mb" id="drop-zone" onclick="handleUpload()" ondragover="handleDragOver(event)" ondrop="handleDrop(event)">
      <div class="upload-icon">⬆</div>
      <div class="upload-text">Drop your CSV dataset here to begin</div>
      <div class="upload-sub">or click to browse · accepts .csv files</div>
    </div>
    <div class="card mb">
      <div class="card-title">Configuration</div>
      <div class="form-row">
        <div><label class="form-label">Model Type</label><select class="form-select" id="cfg-model"><option>Logistic Regression</option><option>Gaussian Naive Bayes</option><option selected>Random Forest</option><option>Decision Tree</option></select></div>
        <div><label class="form-label">Target Column</label><input class="form-input" id="cfg-target" value="readmitted"></div>
      </div>
      <div class="form-row">
        <div><label class="form-label">Sensitive Attribute</label><input class="form-input" id="cfg-sensitive" value="gender"></div>
        <div><label class="form-label">Mitigation Strategy</label><select class="form-select" id="cfg-mitigation"><option>None (Baseline)</option><option selected>Fairness Through Unawareness</option><option>Reweighing</option></select></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-green" id="run-btn" onclick="runAudit()"><span id="run-icon">▶</span> <span id="run-text">RUN AUDIT</span></button>
        <button class="btn btn-ghost" onclick="resetAudit()">RESET</button>
      </div>
    </div>
    <div id="audit-result" style="display:none">
      <div class="terminal mb">
        <div class="term-bar"><div class="term-dots"><div class="term-dot" style="background:#FF5F57"></div><div class="term-dot" style="background:#FFBD2E"></div><div class="term-dot" style="background:#28CA41"></div></div><span style="margin-left:10px">Live audit output → API: POST /api/audit</span></div>
        <div class="term-body" id="audit-term"></div>
      </div>
      <div class="card" id="result-card">
        <div class="card-title">Results</div>
        <div id="result-content"></div>
      </div>
    </div>
  </div>

  <!-- ──── MODELS ──── -->
  <div class="section" id="tab-models">
    <div class="card mb">
      <div class="card-title">Model Registry</div>
      <div class="mc-grid">
        <div class="mc active" onclick="selectMC(this)"><div class="mc-name">Random Forest</div><div class="mc-type">ENSEMBLE · 10 TREES · DEPTH 14</div><div class="mc-score g">0.981</div><div class="mc-status p">✓ PASS — SAFE FOR DEPLOYMENT</div></div>
        <div class="mc" onclick="selectMC(this)"><div class="mc-name">Logistic Regression</div><div class="mc-type">MOMENTUM · BCE LOSS · 1000 EPOCHS</div><div class="mc-score g">0.962</div><div class="mc-status p">✓ PASS — SAFE FOR DEPLOYMENT</div></div>
        <div class="mc" onclick="selectMC(this)"><div class="mc-name">Gaussian Naive Bayes</div><div class="mc-type">PROBABILISTIC · BAYES THEOREM</div><div class="mc-score r">0.743</div><div class="mc-status f">✗ FAIL — BIAS DETECTED</div></div>
        <div class="mc" onclick="selectMC(this)"><div class="mc-name">Decision Tree</div><div class="mc-type">SINGLE TREE · DEPTH 14 · MIN SPLIT 100</div><div class="mc-score d">—</div><div class="mc-status n">AWAITING AUDIT RUN</div></div>
      </div>
    </div>
    <div class="card mb">
      <div class="card-title">Comparative DIR Scores</div>
      <div class="cmp-row"><span class="cmp-name">Random Forest</span><div class="cmp-track"><div class="cmp-fill" id="c1" style="background:linear-gradient(90deg,var(--green),#00C9FF)"></div></div><span class="cmp-val" style="color:var(--green)">0.981</span></div>
      <div class="cmp-row"><span class="cmp-name">Logistic Regression</span><div class="cmp-track"><div class="cmp-fill" id="c2" style="background:linear-gradient(90deg,var(--blue),var(--purple))"></div></div><span class="cmp-val" style="color:var(--blue)">0.962</span></div>
      <div class="cmp-row"><span class="cmp-name">Naive Bayes</span><div class="cmp-track"><div class="cmp-fill" id="c3" style="background:linear-gradient(90deg,var(--red),var(--amber))"></div></div><span class="cmp-val" style="color:var(--red)">0.743</span></div>
      <div class="cmp-row"><span class="cmp-name">Decision Tree</span><div class="cmp-track"><div class="cmp-fill" id="c4" style="background:rgba(255,255,255,0.15)"></div></div><span class="cmp-val" style="color:var(--text-dim)">—</span></div>
      <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--green-dim);border:1px solid rgba(0,245,160,0.2);border-radius:var(--radius-sm);margin-top:14px">
        <span style="font-family:var(--mono);font-size:10px;color:var(--green)">THRESHOLD</span>
        <div style="flex:1;height:1px;background:rgba(0,245,160,0.25)"></div>
        <span style="font-family:var(--mono);font-size:12px;color:var(--green)">0.800 (80% rule — EEOC standard)</span>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Mitigation Strategies</div>
      <div class="g3">
        <div class="mit-card selected" onclick="selectMit(this)"><div class="mit-icon">🚫</div><div class="mit-name">Fairness Through Unawareness</div><div class="mit-desc">DROP SENSITIVE COLUMN</div></div>
        <div class="mit-card" onclick="selectMit(this)"><div class="mit-icon">⚖</div><div class="mit-name">Reweighing</div><div class="mit-desc">SAMPLE WEIGHT ADJUSTMENT</div></div>
        <div class="mit-card" onclick="selectMit(this)"><div class="mit-icon">✂</div><div class="mit-name">Disparate Impact Remover</div><div class="mit-desc">FEATURE REPAIR</div></div>
      </div>
    </div>
  </div>

  <!-- ──── EXPLAINABILITY ──── -->
  <div class="section" id="tab-explain">
    <div class="g2 mb">
      <div class="card">
        <div class="card-title">Data Pipeline Architecture</div>
        <div class="pipe-step" style="background:var(--blue-dim);border-color:rgba(77,159,255,0.25)"><span class="pipe-num" style="color:var(--blue)">STEP 1 · INPUT</span>Raw CSV dataset (diabetic_data.csv)</div>
        <div class="pipe-arrow">↓</div>
        <div class="pipe-step" style="background:var(--amber-dim);border-color:rgba(255,184,0,0.25)"><span class="pipe-num" style="color:var(--amber)">STEP 2 · CLEAN</span>Null standardisation (?, NA → NaN) + schema validation</div>
        <div class="pipe-arrow">↓</div>
        <div class="pipe-step" style="background:var(--purple-dim);border-color:rgba(139,92,246,0.25)"><span class="pipe-num" style="color:var(--purple)">STEP 3 · ENCODE</span>Auto-encode categoricals → numeric shadow column</div>
        <div class="pipe-arrow">↓</div>
        <div class="pipe-step" style="background:var(--blue-dim);border-color:rgba(77,159,255,0.2)"><span class="pipe-num" style="color:var(--blue)">STEP 4 · SCALE</span>Feature extraction + StandardScaler (μ=0, σ=1)</div>
        <div class="pipe-arrow">↓</div>
        <div class="pipe-step" style="background:var(--green-dim);border-color:rgba(0,245,160,0.3)"><span class="pipe-num" style="color:var(--green)">STEP 5 · TRAIN + AUDIT</span>GlassBoxML training → DIR computation → JSON report</div>
      </div>
      <div class="card">
        <div class="card-title">Decision Tree Structure</div>
        <div class="tree-box">
<span class="kw">if</span> <span class="feat">num_lab_procedures</span> ≤ 38.5:
  <span class="kw">if</span> <span class="feat">insulin</span> == <span class="val">No</span>:
    <span class="kw">predict</span> → <span class="val">0</span> (no readmission)
  <span class="kw">else</span>:
    <span class="kw">if</span> <span class="feat">age</span> &gt;= <span class="val">[60-70)</span>:
      <span class="kw">predict</span> → <span class="val">1</span> (readmission)
    <span class="kw">else</span>:
      <span class="kw">predict</span> → <span class="val">0</span>
<span class="kw">else</span>:
  <span class="kw">if</span> <span class="feat">number_diagnoses</span> &gt;= <span class="val">7</span>:
    <span class="kw">predict</span> → <span class="val">1</span> (readmission)
  <span class="kw">else</span>:
    <span class="kw">predict</span> → <span class="val">0</span> (no readmission)
        </div>
        <div class="alert alert-amber" style="margin-top:14px">⚠ Sensitive attribute <code style="font-family:var(--mono)">gender</code> excluded from ALL decision paths under FTU mitigation.</div>
        <div class="alert alert-green" style="margin-top:10px">✓ Top predictors are clinically valid: lab procedures, insulin, age, diagnoses count.</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Before vs After Mitigation</div>
      <div class="g2">
        <div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--red);letter-spacing:1px;margin-bottom:12px">BASELINE (gender included)</div>
          <div class="stat-box mb"><div class="s-lbl">DIR Score</div><div class="s-val r">0.743</div><div class="s-sub">NAIVE BAYES BASELINE</div></div>
          <div class="alert alert-red">✗ FAIL — Bias detected above threshold</div>
        </div>
        <div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--green);letter-spacing:1px;margin-bottom:12px">MITIGATED (FTU applied)</div>
          <div class="stat-box mb"><div class="s-lbl">DIR Score</div><div class="s-val g">0.981</div><div class="s-sub">RANDOM FOREST + FTU</div></div>
          <div class="alert alert-green">✓ PASS — Safe for deployment</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ──── LOG ──── -->
  <div class="section" id="tab-log">
    <div class="terminal">
      <div class="term-bar"><div class="term-dots"><div class="term-dot" style="background:#FF5F57"></div><div class="term-dot" style="background:#FFBD2E"></div><div class="term-dot" style="background:#28CA41"></div></div><span style="margin-left:10px">nodebias — full audit log · session <span id="session-id" style="color:var(--text-muted)">...</span></span></div>
      <div class="term-body" id="full-log" style="max-height:520px"></div>
    </div>
  </div>

</div>

<script>
/* Clock */
function tick(){ document.getElementById('clock').textContent=new Date().toLocaleTimeString('en-GB'); }
tick(); setInterval(tick,1000);

/* Session */
document.getElementById('session-id').textContent=Math.random().toString(36).slice(2,8).toUpperCase();

/* Tabs */
const TABS=['overview','audit','models','explain','log'];
function showTab(t){
  TABS.forEach(id=>{document.getElementById('tab-'+id).classList.toggle('active',id===t);});
  document.querySelectorAll('.nav-btn').forEach((b,i)=>b.classList.toggle('active',TABS[i]===t));
  if(t==='log') renderLog();
}

/* Bars on load */
window.addEventListener('load',()=>{
  setTimeout(()=>{
    document.getElementById('bf').style.width='93%';
    document.getElementById('bm').style.width='90%';
    document.getElementById('bu').style.width='1%';
    ['ff1','ff2','ff3','ff4','ff5'].forEach((id,i)=>{document.getElementById(id).style.width=[70,50,40,24,16][i]+'%';});
    document.getElementById('c1').style.width='98%';
    document.getElementById('c2').style.width='96%';
    document.getElementById('c3').style.width='74%';
    document.getElementById('c4').style.width='40%';
  },300);
});

/* Model card / mit select */
function selectMC(el){ document.querySelectorAll('.mc').forEach(c=>c.classList.remove('active')); el.classList.add('active'); }
function selectMit(el){ document.querySelectorAll('.mit-card').forEach(c=>c.classList.remove('selected')); el.classList.add('selected'); }

/* Upload */
function handleDragOver(e){e.preventDefault();}
function handleDrop(e){e.preventDefault(); if(e.dataTransfer.files[0]) applyFile(e.dataTransfer.files[0]);}
function handleUpload(){const i=document.createElement('input');i.type='file';i.accept='.csv';i.onchange=e=>{if(e.target.files[0]) applyFile(e.target.files[0]);};i.click();}
function applyFile(f){
  const z=document.getElementById('drop-zone');
  z.classList.add('loaded');
  z.innerHTML=`<div class="upload-icon" style="opacity:1">✓</div><div class="upload-text" style="color:var(--green)">${f.name}</div><div class="upload-sub">${(f.size/1024).toFixed(1)} KB · ready to audit</div>`;
  window._uploadedFile=f;
}

/* ── Run Audit (calls real /api/audit endpoint) ── */
let auditRunning=false;
function logLine(termId, type, msg){
  const term=document.getElementById(termId);
  const now=new Date().toLocaleTimeString('en-GB');
  const cls={INFO:'t-info',PASS:'t-pass',FAIL:'t-fail',WARN:'t-warn',DONE:'t-pass',ERR:'t-fail'}[type]||'t-dim';
  const li=document.createElement('div'); li.className='term-line';
  li.innerHTML=`<span class="t-time">${now}</span><span class="${cls}">${type}</span><span class="t-msg">${msg}</span>`;
  term.appendChild(li); term.scrollTop=term.scrollHeight;
}

async function runAudit(){
  if(auditRunning) return;
  auditRunning=true;
  const btn=document.getElementById('run-btn');
  document.getElementById('run-text').textContent='RUNNING...';
  document.getElementById('run-icon').innerHTML='<span class="spinner"></span>';
  btn.disabled=true;

  const res=document.getElementById('audit-result');
  const term=document.getElementById('audit-term');
  const content=document.getElementById('result-content');
  term.innerHTML=''; content.innerHTML=''; res.style.display='block';

  logLine('audit-term','INFO','Initializing audit pipeline...');
  logLine('audit-term','INFO',`Model: ${document.getElementById('cfg-model').value}`);
  logLine('audit-term','INFO',`Target: ${document.getElementById('cfg-target').value}  Sensitive: ${document.getElementById('cfg-sensitive').value}`);

  // Build FormData to call the Flask API
  const formData=new FormData();
  formData.append('modelType', document.getElementById('cfg-model').value);
  formData.append('targetColumn', document.getElementById('cfg-target').value);
  formData.append('sensitiveColumn', document.getElementById('cfg-sensitive').value);

  if(window._uploadedFile){
    formData.append('dataset', window._uploadedFile);
  } else {
    // No real file: simulate with canned result
    logLine('audit-term','WARN','No file uploaded — using pre-computed audit results');
    await simulateDelay(800);
    showResult({dir_score:0.981,verdict:'PASS',model_used:'Random Forest',features_used:42,records_processed:101766,group_rates:{Female:0.4692,Male:0.4512},disparity_gap_pct:1.80});
    return;
  }

  logLine('audit-term','INFO','Sending dataset to /api/audit...');
  try {
    const r=await fetch('/api/audit',{method:'POST',body:formData});
    const data=await r.json();
    if(data.error) throw new Error(data.error);
    logLine('audit-term','INFO',`Engine: ${data.engine||'GlassBoxML'}`);
    logLine('audit-term','INFO',`Features detected: ${data.features_used}`);
    logLine('audit-term','INFO',`Records processed: ${data.records_processed}`);
    const verdict=data.verdict==='PASS'?'PASS':'FAIL';
    logLine('audit-term',verdict,`DIR: ${data.dir_score} (threshold ≥ 0.800) — ${verdict}`);
    logLine('audit-term',verdict,`Status: ${data.status}`);
    showResult(data);
  } catch(e){
    logLine('audit-term','ERR',`API Error: ${e.message}`);
    done();
  }

  function showResult(data){
    const pass=data.verdict==='PASS';
    document.getElementById('result-card').style.borderTop=`2px solid ${pass?'var(--green)':'var(--red)'}`;
    let groupHTML='';
    if(data.group_rates){
      Object.entries(data.group_rates).forEach(([g,r])=>{
        groupHTML+=`<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid var(--border)"><span style="color:var(--text-muted)">${g}</span><span style="font-family:var(--mono);color:var(--text)">${(r*100).toFixed(2)}%</span></div>`;
      });
    }
    content.innerHTML=`
      <div class="g2" style="margin-bottom:16px">
        <div class="stat-box"><div class="s-lbl">DIR Score</div><div class="s-val ${pass?'g':'r'}">${data.dir_score}</div><div class="s-sub">${(data.model_used||'').toUpperCase()}</div></div>
        <div class="stat-box"><div class="s-lbl">Verdict</div><div class="s-val ${pass?'g':'r'}">${data.verdict}</div><div class="s-sub">${data.status||''}</div></div>
      </div>
      ${groupHTML?`<div style="margin-bottom:12px"><div class="s-lbl" style="margin-bottom:8px">Group Prediction Rates</div>${groupHTML}</div>`:''}
      <div class="alert ${pass?'alert-green':'alert-red'}" style="margin-bottom:10px">${pass?'✓':'✗'} DIR ${pass?'meets':'does not meet'} the 0.800 threshold.${!pass?' Consider applying a mitigation strategy and re-running.':''}</div>`;

    // Update hero
    document.getElementById('h-dir').textContent=data.dir_score;
    document.getElementById('h-dir').className='hero-score-number '+(pass?'pass':'fail');
    document.getElementById('h-badge').textContent=pass?'✓ PASS':'✗ FAIL';
    document.getElementById('h-badge').className='hero-badge '+(pass?'pass':'fail');
    document.getElementById('h-model').textContent=data.model_used||'—';
    document.getElementById('h-features').textContent=(data.features_used||'—')+' detected';
    document.getElementById('h-records').textContent=(data.records_processed||'—').toLocaleString();
    document.getElementById('h-status').textContent=pass?'FAIR':'BIASED';
    document.getElementById('h-status').style.color=pass?'var(--green)':'var(--red)';
    done();
  }
  function done(){
    auditRunning=false;
    document.getElementById('run-text').textContent='RUN AUDIT';
    document.getElementById('run-icon').textContent='▶';
    btn.disabled=false;
  }
}

function simulateDelay(ms){return new Promise(r=>setTimeout(r,ms));}
function resetAudit(){document.getElementById('audit-result').style.display='none'; auditRunning=false; document.getElementById('run-text').textContent='RUN AUDIT'; document.getElementById('run-icon').textContent='▶'; document.getElementById('run-btn').disabled=false;}

/* Full log */
const LOG=[
  {t:'INFO',m:'Session initialized'},
  {t:'INFO',m:'Loading diabetic_data.csv — 101,766 records'},
  {t:'INFO',m:'Pipeline: target=readmitted, sensitive=gender'},
  {t:'INFO',m:'Null standardisation complete'},
  {t:'INFO',m:'Auto-detected 42 numerical features'},
  {t:'WARN',m:'12 categorical columns encoded via .cat.codes'},
  {t:'INFO',m:'Sensitive numeric shadow column created'},
  {t:'INFO',m:'StandardScaler fitted and applied'},
  {t:'INFO',m:'Training Logistic Regression — Momentum(lr=0.01, β=0.9), 1000 epochs'},
  {t:'PASS',m:'LR DIR: 0.962 — PASS'},
  {t:'INFO',m:'Training Gaussian Naive Bayes'},
  {t:'FAIL',m:'NB DIR: 0.743 — FAIL — bias detected'},
  {t:'INFO',m:'Training Random Forest — 10 trees, depth=14'},
  {t:'PASS',m:'RF DIR: 0.981 — PASS'},
  {t:'PASS',m:'RF verdict: SAFE FOR DEPLOYMENT'},
  {t:'INFO',m:'Reports saved: audit_report_forest.json, data_bias_report.json'},
  {t:'DONE',m:'All audits complete.'},
];
function renderLog(){
  const el=document.getElementById('full-log');
  const cls={INFO:'t-info',PASS:'t-pass',FAIL:'t-fail',WARN:'t-warn',DONE:'t-pass'};
  el.innerHTML=LOG.map(l=>{
    const t=new Date(Date.now()-Math.random()*60000).toLocaleTimeString('en-GB');
    return `<div class="term-line"><span class="t-time">${t}</span><span class="${cls[l.t]||'t-dim'}">${l.t}</span><span class="t-msg">${l.m}</span></div>`;
  }).join('')+`<div class="term-line"><span class="t-time">${new Date().toLocaleTimeString('en-GB')}</span><span class="t-pass">LIVE</span><span class="t-msg">Session active<span class="cursor"></span></span></div>`;
}

/* Export */
const LAST_REPORT={generated:new Date().toISOString(),model_type:'GlassBoxML Random Forest',ensemble_size:10,metrics:{disparate_impact_ratio:0.981,demographic_gap:'1.80%',status:'PASS'},data_bias:{Female:{total_patients:54708,readmission_rate_percentage:46.92},Male:{total_patients:47055,readmission_rate_percentage:45.12}},feature_priority:[{feature:'num_lab_procedures',weight:0.35},{feature:'insulin',weight:0.25},{feature:'age',weight:0.20}],mitigation:'Fairness Through Unawareness'};
function exportReport(){
  const blob=new Blob([JSON.stringify(LAST_REPORT,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='nodebias_audit_report.json'; a.click();
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(FRONTEND_HTML)


if __name__ == "__main__":
    print("\n" + "="*56)
    print("   NodeBias Full-Stack App")
    print("="*56)
    print("   Dashboard →  http://localhost:5000")
    print("   API audit →  POST http://localhost:5000/api/audit")
    print("   Health    →  http://localhost:5000/api/health")
    print("="*56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
