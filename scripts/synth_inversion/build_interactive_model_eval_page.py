import json
import os
import sys
import numpy as np

eval_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/kim/sao_runs/v5_multi_note_consensus_eval_v2"
run_id = os.path.basename(os.path.normpath(eval_dir))
listen_dir = os.path.join(eval_dir, "listen")
summary_file = os.path.join(eval_dir, "consensus_eval_summary.json")

if not os.path.exists(summary_file):
    print(f"Error: summary file not found at {summary_file}")
    sys.exit(1)

with open(summary_file) as f:
    results = json.load(f)

models = list(results[0]["models"].keys())

# Compute aggregate stats for leaderboard
stats = {}
for m in models:
    mss_list = [r["models"][m]["phrase_mss"] for r in results if m in r["models"]]
    ssim_list = [r["models"][m]["mel_ssim"] for r in results if m in r["models"]]
    cent_deltas = [abs(r["models"][m]["centroid_delta_hz"]) for r in results if m in r["models"]]
    width_deltas = [abs(r["models"][m].get("width_delta", 0.0)) for r in results if m in r["models"]]
    stats[m] = {
        "mean_mss": float(np.mean(mss_list)),
        "median_mss": float(np.median(mss_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "mean_cent_err": float(np.mean(cent_deltas)),
        "mean_w_err": float(np.mean(width_deltas)),
        "wins": 0
    }

for r in results:
    best_m = min(models, key=lambda m: r["models"][m]["phrase_mss"])
    stats[best_m]["wins"] += 1

total_tracks = len(results)

# Build HTML rows
rows = []
for idx, r in enumerate(results):
    stem_id = r["stem_id"]
    name = r["name"]
    style = r["style"]
    bpm = r["bpm"]
    root = r["root_midi"]
    real_cent = r["real_centroid_hz"]
    real_w = r.get("real_width", 0.0)
    real_stereo = r.get("real_stereo_class", "Mono")
    ref_audio = f"{stem_id}__real_phrase.wav"

    best_m = min(models, key=lambda m: r["models"][m]["phrase_mss"])
    
    # JSON bundle of audio sources for synchronized player
    audio_map = {"Real": ref_audio}
    for m in models:
        audio_map[m] = r["models"][m]["audio_file"]
    audio_map_json = json.dumps(audio_map).replace('"', '&quot;')

    model_cells = []
    for m in models:
        d = r["models"][m]
        mss = d["phrase_mss"]
        ssim = d["mel_ssim"]
        cent = d["centroid_hz"]
        d_cent = d["centroid_delta_hz"]
        w_val = d.get("stereo_width", 0.0)
        dw_val = d.get("width_delta", 0.0)
        retrig = d.get("retrigger", "On")
        winner = (m == best_m)
        w_type = d.get("winner_type", "Consensus")
        patch_info = d.get("patch_summary", "")

        card_class = "model-card winner" if winner else "model-card"
        badge_class = "badge badge-win" if winner else "badge"

        cell = f"""
        <div class="{card_class}" data-model="{m}" id="card-{stem_id}-{m}">
            <div class="card-header">
                <span class="model-name">{m}</span>
                <span class="{badge_class}">MSS: {mss:.3f}</span>
            </div>
            <div class="card-audio">
                <audio controls preload="none" src="{d['audio_file']}"></audio>
            </div>
            <div class="card-metrics">
                <span>SSIM: <strong>{ssim:.3f}</strong></span> &bull; 
                <span>Cent: <strong>{cent:.0f} Hz</strong> ({d_cent:+.0f})</span> &bull;
                <span>W: <strong>{w_val:.2f}</strong> ({dw_val:+.2f})</span>
            </div>
            <div class="card-patch">
                <span class="source-tag">{w_type}</span>
                <span class="source-tag" style="color: {'#3fb950' if retrig == 'On' else '#f85149'};">Retrig: {retrig}</span>
                <span class="patch-text">{patch_info}</span>
            </div>
            <div class="user-rank-badge" id="badge-{stem_id}-{m}"></div>
        </div>
        """
        model_cells.append(cell)

    # Build ranking pills for each model
    rate_pills = []
    for m in models:
        num_ranks = min(len(models), 5)
        buttons = "".join(f'<button class="btn-rank" data-rank="{rk}" onclick="setPreference(\'{stem_id}\', \'{m}\', {rk})">{rk}</button>' for rk in range(1, num_ranks + 1))
        rate_pills.append(f"""
        <div class="model-rate-pill" data-model="{m}">
            <span class="rate-mname">{m}:</span>
            <div class="rate-btn-group">
                {buttons}
                <button class="btn-rank btn-clear" title="Clear rank" onclick="clearPreference(\'{stem_id}\', \'{m}\')">&times;</button>
            </div>
        </div>
        """)

    row_html = f"""
    <div class="track-row" data-stem="{stem_id}" data-style="{style}" data-winner="{best_m}">
        <div class="track-header">
            <div class="track-title-box">
                <span class="track-num">#{idx+1:02d}</span>
                <h3 class="track-title">{name}</h3>
                <span class="tag-style">{style}</span>
                <span class="meta-tag">{bpm} BPM</span>
                <span class="meta-tag">Root: {root}</span>
                <span class="meta-tag">Centroid: {real_cent:.0f} Hz</span>
                <span class="meta-tag" style="color: {'#58a6ff' if real_stereo != 'Pure Mono' else '#8b949e'}; font-weight: 600;">{real_stereo} (W={real_w:.2f})</span>
            </div>
            <div class="sync-player-widget" data-audio-map="{audio_map_json}">
                <div class="sync-controls">
                    <button class="play-btn" onclick="toggleSyncPlay(this)">&#9658; Play Synchronized</button>
                    <span class="time-display">0:00 / 0:00</span>
                    <input type="range" class="sync-scrub" min="0" max="100" value="0" step="0.1" oninput="seekSync(this)">
                </div>
                <div class="sync-switch-tabs">
                    <button class="switch-tab active" onclick="switchSyncSource(this, 'Real')">Real Ground Truth</button>
                    {"".join(f'<button class="switch-tab {"best-tab" if m == best_m else ""}" onclick="switchSyncSource(this, \'{m}\')">{m} {"&#9733;" if m == best_m else ""}</button>' for m in models)}
                </div>
                <div class="user-rank-widget" data-stem="{stem_id}">
                    <span class="rank-title">Rate Preference:</span>
                    {"".join(rate_pills)}
                    <span class="rank-summary-pill" id="rank-summary-{stem_id}">Unrated</span>
                </div>
            </div>
        </div>
        <div class="models-grid">
            <div class="model-card ref-card">
                <div class="card-header">
                    <span class="model-name" style="color: #58a6ff;">Ground Truth Real</span>
                    <span class="badge" style="background: #388bfd33; color: #58a6ff;">Reference</span>
                </div>
                <div class="card-audio">
                    <audio controls preload="none" src="{ref_audio}"></audio>
                </div>
                <div class="card-metrics">
                    <span>Centroid: <strong>{real_cent:.0f} Hz</strong></span> &bull; 
                    <span>Stereo Width: <strong>{real_w:.2f} ({real_stereo})</strong></span>
                </div>
                <div class="card-patch">
                    <span class="source-tag" style="color: #58a6ff; border-color: #388bfd66;">Original Stem</span>
                    <span class="patch-text">Unprocessed studio master audio</span>
                </div>
            </div>
            {"".join(model_cells)}
        </div>
    </div>
    """
    rows.append(row_html)

# Leaderboard table rows
leaderboard_rows = []
sorted_models = sorted(models, key=lambda m: stats[m]["mean_mss"])
for m in sorted_models:
    st = stats[m]
    win_pct = (st["wins"] / total_tracks) * 100
    leaderboard_rows.append(f"""
    <tr>
        <td style="font-weight: 600;">{m}</td>
        <td><span class="badge badge-win" style="font-size: 13px;">{st['wins']} / {total_tracks} ({win_pct:.1f}%)</span></td>
        <td><strong>{st['mean_mss']:.3f}</strong></td>
        <td>{st['median_mss']:.3f}</td>
        <td>{st['mean_ssim']:.3f}</td>
        <td>{st['mean_cent_err']:.0f} Hz</td>
        <td>{st['mean_w_err']:.2f}</td>
    </tr>
    """)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>V5 / V6 Multi-Note Consensus Evaluation &amp; Human Preference Rating</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --surface-hover: #1c2128;
            --border: #30363d;
            --text: #c9d1d9;
            --accent: #58a6ff;
            --win: #3fb950;
            --gold: #d29922;
        }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 24px 32px;
            font-size: 14px;
        }}
        h1 {{
            color: var(--accent);
            margin: 0 0 6px 0;
            font-size: 26px;
        }}
        .subtitle {{
            color: #8b949e;
            font-size: 14px;
            margin: 0 0 16px 0;
        }}
        .summary-banner {{
            display: flex;
            gap: 20px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        .banner-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            flex: 1;
            min-width: 280px;
        }}
        .preference-card {{
            background: #161b22;
            border: 1px solid var(--gold);
            box-shadow: 0 0 12px rgba(210, 153, 34, 0.15);
        }}
        .banner-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8b949e;
            margin-bottom: 8px;
        }}
        table.leaderboard {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        table.leaderboard th {{
            text-align: left;
            padding: 6px 8px;
            color: #8b949e;
            border-bottom: 1px solid var(--border);
        }}
        table.leaderboard td {{
            padding: 8px;
            border-bottom: 1px solid #21262d;
        }}
        .filter-bar {{
            display: flex;
            gap: 10px;
            align-items: center;
            background: var(--surface);
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        .search-box {{
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            padding: 6px 12px;
            font-size: 13px;
            width: 240px;
        }}
        .filter-btn {{
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 5px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }}
        .filter-btn.active {{
            background: var(--accent);
            color: #0d1117;
            border-color: var(--accent);
            font-weight: 600;
        }}
        .track-row {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            margin-bottom: 24px;
            padding: 16px;
            transition: border-color 0.15s ease;
        }}
        .track-row:hover {{
            border-color: #58a6ff55;
        }}
        .track-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid #21262d;
            padding-bottom: 14px;
            margin-bottom: 14px;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .track-title-box {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .track-num {{
            background: #30363d;
            color: #8b949e;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .track-title {{
            margin: 0;
            font-size: 17px;
            color: #f0f6fc;
        }}
        .tag-style {{
            background: #23863622;
            color: #3fb950;
            border: 1px solid #23863655;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }}
        .meta-tag {{
            background: #21262d;
            color: #8b949e;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
        }}
        .sync-player-widget {{
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 440px;
        }}
        .sync-controls {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .play-btn {{
            background: var(--accent);
            color: #0d1117;
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: bold;
            cursor: pointer;
            transition: opacity 0.15s ease;
        }}
        .play-btn:hover {{
            opacity: 0.9;
        }}
        .play-btn.playing {{
            background: #f85149;
            color: white;
        }}
        .time-display {{
            font-family: monospace;
            font-size: 12px;
            color: #8b949e;
            min-width: 85px;
        }}
        .sync-scrub {{
            flex: 1;
            accent-color: var(--accent);
            cursor: pointer;
        }}
        .sync-switch-tabs {{
            display: flex;
            gap: 6px;
            overflow-x: auto;
            padding-bottom: 2px;
            flex-wrap: wrap;
        }}
        .switch-tab {{
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }}
        .switch-tab:hover {{
            background: #30363d;
        }}
        .switch-tab.active {{
            background: var(--accent);
            color: #0d1117;
            font-weight: bold;
            border-color: var(--accent);
        }}
        .switch-tab.best-tab {{
            border-color: #3fb95088;
        }}
        .user-rank-widget {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: #13171f;
            border: 1px solid #30363d;
            padding: 6px 10px;
            border-radius: 6px;
            margin-top: 4px;
            flex-wrap: wrap;
        }}
        .rank-title {{
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--gold);
        }}
        .model-rate-pill {{
            display: flex;
            align-items: center;
            gap: 4px;
            background: #1c2128;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #2d333b;
        }}
        .rate-mname {{
            font-size: 11px;
            font-weight: 600;
            color: #c9d1d9;
        }}
        .rate-btn-group {{
            display: flex;
            gap: 2px;
        }}
        .btn-rank {{
            background: #21262d;
            color: #8b949e;
            border: 1px solid #30363d;
            border-radius: 3px;
            padding: 1px 6px;
            font-size: 11px;
            cursor: pointer;
            font-weight: 600;
        }}
        .btn-rank:hover {{
            background: #30363d;
            color: #f0f6fc;
        }}
        .btn-rank.selected-1 {{
            background: var(--gold);
            color: #0d1117;
            border-color: var(--gold);
        }}
        .btn-rank.selected-2 {{
            background: var(--accent);
            color: #0d1117;
            border-color: var(--accent);
        }}
        .btn-rank.selected-3 {{
            background: #a371f7;
            color: #0d1117;
            border-color: #a371f7;
        }}
        .btn-clear {{
            background: transparent;
            border: none;
            color: #6e7681;
            padding: 0 4px;
            font-size: 13px;
            cursor: pointer;
        }}
        .btn-clear:hover {{
            color: #f85149;
        }}
        .rank-summary-pill {{
            font-size: 11px;
            color: #8b949e;
            font-style: italic;
            margin-left: auto;
        }}
        .user-rank-badge {{
            margin-top: 6px;
            font-size: 11px;
            font-weight: bold;
        }}
        .models-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
        }}
        .model-card {{
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            position: relative;
        }}
        .model-card.winner {{
            border-color: var(--win);
            background: #1c2e22;
        }}
        .model-card.ref-card {{
            border-color: #388bfd66;
            background: #101a2c;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .model-name {{
            font-weight: 600;
            font-size: 13px;
        }}
        .badge {{
            background: #21262d;
            color: #8b949e;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-family: monospace;
        }}
        .badge-win {{
            background: #23863633;
            color: var(--win);
            border: 1px solid #23863666;
            font-weight: bold;
        }}
        .card-audio audio {{
            width: 100%;
            height: 32px;
            border-radius: 4px;
        }}
        .card-metrics {{
            font-size: 11px;
            color: #8b949e;
            line-height: 1.4;
        }}
        .card-metrics strong {{
            color: var(--text);
        }}
        .card-patch {{
            font-size: 10px;
            color: #8b949e;
            background: #161b22;
            padding: 6px;
            border-radius: 4px;
            border: 1px solid #21262d;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .source-tag {{
            font-size: 9px;
            padding: 1px 4px;
            border-radius: 3px;
            background: #21262d;
            color: #c9d1d9;
            align-self: flex-start;
            font-family: monospace;
        }}
        .patch-text {{
            font-family: monospace;
            word-break: break-all;
            color: #7ee787;
        }}
        .pref-stats {{
            font-size: 16px;
            font-weight: bold;
            color: var(--gold);
            margin-bottom: 8px;
        }}
        .pref-model-wins {{
            font-size: 12px;
            color: #c9d1d9;
            margin-bottom: 12px;
            line-height: 1.5;
        }}
        .pref-actions {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .pref-btn {{
            background: #21262d;
            color: var(--text);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        }}
        .pref-btn:hover {{
            background: #30363d;
        }}
        .pref-btn-gold {{
            background: var(--gold);
            color: #0d1117;
            font-weight: bold;
            border-color: var(--gold);
        }}
    </style>
</head>
<body>
    <h1>Multi-Note Consensus Evaluation &amp; Human Preference Rating</h1>
    <p class="subtitle">Multi-Note Consensus Inversion | Audio A/B Audition | Interactive Preference Ranking</p>
    
    <div class="summary-banner">
        <div class="banner-card">
            <div class="banner-title">Consensus Leaderboard (Phrase MSS)</div>
            <table class="leaderboard">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Wins</th>
                        <th>Mean MSS</th>
                        <th>Median MSS</th>
                        <th>Mel SSIM</th>
                        <th>Centroid Err</th>
                        <th>Width Err</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(leaderboard_rows)}
                </tbody>
            </table>
        </div>
        <div class="banner-card preference-card">
            <div class="banner-title">Your Human Preference Ratings</div>
            <div class="pref-stats" id="prefSummaryText">0 / {total_tracks} Rated</div>
            <div class="pref-model-wins" id="prefWinsBar">Rate clips below to build human preference scores.</div>
            <div class="pref-actions">
                <button class="pref-btn pref-btn-gold" onclick="exportPreferences()">Export Ratings JSON</button>
                <button class="pref-btn" onclick="importPreferences()">Import Ratings JSON</button>
                <button class="pref-btn" onclick="resetAllPreferences()">Reset Ratings</button>
            </div>
        </div>
    </div>

    <div class="filter-bar">
        <input type="text" class="search-box" placeholder="Filter stems by title or style..." oninput="filterStems(this.value)">
        <button class="filter-btn active" onclick="setFilter(this, 'all')">All Stems ({total_tracks})</button>
        {"".join(f'<button class="filter-btn" onclick="setFilter(this, \'{m}\')">{m} Wins ({stats[m]["wins"]})</button>' for m in models)}
    </div>

    <div id="stemsContainer">
        {"".join(rows)}
    </div>

    <script>
        const MODELS = {json.dumps(models)};
        const TOTAL_TRACKS = {total_tracks};
        const RUN_ID = "{run_id}";
        const STORAGE_KEY = "sao_stem_ratings_" + RUN_ID;

        // Load preferences from localStorage (scoped per evaluation run)
        function loadStoredRatings() {{
            try {{
                let s = localStorage.getItem(STORAGE_KEY);
                // Migrate legacy un-namespaced ratings specifically if this is stems_v6_occam
                if (!s && RUN_ID === "stems_v6_occam") {{
                    s = localStorage.getItem("sao_stem_ratings");
                    if (s) {{
                        localStorage.setItem(STORAGE_KEY, s);
                    }}
                }}
                return s ? JSON.parse(s) : {{}};
            }} catch(e) {{
                return {{}};
            }}
        }}

        function saveStoredRatings(ratings) {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(ratings));
            updatePreferenceDashboard();
            updateAllRatingPills();
        }}

        function setPreference(stemId, modelName, rank) {{
            const ratings = loadStoredRatings();
            if (!ratings[stemId]) ratings[stemId] = {{}};
            
            // If another model had this rank, swap or clear it
            for (const m in ratings[stemId]) {{
                if (ratings[stemId][m] === rank && m !== modelName) {{
                    delete ratings[stemId][m];
                }}
            }}
            ratings[stemId][modelName] = rank;
            saveStoredRatings(ratings);
        }}

        function clearPreference(stemId, modelName) {{
            const ratings = loadStoredRatings();
            if (ratings[stemId] && ratings[stemId][modelName]) {{
                delete ratings[stemId][modelName];
                if (Object.keys(ratings[stemId]).length === 0) {{
                    delete ratings[stemId];
                }}
                saveStoredRatings(ratings);
            }}
        }}

        function updateAllRatingPills() {{
            const ratings = loadStoredRatings();
            
            document.querySelectorAll('.track-row').forEach(row => {{
                const stemId = row.dataset.stem;
                const stemRatings = ratings[stemId] || {{}};
                
                // Update rank buttons inside the player widget
                row.querySelectorAll('.model-rate-pill').forEach(pill => {{
                    const m = pill.dataset.model;
                    const rk = stemRatings[m];
                    pill.querySelectorAll('.btn-rank').forEach(btn => {{
                        btn.className = "btn-rank" + (btn.classList.contains("btn-clear") ? " btn-clear" : "");
                        if (rk && btn.dataset.rank == rk) {{
                            btn.classList.add("selected-" + Math.min(rk, 3));
                        }}
                    }});
                }});

                // Update summary text
                const summaryPill = document.getElementById("rank-summary-" + stemId);
                const sortedRanks = Object.entries(stemRatings).sort((a,b) => a[1] - b[1]);
                if (sortedRanks.length > 0) {{
                    summaryPill.innerText = sortedRanks.map(([m, rk]) => "#" + rk + ": " + m).join(" | ");
                    summaryPill.style.color = "#d29922";
                    summaryPill.style.fontStyle = "normal";
                    summaryPill.style.fontWeight = "bold";
                }} else {{
                    summaryPill.innerText = "Unrated";
                    summaryPill.style.color = "#8b949e";
                    summaryPill.style.fontStyle = "italic";
                    summaryPill.style.fontWeight = "normal";
                }}

                // Update badges on individual model cards
                MODELS.forEach(m => {{
                    const badge = document.getElementById("badge-" + stemId + "-" + m);
                    if (badge) {{
                        const rk = stemRatings[m];
                        if (rk === 1) {{
                            badge.innerHTML = "<span style=\'color: var(--gold);\'>&#9733; #1 Your Pick</span>";
                        }} else if (rk) {{
                            badge.innerHTML = "<span style=\'color: #8b949e;\'>#" + rk + " Pick</span>";
                        }} else {{
                            badge.innerHTML = "";
                        }}
                    }}
                }});
            }});
        }}

        function updatePreferenceDashboard() {{
            const ratings = loadStoredRatings();
            const ratedStems = Object.keys(ratings).length;
            document.getElementById("prefSummaryText").innerText = ratedStems + " / " + TOTAL_TRACKS + " Stems Rated";
            
            const firstPlaceWins = {{}};
            MODELS.forEach(m => firstPlaceWins[m] = 0);
            
            for (const stemId in ratings) {{
                for (const m in ratings[stemId]) {{
                    if (ratings[stemId][m] === 1) {{
                        firstPlaceWins[m] = (firstPlaceWins[m] || 0) + 1;
                    }}
                }}
            }}

            const sortedWins = Object.entries(firstPlaceWins).sort((a,b) => b[1] - a[1]);
            const winText = sortedWins.map(([m, w]) => "<strong>" + m + ":</strong> " + w + " wins (" + (ratedStems ? ((w/ratedStems)*100).toFixed(0) : 0) + "%)").join(" &bull; ");
            document.getElementById("prefWinsBar").innerHTML = winText;
        }}

        function exportPreferences() {{
            const ratings = loadStoredRatings();
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(ratings, null, 2));
            const dlAnchor = document.createElement('a');
            dlAnchor.setAttribute("href", dataStr);
            dlAnchor.setAttribute("download", "human_preferences.json");
            document.body.appendChild(dlAnchor);
            dlAnchor.click();
            dlAnchor.remove();
        }}

        function importPreferences() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = e => {{
                const file = e.target.files[0];
                const reader = new FileReader();
                reader.readAsText(file, 'UTF-8');
                reader.onload = re => {{
                    try {{
                        const parsed = JSON.parse(re.target.result);
                        saveStoredRatings(parsed);
                        alert("Preferences successfully imported!");
                    }} catch(err) {{
                        alert("Error parsing JSON file: " + err);
                    }}
                }};
            }};
            input.click();
        }}

        function resetAllPreferences() {{
            if (confirm("Are you sure you want to clear all your ratings?")) {{
                localStorage.removeItem(STORAGE_KEY);
                updatePreferenceDashboard();
                updateAllRatingPills();
            }}
        }}

        // Synchronized A/B Player Controller
        function getOrCreateAudio(widget) {{
            if (!widget._audio) {{
                widget._audio = new Audio();
                widget._audio.addEventListener('timeupdate', () => {{
                    const cur = widget._audio.currentTime;
                    const dur = widget._audio.duration || 1;
                    const pct = (cur / dur) * 100;
                    widget.querySelector('.sync-scrub').value = pct;
                    widget.querySelector('.time-display').innerText = formatTime(cur) + ' / ' + formatTime(dur);
                }});
                widget._audio.addEventListener('ended', () => {{
                    widget.querySelector('.play-btn').innerText = '&#9658; Play Synchronized';
                    widget.querySelector('.play-btn').classList.remove('playing');
                }});
            }}
            return widget._audio;
        }}

        function formatTime(s) {{
            const m = Math.floor(s / 60);
            const sec = Math.floor(s % 60);
            return m + ':' + (sec < 10 ? '0' : '') + sec;
        }}

        function toggleSyncPlay(btn) {{
            const widget = btn.closest('.sync-player-widget');
            const audioMap = JSON.parse(widget.dataset.audioMap);
            const activeTab = widget.querySelector('.switch-tab.active');
            const sourceKey = activeTab ? activeTab.innerText.replace(' ★', '').trim() : 'Real';
            const audioSrc = audioMap[sourceKey] || audioMap['Real'];

            const audio = getOrCreateAudio(widget);
            if (!audio.src || !audio.src.endsWith(audioSrc)) {{
                const curTime = audio.currentTime || 0;
                audio.src = audioSrc;
                audio.currentTime = curTime;
            }}

            if (audio.paused) {{
                document.querySelectorAll('.sync-player-widget').forEach(w => {{
                    if (w !== widget && w._audio && !w._audio.paused) {{
                        w._audio.pause();
                        w.querySelector('.play-btn').innerText = '&#9658; Play Synchronized';
                        w.querySelector('.play-btn').classList.remove('playing');
                    }}
                }});
                audio.play();
                btn.innerText = '&#10074;&#10074; Pause';
                btn.classList.add('playing');
            }} else {{
                audio.pause();
                btn.innerText = '&#9658; Play Synchronized';
                btn.classList.remove('playing');
            }}
        }}

        function switchSyncSource(tab, sourceKey) {{
            const widget = tab.closest('.sync-player-widget');
            const audioMap = JSON.parse(widget.dataset.audioMap);
            const newSrc = audioMap[sourceKey];
            if (!newSrc) return;

            widget.querySelectorAll('.switch-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const audio = getOrCreateAudio(widget);
            const curTime = audio.currentTime || 0;
            const wasPlaying = !audio.paused;

            audio.src = newSrc;
            audio.currentTime = curTime;
            if (wasPlaying) {{
                audio.play();
            }}
        }}

        function seekSync(range) {{
            const widget = range.closest('.sync-player-widget');
            const audio = getOrCreateAudio(widget);
            if (audio.duration) {{
                audio.currentTime = (range.value / 100) * audio.duration;
            }}
        }}

        function filterStems(query) {{
            query = query.toLowerCase();
            document.querySelectorAll('.track-row').forEach(row => {{
                const stem = row.dataset.stem.toLowerCase();
                const style = row.dataset.style.toLowerCase();
                const match = stem.includes(query) || style.includes(query);
                row.style.display = match ? 'block' : 'none';
            }});
        }}

        function setFilter(btn, filter) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.track-row').forEach(row => {{
                if (filter === 'all') {{
                    row.style.display = 'block';
                }} else {{
                    row.style.display = (row.dataset.winner === filter) ? 'block' : 'none';
                }}
            }});
        }}

        // Initialize ratings on load
        window.addEventListener('DOMContentLoaded', () => {{
            updatePreferenceDashboard();
            updateAllRatingPills();
        }});
    </script>
</body>
</html>
"""

out_html_path = os.path.join(listen_dir, "index.html")
with open(out_html_path, "w") as f:
    f.write(html_content)

print(f"Successfully generated interactive model evaluation page with rating widget at: {out_html_path}")
