"""
ngl_sync_viewer.py
-------------------
Drop-in NGL-based replacement for the py3Dmol single/multi viewers in the
FlexiFold app. Gives true bi-directional 1D <-> 3D sync (hover AND click,
both directions) for 1-4 simultaneous conditions, using the same pattern
as the "Bi-Directional Parallel Mapping Workspace" prototype, generalized
from a single overview+inset pair to an N-condition grid.

Public API
----------
fetch_unimod_data(unimod_ids: tuple[str]) -> dict
    Live-fetches name / mass / composition / an auto-suggested color for
    each "unimod:NN" id straight from unimod.org's own record page.

seed_ptm_config_defaults(configs: dict, unimod_ids, force_refresh_auto=False) -> dict
    Mutates `configs` (e.g. st.session_state.ptm_configs) in place so every
    detected UniMod id has a config entry. Only SETS a default the first
    time an id is seen -- an existing user-edited color/label is never
    overwritten. Always refreshes the remembered "auto_color"/"auto_label"
    so the reset (<-) button in the UI has something current to reset to.

render_ngl_synced_viewer(...)
    Renders the full synced 3D-grid + sequence-grid block via
    streamlit.components.v1.html. See docstring below for parameters.

Nothing in this module imports streamlit at module scope except inside the
functions that need it, so it can be unit-imported/tested without a live
Streamlit session for the pure-Python pieces (fetch_unimod_data etc.).
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors


# ============================================================================
# LIVE UNIMOD RECORD-PAGE FETCHING  (ported from the NGL prototype)
# ============================================================================

def _parse_unimod_view_page(html: str) -> dict:
    """Flatten the record-page's <td> label/value pairs into a dict."""
    soup = BeautifulSoup(html, "html.parser")
    cells = [td.get_text(strip=True) for td in soup.find_all("td")]

    def value_after(label):
        for i, c in enumerate(cells):
            if c == label and i + 1 < len(cells):
                return cells[i + 1]
        return ""

    psi_name = value_after("PSI-MS Name")
    interim_name = value_after("Interim Name")
    description = value_after("Description")
    composition = value_after("Composition")
    mono_mass_str = value_after("Monoisotopic")

    try:
        mono_mass = float(mono_mass_str)
    except ValueError:
        mono_mass = 0.0

    name = psi_name or interim_name or description or "Unknown"

    return {
        "name": name,
        "description": description,
        "mono_mass": mono_mass,
        "composition": composition,
    }


def _fetch_unimod_view_page_uncached(accession: str) -> Optional[dict]:
    url = f"https://www.unimod.org/modifications_view.php?editid1={accession}"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or not resp.text:
            return None
        parsed = _parse_unimod_view_page(resp.text)
        if not parsed["name"] or parsed["name"] == "Unknown":
            return None
        parsed["accession"] = accession
        parsed["source_url"] = url
        return parsed
    except Exception:
        return None


def _auto_color(composition: str, mass: float) -> str:
    """Derive a default hex color suggestion from chemical composition."""
    comp = (composition or "").upper()
    if "P" in comp:
        return "#ef4444"   # phosphorus -> red (phosphorylation)
    if "S" in comp and mass > 50:
        return "#a855f7"   # sulfur + heavy -> purple (oxidation, alkylation)
    if "S" in comp:
        return "#6b7280"   # sulfur light -> grey (carbamidomethyl)
    if mass < 0:
        return "#0ea5e9"   # mass loss -> blue (deamidation loss)
    if mass < 20:
        return "#10b981"   # tiny addition -> green (methylation)
    if mass < 60:
        return "#eab308"   # medium -> yellow (acetylation)
    if "N" in comp and "O" in comp:
        return "#f97316"   # nitrogen+oxygen -> orange (nitration)
    return "#94a3b8"        # default slate


def fetch_unimod_data(unimod_ids: Tuple[str, ...]) -> Dict[str, dict]:
    """
    Build the enrichment dict for every UniMod id passed in.
    `unimod_ids` items can be like "unimod:21" or "UniMod:21" or "21".
    Returns: { "<original id string>": {name, mono_mass, composition,
                                          color, source_url}, ... }
    Caches nothing itself -- wrap this call in st.cache_data at the
    call site (it needs `unimod_ids` to be a hashable tuple, which it is).
    """
    result = {}
    for uid in unimod_ids:
        accession_match = re.search(r"\d+", str(uid))
        if not accession_match:
            continue
        accession = accession_match.group()
        parsed = _fetch_unimod_view_page_uncached(accession)
        if not parsed:
            continue
        comp = parsed["composition"]
        mass = parsed["mono_mass"]
        result[uid] = {
            "name": parsed["name"],
            "mono_mass": mass,
            "composition": comp,
            "color": _auto_color(comp, mass),
            "source_url": parsed["source_url"],
        }
    return result


def seed_ptm_config_defaults(configs: dict, unimod_ids, unimod_enriched: dict = None) -> dict:
    """
    Ensure every id in `unimod_ids` has an entry in `configs` shaped like:
        { 'selected': bool, 'label': str, 'color': hex, 'auto_color': hex,
          'auto_label': str }
    Never overwrites an existing 'label'/'color'/'selected' (those are the
    user's manual overrides) -- only fills them in the first time an id is
    seen, and always refreshes 'auto_color'/'auto_label' so a "reset to
    auto" control always has a current target.
    `unimod_enriched` is the dict returned by fetch_unimod_data (already
    fetched by the caller, usually via st.cache_data so it's cheap).
    """
    unimod_enriched = unimod_enriched or {}
    for uid in unimod_ids:
        enriched = unimod_enriched.get(uid, {})
        auto_label = enriched.get("name") or str(uid).upper()
        auto_color = enriched.get("color") or "#436da9"

        if uid not in configs:
            configs[uid] = {
                "selected": True,
                "label": auto_label,
                "color": auto_color,
                "auto_color": auto_color,
                "auto_label": auto_label,
            }
        else:
            configs[uid]["auto_color"] = auto_color
            configs[uid]["auto_label"] = auto_label
            configs[uid].setdefault("selected", True)
            configs[uid].setdefault("label", auto_label)
            configs[uid].setdefault("color", auto_color)
    return configs


# ============================================================================
# COLOR MAPPING (self-contained copy so this module has no app.py import)
# ============================================================================

def _generate_hex_colors(residue_vals, cmap_name, not_mapped_color, vmin, vmax):
    cmap = colormaps[cmap_name]
    norm = Normalize(vmin=vmin, vmax=vmax)
    hex_colors = []
    for val in residue_vals:
        if val is None:
            hex_colors.append(not_mapped_color)
        else:
            denom = (vmax - vmin) if vmax > vmin else 1.0
            t = (val - vmin) / denom
            t = min(1.0, max(0.0, t))
            rgb = cmap(norm(val) if vmax > vmin else 0.5)[:3]
            hex_colors.append(mcolors.rgb2hex(rgb))
    return hex_colors


def _generate_color_segments(residue_vals, cmap_name, not_mapped_color, vmin, vmax, n_bins=48):
    """
    Quantizes the z-score colormap into `n_bins` discrete steps and
    run-length-encodes consecutive same-bucket residues into
    (hex, start_1based, end_1based_inclusive) segments.

    This feeds NGL's ColormakerRegistry.addSelectionScheme (a stable,
    long-standing "color these residue ranges these colors" API), which is
    far more reliably supported across NGL builds than the function-based
    addScheme() custom-Colormaker API -- the latter is what silently fell
    back to NGL's default chain coloring and produced the flat
    "detected/undetected in one color" look.
    """
    cmap = colormaps[cmap_name]
    denom = (vmax - vmin) if vmax > vmin else 1.0

    def bucket_hex(val):
        if val is None:
            return not_mapped_color
        t = (val - vmin) / denom
        t = min(1.0, max(0.0, t))
        b = round(t * (n_bins - 1)) if n_bins > 1 else 0
        t_q = (b / (n_bins - 1)) if n_bins > 1 else 0.5
        rgb = cmap(t_q)[:3]
        return mcolors.rgb2hex(rgb)

    colors = [bucket_hex(v) for v in residue_vals]
    segments = []
    n = len(colors)
    i = 0
    while i < n:
        c = colors[i]
        j = i
        while j + 1 < n and colors[j + 1] == c:
            j += 1
        segments.append([c, i + 1, j + 1])  # 1-based inclusive
        i = j + 1
    return segments


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render_ngl_synced_viewer(
    pdb_str: str,
    protein_seq: str,
    conditions: List[str],
    residue_data: Dict[str, list],
    ptm_data: Dict[str, dict],
    vmin: float,
    vmax: float,
    cmap_name: str = "autumn",
    not_mapped_color: str = "#d3d3d3",
    backbone_style: str = "ribbon",
    manual_zoom: Optional[dict] = None,
    height_per_row: int = 380,
    key_suffix: str = "",
    value_label: str = "Z-Score",
):
    """
    Renders `len(conditions)` synced NGL 3D panels (1-4, wraps to a 2nd row
    above 2) plus ONE shared sequence grid underneath, with full
    bi-directional hover/click linking, click-to-zoom on the contiguous
    mapped region under the cursor, a manual residue-range zoom, a
    ribbon/CPK backbone toggle, and chemically-accurate PTM hyperball
    markers per condition (color/label taken from that condition's
    ptm_data, which the caller should already have merged with
    seed_ptm_config_defaults()).

    residue_data: { condition_name: [float|None, ...] }  (len == len(protein_seq))
    ptm_data:     { condition_name: { unimod_id: {'positions':[int,...],
                                                    'selected':bool,
                                                    'color':hex,
                                                    'label':str} } }
    manual_zoom:  {'start': int, 'end': int} (1-based, inclusive) or None
    """
    import streamlit as st
    import streamlit.components.v1 as components

    seq_len = len(protein_seq)
    conditions = list(conditions)[:4]  # hard cap at 4 panels, matches original app's 2-4 rule where relevant
    n = len(conditions)

    # ---- per-condition color SEGMENTS (quantized + run-length-encoded) ----
    # (see _generate_color_segments docstring for why segments, not a
    # per-atom custom Colormaker, are used to drive the 3D coloring)
    condition_segments = {}
    condition_vals = {}
    for cond in conditions:
        vals = residue_data.get(cond, [None] * seq_len)
        condition_segments[cond] = _generate_color_segments(vals, cmap_name, not_mapped_color, vmin, vmax)
        condition_vals[cond] = [None if v is None else round(float(v), 3) for v in vals]

    # ---- union "mapped" map used to build clickable peptide segments -----
    union_mapped = [False] * seq_len
    for cond in conditions:
        for i, v in enumerate(residue_data.get(cond, [])):
            if v is not None:
                union_mapped[i] = True

    # ---- flatten PTM data per condition, per residue ----------------------
    # ptm_by_cond[cond][res_idx_0based] = {id,name,color,mass,composition,source_url,label}
    ptm_by_cond = {cond: {} for cond in conditions}
    for cond in conditions:
        cond_ptms = ptm_data.get(cond, {}) or {}
        for uid, info in cond_ptms.items():
            if not info.get("selected", True):
                continue
            positions = info.get("positions", [])
            if isinstance(positions, dict):
                positions = positions.get("positions", [])
            color = info.get("color", "#436da9")
            label = info.get("label", str(uid))
            for pos0 in positions:
                pos0 = int(pos0)
                if 0 <= pos0 < seq_len:
                    ptm_by_cond[cond][pos0] = {
                        "id": str(uid).upper(),
                        "name": label,
                        "color": color,
                    }

    js_data = {
        "fullSeq": protein_seq,
        "conditions": conditions,
        "conditionSegments": condition_segments,
        "conditionVals": condition_vals,
        "unionMapped": union_mapped,
        "ptmByCond": {c: {str(k): v for k, v in ptm_by_cond[c].items()} for c in conditions},
        "backboneStyle": backbone_style,
        "manualSelection": manual_zoom,
        "valueLabel": value_label,
        "pdbStr": pdb_str,
    }

    n_cols = 2 if n > 1 else 1
    panel_min_width = 320

    html_head = """
<style>
  * { box-sizing: border-box; }
  .ngls-root { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  .ngls-grid {
    display: grid;
    grid-template-columns: repeat(""" + str(n_cols) + """, minmax(""" + str(panel_min_width) + """px, 1fr));
    gap: 14px;
    margin-bottom: 14px;
  }
  .ngls-panel-wrap { display:flex; flex-direction:column; gap:6px; min-width:0; }
  .ngls-panel-label {
    font-size: 12px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
    color:#94a3b8; padding-left:4px;
  }
  .ngls-viewport {
    width:100%; height: """ + str(height_per_row) + """px; background:#0b0f19;
    border:1px solid #1e293b; border-radius:12px; position:relative; overflow:hidden;
  }
  #ngls-hud""" + key_suffix + """ {
    position:absolute; bottom:10px; left:10px; right:10px;
    background:rgba(15,23,42,.95); border:1px solid #334155; border-radius:8px;
    padding:10px 12px; color:#f8fafc; font-size:12px; display:none;
    backdrop-filter: blur(6px); z-index:1000; pointer-events:none;
  }
  #ngls-badge""" + key_suffix + """ {
    display:none; position:fixed; z-index:999; background:rgba(244,63,94,.18);
    border:1px solid #f43f5e; border-radius:6px; padding:5px 10px; color:#fda4af;
    font-size:12px; font-weight:600; cursor:pointer; user-select:none;
  }
  .ngls-seq-row {
    border:1px solid #e2e8f0; border-radius:12px; background:#fff; overflow:hidden;
  }
  .ngls-seq-header { display:flex; justify-content:space-between; align-items:center; padding:10px 16px; border-bottom:2px solid #f1f5f9; }
  .ngls-seq-header h4 { margin:0; color:#1e293b; font-size:14px; }
  #ngls-seq-container""" + key_suffix + """ {
    height:170px; overflow-y:auto; padding:14px 18px; letter-spacing:7px; line-height:34px;
    font-family: 'SFMono-Regular', Consolas, monospace; font-size:15px; word-break:break-all; user-select:none;
  }
</style>
<div class="ngls-root">
  <div id="ngls-badge""" + key_suffix + """">Zoomed \u2014 click to reset \u2715</div>
  <div class="ngls-grid" id="ngls-grid""" + key_suffix + """"></div>
  <div class="ngls-seq-row">
    <div class="ngls-seq-header"><h4>Synced Sequence (hover/click links to every 3D panel above)</h4></div>
    <div id="ngls-seq-container""" + key_suffix + """"></div>
  </div>
</div>
<script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
<script>
(function() {
const SFX = \"""" + key_suffix + """\";
const data = """ + json.dumps(js_data) + """;

const fullSeq          = data.fullSeq;
const conditions        = data.conditions;
const conditionSegments = data.conditionSegments;
const conditionVals     = data.conditionVals;
const unionMapped     = data.unionMapped;
const ptmByCond       = data.ptmByCond;
const BACKBONE_STYLE  = data.backboneStyle;
const MANUAL_SELECTION= data.manualSelection;
const VALUE_LABEL     = data.valueLabel;
const PDB_STR         = data.pdbStr;

function nglSele(a,b){ return a + "-" + b + ":A"; }
function resolveAtom(proxy){
  if (!proxy) return null;
  if (proxy.atom) return proxy.atom;
  if (proxy.closestBondAtom) return proxy.closestBondAtom;
  if (proxy.bond) return proxy.bond.atom1;
  return null;
}

// contiguous "mapped" segments, used for click-to-zoom
const segments = [];
(function(){
  let s=null;
  for (let i=0;i<unionMapped.length;i++){
    if (unionMapped[i]) { if (s===null) s=i+1; }
    else { if (s!==null){ segments.push({start:s,end:i}); s=null; } }
  }
  if (s!==null) segments.push({start:s, end: unionMapped.length});
})();
function findSegment(resNum){
  for (const seg of segments) if (resNum>=seg.start && resNum<=seg.end) return seg;
  return null;
}

// ---- build the grid + stages --------------------------------------------
const gridEl = document.getElementById("ngls-grid"+SFX);
const stages = {};
const colorSchemeNames = {};

conditions.forEach(function(cond, idx){
  const wrap = document.createElement("div");
  wrap.className = "ngls-panel-wrap";
  const label = document.createElement("div");
  label.className = "ngls-panel-label";
  label.textContent = cond;
  const viewport = document.createElement("div");
  viewport.className = "ngls-viewport";
  viewport.id = "ngls-vp-" + idx + SFX;
  const hud = document.createElement("div");
  hud.id = "ngls-hud" + SFX;
  wrap.appendChild(label);
  viewport.appendChild(hud);
  wrap.appendChild(viewport);
  gridEl.appendChild(wrap);

  const stage = new NGL.Stage(viewport.id, {backgroundColor:"#0b0f19"});
  stages[cond] = stage;

  const schemeName = "scheme_" + idx + SFX;
  const segs = conditionSegments[cond] || [];
  const pairs = segs.map(function(seg){
    // seg = [hex, startResno, endResno] (1-based, inclusive)
    return [seg[0], seg[1] + "-" + seg[2] + ":A"];
  });
  let registeredId = schemeName;
  try {
    registeredId = NGL.ColormakerRegistry.addSelectionScheme(pairs, schemeName) || schemeName;
  } catch (e) {
    console.error("NGL addSelectionScheme failed for", cond, e);
  }
  colorSchemeNames[cond] = registeredId;
});

const hudEl = document.getElementById("ngls-hud"+SFX);
const badgeEl = document.getElementById("ngls-badge"+SFX);
let zoomedSeg = null;
let currentHighlightReps = {}; // cond -> repr handle

function addPtmReprs(comp, cond, rn, emphasize){
  const ptm = ptmByCond[cond] ? ptmByCond[cond][String(rn-1)] : null;
  if (!ptm) return;
  const sele = nglSele(rn, rn);
  const scale = emphasize ? 0.55 : 0.3;
  comp.addRepresentation("hyperball", {
    sele: sele + " AND sidechain", color: ptm.color, scale: scale, opacity: 1.0
  });
  comp.addRepresentation("spacefill", {
    sele: sele + " AND .CA", color: ptm.color, scale: 0.4, opacity: 1.0
  });
}

function addBackboneRepr(comp, sele, schemeName, opacity, cpkScale){
  try {
    if (BACKBONE_STYLE === "cpk") {
      comp.addRepresentation("hyperball", {sele: sele, color: schemeName, opacity: opacity, scale: cpkScale || 0.2});
    } else {
      comp.addRepresentation("cartoon", {sele: sele, color: schemeName, opacity: opacity});
    }
  } catch (e) {
    console.error("addBackboneRepr failed, falling back to flat color", e);
    const fallbackRepr = (BACKBONE_STYLE === "cpk") ? "hyperball" : "cartoon";
    comp.addRepresentation(fallbackRepr, {sele: sele, color: "#a7a5a5", opacity: opacity});
  }
}

function applyBaseReps(cond){
  const stage = stages[cond];
  const comp = stage.compList[0];
  if (!comp) return;
  comp.removeAllRepresentations();
  addBackboneRepr(comp, "polymer", colorSchemeNames[cond], 0.98, 0.22);
  for (let i=0;i<fullSeq.length;i++){
    if (ptmByCond[cond] && ptmByCond[cond][String(i)]) addPtmReprs(comp, cond, i+1, true);
  }
}

function zoomAllTo(seg){
  conditions.forEach(function(cond){
    const stage = stages[cond];
    const comp = stage.compList[0];
    if (comp) comp.autoView(nglSele(seg.start, seg.end), 900);
  });
  zoomedSeg = seg;
  badgeEl.style.display = "block";
  updateSeqSelection(seg);
}
function resetZoomAll(){
  conditions.forEach(function(cond){
    const stage = stages[cond];
    const comp = stage.compList[0];
    if (comp) comp.autoView(900);
  });
  zoomedSeg = null;
  badgeEl.style.display = "none";
  clearSeqSelection();
}
badgeEl.addEventListener("click", resetZoomAll);

// ---- crosshair + hud on hover/click, mirrored across ALL panels ---------
function showCrosshair(resNum){
  conditions.forEach(function(cond){
    const stage = stages[cond];
    const comp = stage.compList[0];
    if (!comp) return;
    if (currentHighlightReps[cond]) {
      try { comp.removeRepresentation(currentHighlightReps[cond]); } catch(e) {}
    }
    currentHighlightReps[cond] = comp.addRepresentation("spacefill", {
      sele: nglSele(resNum, resNum) + " AND .CA", color: "#f43f5e", scale: 1.5
    });
  });
}

function populateHud(resNum){
  const idx = resNum - 1;
  const letter = fullSeq[idx] || "?";
  let rows = "";
  conditions.forEach(function(cond){
    const v = conditionVals[cond][idx];
    const vTxt = (v === null || v === undefined) ? "not mapped" : v.toFixed(3);
    const ptm = ptmByCond[cond] ? ptmByCond[cond][String(idx)] : null;
    const ptmTxt = ptm ? (" &nbsp;<span style='color:" + ptm.color + ";font-weight:bold;'>\u25CF " + ptm.name + "</span>") : "";
    rows += "<div style='display:flex;justify-content:space-between;gap:10px;padding:2px 0;'>" +
              "<span style='color:#94a3b8;'>" + cond + "</span>" +
              "<span>" + VALUE_LABEL + ": " + vTxt + ptmTxt + "</span>" +
            "</div>";
  });
  hudEl.innerHTML = "<div style='font-weight:bold;color:#3b82f6;margin-bottom:6px;'>Residue " + letter + resNum + "</div>" + rows;
  hudEl.style.display = "block";
}

function updateSeqHighlight(resNum, scroll){
  document.querySelectorAll("#ngls-seq-container"+SFX+" span[data-resnum]").forEach(function(sp){
    const rn = parseInt(sp.getAttribute("data-resnum"));
    if (rn === resNum) {
      sp.style.outline = "3px solid #f43f5e";
      sp.style.transform = "scale(1.3)";
      sp.style.zIndex = "100";
    } else if (!zoomedSeg) {
      sp.style.outline = "none";
      sp.style.transform = "none";
    }
  });
  if (scroll) {
    const el = document.getElementById("ngls-res-" + resNum + SFX);
    if (el) el.scrollIntoView({behavior:"smooth", block:"center"});
  }
}

function updateSeqSelection(seg){
  document.querySelectorAll("#ngls-seq-container"+SFX+" span[data-resnum]").forEach(function(sp){
    const rn = parseInt(sp.getAttribute("data-resnum"));
    if (rn >= seg.start && rn <= seg.end) {
      sp.style.opacity = "1"; sp.style.outline = "2px solid #f43f5e";
    } else {
      sp.style.opacity = "0.2"; sp.style.outline = "none";
    }
  });
  const anchor = document.getElementById("ngls-res-" + seg.start + SFX);
  if (anchor) anchor.scrollIntoView({behavior:"smooth", block:"center"});
}
function clearSeqSelection(){
  document.querySelectorAll("#ngls-seq-container"+SFX+" span[data-resnum]").forEach(function(sp){
    sp.style.opacity = "1"; sp.style.outline = "none"; sp.style.transform="none";
  });
}

function focusResidue(resNum, fromCanvas, andZoom){
  if (resNum < 1 || resNum > fullSeq.length) return;
  showCrosshair(resNum);
  populateHud(resNum);
  updateSeqHighlight(resNum, fromCanvas);
  if (andZoom) {
    const seg = findSegment(resNum);
    if (seg) zoomAllTo(seg);
  }
}

// ---- load PDB into every stage, wire hover/click -------------------------
const pdbBlob = new Blob([PDB_STR], {type:"text/plain"});
let loadedCount = 0;
conditions.forEach(function(cond){
  const stage = stages[cond];
  stage.loadFile(pdbBlob, {ext:"pdb"}).then(function(comp){
    applyBaseReps(cond);
    if (MANUAL_SELECTION && MANUAL_SELECTION.start && MANUAL_SELECTION.end) {
      comp.autoView(nglSele(MANUAL_SELECTION.start, MANUAL_SELECTION.end), 0);
    } else {
      comp.autoView();
    }
    stage.signals.hovered.add(function(proxy){
      const atom = resolveAtom(proxy);
      if (!atom || !atom.resno) return;
      focusResidue(atom.resno, false, false);
    });
    stage.signals.clicked.add(function(proxy){
      const atom = resolveAtom(proxy);
      if (!atom || !atom.resno) { resetZoomAll(); return; }
      focusResidue(atom.resno, true, true);
    });
    loadedCount++;
    if (loadedCount === conditions.length && MANUAL_SELECTION && MANUAL_SELECTION.start && MANUAL_SELECTION.end) {
      zoomAllTo({start: MANUAL_SELECTION.start, end: MANUAL_SELECTION.end});
    }
  });
});

// ---- build the shared sequence grid ---------------------------------------
const seqContainer = document.getElementById("ngls-seq-container"+SFX);
for (let i=0;i<fullSeq.length;i++){
  const resNum = i+1;
  const letter = fullSeq[i];
  const wrapper = document.createElement("div");
  wrapper.style.display = "inline-block";
  wrapper.style.position = "relative";

  const span = document.createElement("span");
  span.innerText = letter;
  span.id = "ngls-res-" + resNum + SFX;
  span.setAttribute("data-resnum", resNum);
  span.style.padding = "2px 4px";
  span.style.cursor = "pointer";
  span.style.borderRadius = "4px";
  span.style.transition = "all .12s ease";

  if (unionMapped[i]) {
    span.style.backgroundColor = "#dbeafe";
    span.style.color = "#1e40af";
    span.style.fontWeight = "bold";
  } else {
    span.style.color = "#94a3b8";
  }

  // dot if ANY condition has a PTM at this residue
  let dotColor = null;
  for (const cond of conditions) {
    if (ptmByCond[cond] && ptmByCond[cond][String(i)]) { dotColor = ptmByCond[cond][String(i)].color; break; }
  }
  if (dotColor) {
    const dot = document.createElement("div");
    dot.style.position="absolute"; dot.style.top="-4px"; dot.style.left="35%";
    dot.style.width="8px"; dot.style.height="8px"; dot.style.borderRadius="50%";
    dot.style.backgroundColor = dotColor; dot.style.border="1.5px solid #fff";
    wrapper.appendChild(dot);
  }

  span.addEventListener("mouseenter", function(){ focusResidue(resNum, false, false); });
  span.addEventListener("click", function(){ focusResidue(resNum, false, true); });

  wrapper.appendChild(span);
  seqContainer.appendChild(wrapper);
}
})();
</script>
"""

    total_height = (height_per_row * ((n + n_cols - 1) // n_cols)) + 260
    components.html(html_head, height=total_height, scrolling=True)
