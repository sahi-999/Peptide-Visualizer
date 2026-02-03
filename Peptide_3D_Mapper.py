import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
import py3Dmol
import io
import requests
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
import matplotlib.colors as mcolors
import plotly.express as px
from matplotlib.cm import ScalarMappable
import zipfile
from streamlit.components.v1 import html
import json  # Keep only this import
from Bio.PDB import PDBParser
import re
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import List,Optional,Tuple,Dict
from streamlit import cache_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


st.session_state.setdefault("main_tab", "Qualitative Analysis")      # default main tab
st.session_state.setdefault("active_quant_tab", "single")           # quant sub-tab
st.session_state.setdefault("processed_single", {"processed": False})
st.session_state.setdefault("processed_multi", {"processed": False})

# Set wide layout #
# Page config
st.set_page_config(page_title="Peptide3D Mapper", page_icon="⚛️", layout="wide")
#st.title("🧬 Peptide3D Mapper")
# === Session State ===
for key in ["is_frequency", "ptm_enabled", "apply_tryptic", "all_unimods", "selected_unimods","ptm_configs", "pdb_source", "uploaded_pdb"]:
    if key not in st.session_state:
        if key in ["is_frequency", "ptm_enabled", "apply_tryptic"]:
            st.session_state[key] = False
        elif key in ["all_unimods", "selected_unimods"]:
            st.session_state[key] = []
        elif key == "ptm_configs":
            st.session_state[key] = {}   # ← Critical!
        elif key == "pdb_source":
            st.session_state[key] = "AlphaFold"
        else:
            st.session_state[key] = None

@cache_data(ttl=60*60*24)  # cache for 24 h – DisProt changes rarely
def get_disprot_info(uniprot_id):
    url = f"https://disprot.org/api/search?query={uniprot_id}&namespace=uniprot&format=json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"found": False}
        data = response.json()
        results = data.get("results", [])
        if not results:
            return {"found": False}
        # Take the first entry (or you can add logic to select specific isoform if needed)
        entry = results[0]

        disorder_str = entry.get("disorder_content", "0%")
        # Extract number from "38.58%" or "100.00%"
        try:
            disorder_percent = float(disorder_str.replace("%", "").strip())
        except:
            disorder_percent = None
        return {
            "found": True,
            "disprot_id": entry.get("disprot_id"),
            "name": entry.get("name"),
            "organism": entry.get("organism"),
            "sequence_length": entry.get("sequence_length"),
            "disorder_percent": disorder_percent,
        }
    except:
        return {"found": False}
def has_missed_cleavage(peptide: str) -> bool:
    """
    Returns True if the peptide has one or more missed cleavages (i.e. NOT fully tryptic).
    Respects the biological rule: Trypsin does NOT cut at K-P or R-P bonds.
    
    Fully tryptic peptides must:
    - NOT start with K or R
    - End with K or R
    - Have no internal K/R unless followed by P (proline exception)
    """
    seq = peptide.strip().upper()
    if len(seq) < 2:
        return True

    # 1. Starts with K/R → missed cleavage at N-terminus
    if seq[0] in 'KR':
        return True

    # 2. Does not end with K/R → missed cleavage at C-terminus
    if seq[-1] not in 'KR':
        return True

    # 3. Internal missed cleavage: K or R not followed by P (and not at the end)
    for i in range(len(seq) - 2):
        if seq[i] in 'KR' and seq[i + 1] != 'P':
            return True  # There's a cleavable site inside → missed cleavage

    return False  # Fully tryptic!


def find_valid_peptides(sequence: str, peptides: List[str], protein_id: str = ""
                        ) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Returns (valid_peptides, invalid_peptides) based on strict proteotypic rules.
    Accepts peptides that are:
      - Preceded by K or R
      - At protein start (position 0)
      - At position 1 if protein starts with 'M' (common in UniProt/FASTA)
    """
    seq = sequence.upper()
    valid : List[str]=[]
    unmatched: List[tuple[str,str]]= []
    
    # Allow N-terminal peptides at position 0 or 1 (if protein starts with M)
    n_term_positions = [0]
    if seq.startswith('M'):
        n_term_positions.append(1)

    seen = set()  # Avoid duplicates
    for peptide in peptides:
        pep = peptide.strip().upper()
        if not pep or len(pep) < 6 or pep in seen:
            continue
        seen.add(pep)

        # Find all occurrences
        start_pos = 0
        found_valid = False
        reason = ""

        while True:
            pos = seq.find(pep, start_pos)
            if pos == -1:
                break

            # Check if this occurrence is validly cleaved
            if pos in n_term_positions:
                found_valid = True
                reason = "N-terminal peptide (incl. after M)" if pos == 1 else "Protein N-terminus"
                break
            elif pos > 0 and seq[pos - 1] in 'KR':
                found_valid = True
                reason = f"Preceded by {seq[pos-1]}"
                break

            start_pos = pos + 1  # Continue searching overlapping

        if found_valid:
            valid.append(pep)
            print(f"Peptide '{pep}' for {protein_id} validated: {reason}")
        else:
            # Check if peptide exists at all
            if seq.find(pep) != -1:
                unmatched.append((pep, "Found but no valid tryptic cleavage site"))
            else:
                unmatched.append((pep, "Not found in protein sequence"))

    return valid, unmatched
# --- Helper Functions ---
def compute_z_scores(intensities, is_frequency=False):
    if is_frequency:                     # frequency already 0-1 → simple standardisation
        m, s = np.mean(intensities), np.std(intensities)
        return np.zeros_like(intensities) if s == 0 else (intensities - m) / s
    else:
        log_int = np.log10(intensities + 1)
        m, s = np.mean(log_int), np.std(log_int)
        return np.zeros_like(log_int) if s == 0 else (log_int - m) / s
def clean_and_find_mods(peptide):
    """
    Cleans a peptide sequence and finds UniMod modification positions and types.
    Returns:
        cleaned_seq: sequence without UniMod tags
        mod_list: list of tuples (0-based position, unimod_type)
    """
    mod_list = []
    cleaned_seq = ""
    index = -1  # Start at -1 to make position 0-based after first increment
    pattern = re.compile(r"([A-Z])(\(UniMod:(\d+)\))?", re.IGNORECASE)
    for match in pattern.finditer(peptide):
        aa, mod, num = match.groups()
        index += 1
        cleaned_seq += aa
        if num:
            mod_list.append((index, num))  # 0-based position
    #st.write(f"Parsed PTM: {peptide} -> Cleaned: {cleaned_seq}, Mods: {mod_list}")  # Debug
    return cleaned_seq, mod_list

def map_peptides_to_residues(df, protein_seq, intensity_col, overlap_strategy='merge', ptm_col=None, apply_tryptic=False,proteotypic_only=False):
    seq_len = len(protein_seq)
    residue_vals = [None] * seq_len
    ptm_positions = {}  # ← This will collect ALL PTMs, even from zero-intensity peptides
    n_term_offset = 1 if protein_seq.startswith('M') else 0
    # ------------------------------------------------------------------
    # STEP 1: PRE-COLLECT ALL PTMs — INDEPENDENT OF INTENSITY (THE FIX)
    # ------------------------------------------------------------------
    if ptm_col and st.session_state.ptm_enabled:
        for _, row in df.iterrows():
            if ptm_col not in row or pd.isna(row[ptm_col]):
                continue
            mod_seq = str(row[ptm_col])
            if '(UniMod:' not in mod_seq:
                continue

            pep = row['Stripped.Sequence']
            if pd.isna(pep):
                continue

            cleaned_pep, mods = clean_and_find_mods(mod_seq)
            if cleaned_pep != pep:
                continue

            # Find all positions of this peptide in protein
            matches = list(re.finditer(re.escape(pep), protein_seq))
            for match in matches:
                if apply_tryptic and match.start() > 0 and protein_seq[match.start() - 1] not in 'KR':
                    continue
                start = match.start()
                for rel_pos, unismod in mods:
                    abs_pos = start + rel_pos
                    if 0 <= abs_pos < seq_len:
                        key = f"UniMod:{unismod}"
                        ptm_positions.setdefault(key, set()).add(abs_pos)

    # ------------------------------------------------------------------
    # STEP 2: NORMAL intensity/frequency mapping (unchanged — keeps real coverage!)
    # ------------------------------------------------------------------
    peptides = df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
    z_scores = compute_z_scores(peptides[intensity_col], is_frequency=st.session_state.is_frequency)

    for idx, row in df.iterrows():
        pep = row['Stripped.Sequence']
        intensity = row[intensity_col]
        
        has_modification = (
            ptm_col and 
            ptm_col in row and 
            pd.notna(row[ptm_col]) and 
            '(UniMod:' in str(row[ptm_col])
        )

        if pd.isna(intensity):
            #if not (st.session_state.is_frequency and has_modification):
                continue
            # Allow processing with intensity = 0 → frequency contribution = 0
        if  intensity == 0: 
            if not st.session_state.is_frequency:
                continue
            if not has_modification:
                continue
    # In intensity mode: zero is zero → skip unless modified (but still parse PTM)
        matches = list(re.finditer(re.escape(pep), protein_seq))
        if not matches:
            continue

        valid_found = False
        for match in matches:
            start = match.start()
            if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                continue
            valid_found = True
            end = start + len(pep)

            peptide_match = peptides[peptides['Stripped.Sequence'] == pep]
            if peptide_match.empty:
                continue
            z_val = z_scores[peptide_match.index[0]]

            for i in range(start, end):
                if residue_vals[i] is None:
                    residue_vals[i] = [z_val]
                else:
                    residue_vals[i].append(z_val)

            # PTM parsing (only from non-zero intensity peptides — but we already pre-collected all!)
            if ptm_col and ptm_col in row and pd.notna(row[ptm_col]) and '(UniMod:' in str(row[ptm_col]):
                cleaned_pep, mods = clean_and_find_mods(row[ptm_col])
                if cleaned_pep == pep:
                    for rel_pos, unismod in mods:
                        abs_pos = start + rel_pos
                        if 0 <= abs_pos < seq_len:
                            key = f"UniMod:{unismod}"
                            ptm_positions.setdefault(key, set()).add(abs_pos)

        if not valid_found:
            continue

    # Resolve overlaps (unchanged)
    for i in range(seq_len):
        if residue_vals[i]:
            if overlap_strategy == 'merge':
                residue_vals[i] = np.mean(residue_vals[i])
            elif overlap_strategy == 'highest':
                residue_vals[i] = np.max(residue_vals[i])
            else:
                residue_vals[i] = residue_vals[i][-1]
        else:
            residue_vals[i] = None

    # Ensure PTM positions are visible in linear plot (frequency mode)
    if st.session_state.is_frequency:
        for positions in ptm_positions.values():
            for pos in positions:
                if residue_vals[pos] is None:
                    residue_vals[pos] = 0  # ← only for visualization!

    # Convert sets to sorted lists
    for k in ptm_positions:
        ptm_positions[k] = sorted(list(ptm_positions[k]))

    return residue_vals, ptm_positions

def generate_colormap(residue_vals, cmap_name='autumn',
                      not_mapped_color='#d3d3d3',
                      vmin=None, vmax=None):
    cmap = colormaps[cmap_name]
    vals = [v for v in residue_vals if v is not None]
    if not vals:
        vmin, vmax = 0, 1
    else:
        if vmin is None or vmax is None:
            vmin, vmax = min(vals), max(vals)
        # (optional) add a tiny margin
        margin = max(0.01, (vmax - vmin) * 0.05)
        vmin -= margin
        vmax += margin

    hex_colors = []
    for val in residue_vals:
        if val is None:
            hex_colors.append(not_mapped_color)
        else:
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            rgb = cmap(norm)[:3]
            hex_colors.append(mcolors.rgb2hex(rgb))
    return hex_colors, vmin, vmax
def _z_range(residue_vals):
    """Return min/max of non-None Z-scores with a tiny margin."""
    valid = [v for v in residue_vals if v is not None]
    if not valid:
        return 0, 1
    mn, mx = min(valid), max(valid)
    margin = max(0.01, (mx - mn) * 0.05)
    return mn - margin, mx + margin
def extract_plddt_and_model(pdb_str, protein_seq):
    parser = PDBParser(QUIET=True)
    pdb_io = io.StringIO(pdb_str)
    structure = parser.get_structure('model', pdb_io)
    model_name = "Unknown Model"
    plddt_list = [None] * len(protein_seq)
    if 'HEADER' in pdb_str:
        header_match = re.search(r'HEADER\s+\S+\s+\S+\s+(.+?)\s+\d{2}', pdb_str, re.IGNORECASE)
        if header_match:
            model_name = header_match.group(1).strip()
    else:
        model_name = "AF-" + base_id + "-F1-model_v6" if 'base_id' in globals() else "AlphaFold Model"
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    res_id = residue.id[1] - 1
                    if 0 <= res_id < len(protein_seq):
                        b_factor = residue['CA'].get_bfactor()
                        plddt_list[res_id] = b_factor
                    #st.write(f"PDB residue: {res_id}, chain: {chain.id}, pLDDT: {b_factor}")  # Debug
    valid_plddt = [v for v in plddt_list if v is not None]
    mean_plddt = np.mean(valid_plddt) if valid_plddt else None
    return plddt_list, model_name, mean_plddt
def add_ptm_spheres(viewer_index, ptm_dict, condition_name, view,
                    coords_map=None, pdb_str=None):
    """
    Fixed version: ONE sphere per modified residue. No double nesting.
    """
    if not ptm_dict:
        return

    for unimod, info in ptm_dict.items():
        # Normalize input
        if isinstance(info, (list, set, tuple)):
            positions = list(info)
            selected = True
            color = "#FF0000"
            label = unimod
        elif isinstance(info, dict):
            positions = info.get("positions", [])
            if isinstance(positions, dict):
                positions = positions.get("positions", [])
            positions = list(positions)
            selected = info.get("selected", True)
            color = info.get("color", "#FF0000")
            label = info.get("label", unimod)
        else:
            continue

        if not selected or not positions:
            continue

        num = re.search(r"\d+", str(unimod))
        num = num.group() if num else "?"

        # ONE LOOP ONLY — try all methods until success
        for pos0 in positions:
            resi = str(int(pos0) + 1)
            added = False

            # 1. Try standard CA atom
            try:
                view.addSphere({
                    "center": {"resi": resi, "chain": "A", "atom": "CA"},
                    "radius": 2.2, "color": color, "alpha": 0.95
                }, viewer=viewer_index)
                view.addLabel(num, {
                    "fontSize": 15, "fontColor": color,
                    "backgroundColor": "white", "backgroundOpacity": 0.9
                }, {"resi": resi, "chain": "A"}, viewer=viewer_index)
                added = True
            except:
                pass  # CA not found → try next

            # 2. Try HETATM with same residue number (e.g. PTR, SEP, TPO)
            if not added and pdb_str:
                het_line = None
                for line in pdb_str.splitlines():
                    if line.startswith(("ATOM  ", "HETATM")) and line[22:26].strip() == resi:
                        het_line = line
                        break  # take first atom of the residue
                if het_line:
                    try:
                        x = float(het_line[30:38])
                        y = float(het_line[38:46])
                        z = float(het_line[46:54])
                        view.addSphere({
                            "center": {"x": x, "y": y, "z": z},
                            "radius": 2.6, "color": color, "alpha": 0.98
                        }, viewer=viewer_index)
                        view.addLabel(num, {
                            "fontSize": 16, "fontColor": color,
                            "backgroundColor": "white", "backgroundOpacity": 0.9
                        }, {"position": {"x": x, "y": y, "z": z}}, viewer=viewer_index)
                        added = True
                    except:
                        pass

            # 3. Final fallback: use pre-parsed CA coords
            if not added and coords_map and int(pos0) in coords_map:
                x, y, z, _ = coords_map[int(pos0)]
                view.addSphere({
                    "center": {"x": x, "y": y, "z": z},
                    "radius": 2.6, "color": color, "alpha": 0.98
                }, viewer=viewer_index)
                view.addLabel(num, {
                    "fontSize": 16, "fontColor": color,
                    "backgroundColor": "white", "backgroundOpacity": 0.9
                }, {"position": {"x": x, "y": y, "z": z}}, viewer=viewer_index)
    # Force re-render (sometimes needed)
    try:
        view.render()
    except Exception:
        pass
       
def normalize_ptm_data(ptm_dict):
        """Ensure ptm_data has flat 'positions': [1,2,3] and not nested dicts."""
        if not ptm_dict:
            return {}
        normalized = {}
        for unimod, info in ptm_dict.items():
            if isinstance(info, dict):
                positions = info.get('positions', [])
                # Handle double-nested case
                if isinstance(positions, dict) and 'positions' in positions:
                    positions = positions['positions']
                # Ensure it's a list
                if isinstance(positions, (list, set, tuple)):
                    positions = sorted(list(positions))
                else:
                    positions = []
                normalized[unimod] = {
                    'positions': positions,
                    'selected': info.get('selected', True),
                    'color': info.get('color', '#FF0000'),
                    'label': info.get('label', unimod)
                }
            else:
                # Fallback: treat as list of positions
                try:
                    normalized[unimod] = {
                        'positions': sorted(list(info)),
                        'selected': True,
                        'color': '#FF0000',
                        'label': unimod
                    }
                except:
                    continue
        return normalized
# In render_linear_plot function
def render_linear_plot(residue_vals, title, seq_len, vmin, vmax, protein_seq, model_name, plddt_list, mean_plddt,
                       cmap_name='viridis', not_mapped_color='#BEFDF9', highlight_residues=[], ptm_data=None,show_full_header=True,show_ptm_legend=True):
    #Compute Z-score range (dynamic for Frequency, fixed for Intensity)
    if st.session_state.is_frequency:
        vmin, vmax = _z_range(residue_vals)          # <-- dynamic
    else:
        vmin, vmax = -3, 3                           # <-- classic intensity scale
    hex_colors, _, _ = generate_colormap(residue_vals, cmap_name, not_mapped_color,vmin=vmin,vmax=vmax)
    mapped = [i for i, v in enumerate(residue_vals) if v is not None]
    mapped_js = str(mapped)
    total_svg_width = 1200.0  # virtual canvas width in pixels (will scale to container)
    bar_height = 30
    lollipop_extra = 60
    bar_y_start = lollipop_extra
    # scale (pixels per residue in the virtual canvas)
    scale = (total_svg_width / seq_len) if seq_len > 0 else 1.0
    total_width = total_svg_width
    # place numeric labels below the bar (offset from bar_y_start)
    label_offset = bar_y_start + bar_height + 15
    # === SVG canvas size ===
    #svg_top_padding = lollipop_extra + 5
    svg_bottom_padding = 40
    total_height = label_offset + 20
    container_height = total_height + 120 + svg_bottom_padding
    #total_height = label_offset + 20 + lollipop_extra
    bars = ""
    for i in range(seq_len):
        x = i * scale
        color = hex_colors[i]
        width = max(1.0, scale)
        aa = protein_seq[i] if i < len(protein_seq) else 'X'
        z_val = f"{residue_vals[i]:.2f}" if residue_vals[i] is not None else "N/A"
        tooltip = f"Pos {i+1} ({aa}): Z-Score={z_val}"
        is_mapped = i in mapped
        mapped_attr = 'True' if is_mapped else 'False'
        bars += f'<rect x="{x}" y="{bar_y_start}" width="{width}" height="{bar_height}" fill="{color}" '
        bars += f'stroke="#666" stroke-width="0.5" data-pos="{i}" data-mapped="{mapped_attr}" title="{tooltip}" />'

        # --------------------------------------------------------------
        #  LOLLIPOP PTM MARKERS – collision-aware stacking above the bar
        # --------------------------------------------------------------
        ptm_lines = ""
        ptm_labels = ""

        # Lollipop layout (relative to bar at y = bar_y_start)
        stem_height = 22
        circle_radius = 3.0
        # Place candy just above the bar (small gap) so it visually sits on the bar
        candy_y = bar_y_start - circle_radius - 2
        base_label_y = candy_y - circle_radius - 4
        font_size = 12 if seq_len <= 600 else 9

        def _flat(info):
            p = info.get("positions", [])
            if isinstance(p, dict) and "positions" in p:
                p = p["positions"]
            if isinstance(p, (list, set, tuple)):
                return [int(x) for x in p if x is not None]
            try:
                return [int(p)]
            except Exception:
                return []

        # Collect markers first so we can do overlap detection and stacking
        markers = []  # each: {'x': float, 'txt': str, 'color': str, 'pos0': int}
        if ptm_data:
            for unimod, info in ptm_data.items():
                if not info.get("selected", True):
                    continue
                color = info.get("color", "#FF0000")
                label = info.get("label", unimod)
                for pos0 in _flat(info):
                    try:
                        x = pos0 * scale
                    except Exception:
                        continue
                    # Use the raw position as stored in ptm_data (0-based) so labels match debug output
                    txt = f"{pos0}"
                    markers.append({'x':float(x),'txt': txt, 'color': color, 'pos0': int(pos0)})
        placed_levels = []  # list of lists of x positions per level
        level_assignments = []  # parallel to markers
        # Estimate horizontal threshold in pixels: depends on text length and font size
        for m in markers:
            txt_len = max(4, len(m['txt']))
            # approximate char width multiplier; scaled by pixel_per_res for narrow plots
            approx_char_w = font_size * 0.6
            th = max(12, approx_char_w * txt_len)
            # find lowest level where this marker doesn't collide
            level = 0
            while True:
                if level >= len(placed_levels):
                    placed_levels.append([m['x']])
                    level_assignments.append(level)
                    break
                else:
                    collide = False
                    for ox in placed_levels[level]:
                        if abs(ox - m['x']) < th:
                            collide = True
                            break
                    if not collide:
                        placed_levels[level].append(m['x'])
                        level_assignments.append(level)
                        break
                    level += 1

        # Now build SVG pieces using assigned levels
        label_spacing = max(14, int(font_size * 1.5))
        for m, lvl in zip(markers, level_assignments):
            x = m['x'] + (scale / 2.0)
            # start the stem from the exact bottom of the bar so it visibly emerges from the bar
            stem_y1 = bar_y_start + bar_height
            # stem top should reach the bottom of the candy circle
            stem_y2 = candy_y + circle_radius
            ptm_lines += f'<line x1="{x}" y1="{stem_y1}" x2="{x}" y2="{stem_y2}" ' \
                        f'stroke="{m["color"]}" stroke-width="1.4"/>'

            ptm_lines += f'<circle cx="{x}" cy="{candy_y}" r="{circle_radius}" ' \
                        f'fill="{m["color"]}" stroke="{m["color"]}" stroke-width="1.0"/>'

            # label y depends on level (higher level => more negative y)
            label_y = base_label_y - (lvl * label_spacing)
            # center text and make it bold for visibility
            safe_txt = m['txt']
            ptm_labels += f'<text x="{x}" y="{label_y}" font-size="{font_size}" ' \
                          f'fill="{m["color"]}" text-anchor="middle" font-weight="bold">{safe_txt}</text>'
    label_step = max(1, int(50 / scale) if seq_len>100 else 5)
    labels = ""
    for i in range(0, seq_len, label_step):
        x = i * scale + (scale / 2)
        labels += f'<text x="{x}" y="{label_offset}" font-size="{12 if seq_len<=500 else 10}" text-anchor="middle" fill="#333">{i+1}</text>'
    mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
    if show_full_header:
        if st.session_state.is_frequency:
            title_html = f'<div style="text-align:center; font-size:18px;margin-bottom:5px; font-weight:bold;color:#87CEEB;">{title}<br><span style="font-size:12px; color:#87CEEB;"> Frequency (0-1) | Mean pLDDT: {mean_plddt_display}</span></div>'
        else:
            title_html = f'<div style="text-align:center; font-size:18px; margin-bottom:5px; font-weight:bold; color:#87CEEB;">{title}<br><span style="font-size:12px; color:#87CEEB;">{model_name} | Mean pLDDT: {mean_plddt_display}</span></div>'
    else:
        title_html=f'<div style="text-align:center; font-size:18px;margin-bottom:5px; font-weight:bold;color:#87CEEB;">{title}</div>'
        # Build a legend for PTM colors (show UniMod id -> color) so users can relate colors chosen in PTM config
    legend_html = ""
    if show_ptm_legend:
        try:
            if ptm_data and isinstance(ptm_data, dict):
                items = []
                for unimod, info in ptm_data.items():
                    if not info.get("selected", True):
                        continue
                    color = info.get("color", "#FF0000")
                    # Prefer explicit label if provided; otherwise show numeric unimod (strip prefix if present)
                    label = info.get("label", unimod)
                    # normalize label text
                    lab_text = str(label).replace("UniMod:", "").strip()
                    items.append((lab_text, color))
                if items:
                    # build a horizontal legend with small color swatches
                    legend_items_html = "".join([
                        f'<div style="display:flex; align-items:center; margin-right:12px;">'
                        f'<div style="width:14px; height:14px; background:{c}; border:1px solid #444; margin-right:6px; border-radius:2px;"></div>'
                        f'<div style="font-size:12px; color:#ffff;">UniMod:{lbl}</div>'
                        f'</div>' for lbl, c in items
                    ])
                    legend_html = f'<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin-bottom:6px;">{legend_items_html}</div>'
        except Exception:
            legend_html = ""
    extra_top_space = 30
    # Draw bars first, then PTM lollipops and labels on top so markers are visible
    svg = f'<svg viewBox="0 0 {int(total_width + 40)} {int(total_height + extra_top_space + 40)}" width="100%" preserveAspectRatio="xMidYMid meet" ' \
        f'style="overflow:visible; background:#fff; border:1px solid #ddd; border-radius:6px; padding:20px;">' \
        f'{bars}{ptm_lines}{ptm_labels}{labels}</svg>'
    container_html = f'<div style="overflow-x:auto; max-width:100%; margin:10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{svg}</div>'
    # combine title, legend and svg
    # legend_html may be empty if no PTMs configured
    js = f"""
    <script>
    const mapped = {mapped_js};
    const rects = document.querySelectorAll('rect[data-pos]');
    rects.forEach(el => {{
    const pos = parseInt(el.getAttribute('data-pos'));
    const isMapped = mapped.includes(pos);
    if (isMapped) {{
        el.style.cursor = 'pointer';
        el.addEventListener('click', e => {{
        const pos = parseInt(e.target.getAttribute('data-pos'));
        window.parent.postMessage({{ type: 'SELECT_RESIDUE', residue: pos }}, '*');
        }});
    }}
    el.addEventListener('mouseover', e => {{
        if (e.target.title) {{
        e.target.style.opacity = '0.7';
        e.target.style.strokeWidth = '1';
        }}
    }});
    el.addEventListener('mouseout', e => {{
        e.target.style.opacity = '1';
        e.target.style.strokeWidth = '0.5';
    }});
    // ---- NEW: visual highlight of the clicked bar ----
    const prev = document.querySelector('.clicked-residue');
    if (prev) prev.classList.remove('clicked-residue');
    el.classList.add('clicked-residue');
    // ------------------------------------------------
    }});
    // ---- CSS for highlight ----
    const style = document.createElement('style');
    style.innerHTML = `
    .clicked-residue {{ filter: brightness(1.3); stroke: #FFD700; stroke-width: 3; }}
        `;
    document.head.appendChild(style);
    </script>
    """
    st.components.v1.html(title_html + legend_html + container_html + js, height=container_height+20)
    return None,None

def render_synced_viewers(pdb_str, residue_vals_list, bg_color, title_list, cmap_name='autumn', not_mapped_color='#d3d3d3', ptm_data_list=None):
    num_conditions = len(residue_vals_list)
    if num_conditions < 2 or num_conditions > 4:
        st.error(f"Unsupported number of conditions: {num_conditions}. Must be 2-4.")
        return

    # Determine grid based on number of conditions
    if num_conditions == 2:
        grid = (1, 2)
    elif num_conditions == 3:
        grid = (1, 3)
    elif num_conditions == 4:
        grid = (2, 2)

    # Compute global vmin/vmax across all conditions
    all_vals = [v for res in residue_vals_list for v in res if v is not None]
    if st.session_state.is_frequency:
        vmin= min(all_vals, default=0)
        vmax= max(all_vals,default= 1)
        margin = max(0.01, (vmax - vmin) * 0.05)
        vmin -= margin
        vmax += margin
    else:
        vmin, vmax = -3, 3

    # Generate hex_colors for each condition using global vmin/vmax
    hex_colors_list = []
    for residue_vals in residue_vals_list:
        hex_colors, _, _ = generate_colormap(residue_vals, cmap_name, not_mapped_color, vmin=vmin, vmax=vmax)
        hex_colors_list.append(hex_colors)

    # Residues JS for each
    residues_js_list = [json.dumps([i for i, v in enumerate(res) if v is not None]) for res in residue_vals_list]

    # Create view with grid
    view = py3Dmol.view(width='100vw', height='400px', viewergrid=grid, linked=True)

    # Add models to each viewer
    for idx in range(num_conditions):
        row = idx // grid[1]
        col = idx % grid[1]
        view.addModel(pdb_str, 'pdb', viewer=(row, col))

    bg_color_map = {'white': '#FFFFFF', 'black': '#000000', 'darkgrey': '#4A4A4A'}
    bg_color_hex = bg_color_map.get(bg_color.lower(), '#000000')

    # Set background and initial style for each viewer
    for idx in range(num_conditions):
        row = idx // grid[1]
        col = idx % grid[1]
        view.setBackgroundColor(bg_color_hex, viewer=(row, col))
        view.setStyle({}, {'cartoon': {'color': 'lightgray'}}, viewer=(row, col))

    # Apply colors
    cmap = colormaps[cmap_name]
    norm = Normalize(vmin=vmin, vmax=vmax)

    def apply_colors(viewer_idx, residue_vals, hex_colors):
        row = viewer_idx // grid[1]
        col = viewer_idx % grid[1]
        for i, val in enumerate(residue_vals):
            if val is None:
                color = not_mapped_color
            else:
                rgb = cmap(norm(val))[:3]
                color = mcolors.rgb2hex(rgb)
            view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': color}}, viewer=(row, col))

    for idx, (residue_vals, hex_colors) in enumerate(zip(residue_vals_list, hex_colors_list)):
        apply_colors(idx, residue_vals, hex_colors)

    # PTM spheres
    coords_map = {}
    try:
        parser = PDBParser(QUIET=True)
        pdb_io_local = io.StringIO(pdb_str)
        struct = parser.get_structure('tmp', pdb_io_local)
        for model in struct:
            for chain in model:
                for residue in chain:
                    try:
                        if 'CA' in residue:
                            res_index0 = residue.id[1] - 1
                            ca = residue['CA'].get_coord()
                            coords_map[res_index0] = (float(ca[0]), float(ca[1]), float(ca[2]), chain.id)
                    except Exception:
                        continue
    except Exception as e:
        print(f"render_synced_viewers: failed to parse PDB for coords_map: {e}")

    # ——————————————————————————————————————————————————
    # MERGE ALL PTMs ACROSS CONDITIONS → ONE sphere per site
    # ——————————————————————————————————————————————————
    merged_ptm_dict = {}

    for cond_ptm_dict in ptm_data_list:
        if not cond_ptm_dict:
            continue
        for unimod, info in cond_ptm_dict.items():
            # Normalize input format
            if isinstance(info, (list, set, tuple)):
                positions = list(info)
                color = "#FF0000"
                selected = True
                label = unimod
            elif isinstance(info, dict):
                positions = info.get("positions", [])
                if isinstance(positions, dict):
                    positions = positions.get("positions", [])
                positions = list(positions)
                selected = info.get("selected", True)
                color = info.get("color", "#FF0000")
                label = info.get("label", unimod)
            else:
                continue

            if not selected or not positions:
                continue

            if unimod not in merged_ptm_dict:
                merged_ptm_dict[unimod] = {
                    "positions": set(),
                    "color": color,
                    "label": label,
                    "selected": True
                }
            merged_ptm_dict[unimod]["positions"].update(positions)

    # Convert sets → sorted lists
    for unimod in merged_ptm_dict:
        merged_ptm_dict[unimod]["positions"] = sorted(merged_ptm_dict[unimod]["positions"])

    # ——————————————————————————————————
    # DRAW THE MERGED PTMs ON EVERY VIEWER (once per site!)
    # ——————————————————————————————————
    for idx in range(num_conditions):
        row = idx // grid[1]
        col = idx % grid[1]
        add_ptm_spheres(
            viewer_index=(row, col),
            ptm_dict=merged_ptm_dict,           # ← same dict for all viewers
            condition_name="All Conditions",    # or title_list[idx] if you prefer
            view=view,
            coords_map=coords_map,
            pdb_str=pdb_str
        )

    # Zoom to each viewer
    for idx in range(num_conditions):
        row = idx // grid[1]
        col = idx % grid[1]
        #view.setStyle({}, {"cartoon":{"opacity":0.85}}, viewer=(row, col))
        view.zoomTo(viewer=(row, col))

    view.render()

    # JS for hover and pick
    hover_js = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        function tryInitViewers(retryCount = 8, delay = 400) {{
            try {{
                let viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
                if (!viewerElems || viewerElems.length < {num_conditions}) {{
                    if (retryCount > 0) {{
                        console.warn('Not enough viewer elements found (' + (viewerElems ? viewerElems.length : 0) + '/{num_conditions}), retrying in ' + delay + 'ms');
                        setTimeout(() => tryInitViewers(retryCount - 1, delay), delay);
                    }} else {{
                        console.error('Failed to find enough viewer elements after retries');
                    }}
                    return;
                }}

                let viewers = [];
                try {{
                    if (window.$3Dmol && typeof $3Dmol.getViewer === 'function') {{
                        for (let i = 0; i < {num_conditions}; i++) {{
                            viewers.push($3Dmol.getViewer(viewerElems[i]));
                        }}
                    }} else {{
                        for (let i = 0; i < {num_conditions}; i++) {{
                            viewers.push(viewerElems[i].querySelector('div > canvas')?.parentElement?.viewer);
                        }}
                    }}
                }} catch (err) {{
                    console.warn('Error resolving viewers', err);
                }}

                if (viewers.length < {num_conditions} || viewers.some(v => !v)) {{
                    if (retryCount > 0) {{
                        setTimeout(() => tryInitViewers(retryCount - 1, delay), delay);
                    }} else {{
                        console.error('Viewers not resolved');
                    }}
                    return;
                }}

                const residues_list = {json.dumps(residues_js_list)};

                function handlePick(viewer_idx) {{
                    return function(atom, event) {{
                        if (!atom) return;
                        const resi = parseInt(atom.resi, 10) - 1;
                        if (residues_list[viewer_idx].includes(resi)) {{
                            window.parent.postMessage({{ type: 'SELECT_RESIDUE', residue: resi }}, '*');
                        }}
                    }}
                }}

                for (let i = 0; i < {num_conditions}; i++) {{
                    viewers[i].setClickable({{}}, true, handlePick(i));
                }}

            }} catch(e) {{
                if (retryCount > 0) {{
                    setTimeout(() => tryInitViewers(retryCount - 1, delay), delay);
                }}
            }}
        }}
        tryInitViewers();
    }});
    </script>
    """

    # Listener JS
    listener_js = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        let previous_selected = null;
        const observer = new MutationObserver(() => {{
            const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
            if (viewerElems.length >= {num_conditions}) {{
                observer.disconnect();
            }}
        }});
        observer.observe(document.body, {{ childList: true, subtree: true }});
        
        window.addEventListener("message", (event) => {{
            if (event.data && event.data.type === "SELECT_RESIDUE") {{
                const residue = event.data.residue;
                try {{
                    const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
                    if (viewerElems.length < {num_conditions}) {{
                        setTimeout(() => window.dispatchEvent(new MessageEvent("message", {{ data: event.data }})), 100);
                        return;
                    }}
                    let viewers = [];
                    for (let i = 0; i < {num_conditions}; i++) {{
                        viewers.push(viewerElems[i].querySelector('div > canvas').parentElement.viewer);
                    }}
                    
                    if (previous_selected !== null) {{
                        const prev_span = document.querySelector(`.aa[data-pos="${{previous_selected}}"]`);
                        if (prev_span) {{
                            prev_span.style.backgroundColor = "";
                            prev_span.style.fontWeight = "";
                        }}
                        const prev_bars = document.querySelectorAll(`rect[data-pos="${{previous_selected}}"]`);
                        prev_bars.forEach(bar => {{
                            bar.style.stroke = "none";
                            bar.style.strokeWidth = "0";
                        }});
                        viewers.forEach(v => {{
                            v.removeAllShapes();
                            v.render();
                        }});
                    }}
                    
                    const span = document.querySelector(`.aa[data-pos="${{residue}}"]`);
                    if (span) {{
                        span.style.backgroundColor = "yellow";
                        span.style.fontWeight = "bold";
                    }}
                    const bars = document.querySelectorAll(`rect[data-pos="${{residue}}"]`);
                    bars.forEach(bar => {{
                        bar.style.stroke = "red";
                        bar.style.strokeWidth = "2";
                    }});
                    
                    const resi_str = (residue + 1).toString();
                    const spec = {{center: {{resi: resi_str, atom: 'CA'}}, radius: 5.0, color: 'red', alpha: 0.6}};
                    viewers.forEach(v => {{
                        v.addSphere(spec);
                        v.center({{resi: resi_str, atom: 'CA'}});
                        v.render();
                    }});
                    previous_selected = residue;
                }} catch (e) {{
                    console.error("Error adding 3D highlight:", e);
                }}
            }}
        }});
    }});
    </script>
    """

    html = view._make_html()
    titles_str = " | ".join(title_list)
    st.markdown(f"#### {titles_str}")
    st.components.v1.html(html + hover_js, height=420)
    st.components.v1.html(listener_js, height=0)

def create_download_zip(
    protein_of_interest: str,
    pdb_str: str,
    peptide_data: Dict[str, pd.DataFrame],
    residue_data: Dict[str, List[Optional[float]]],
    conditions: List[str],
    min_max_logs: Dict[str, Tuple[float, float]],
    seq_len: int,
    cmap_name: str = 'autumn',
    not_mapped_color: str = '#d3d3d3',
    ptm_data: Optional[Dict[str, Dict]] = None,
    selected_df: Optional[pd.DataFrame] = None,
    protein_seq: Optional[str] = None,
    apply_tryptic: Optional[bool] = None,
    metadata: Optional[pd.DataFrame] = None,
) -> io.BytesIO:
    """
    Packs every artefact the user can see in the UI into a single ZIP.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:

        # ------------------------------------------------------------------
        # 1. PDB + peptide CSVs (unchanged)
        # ------------------------------------------------------------------
        zipf.writestr(f"{protein_of_interest}_protein.pdb", pdb_str)
        for cond in conditions:
            csv_bytes = peptide_data[cond].to_csv(index=False).encode()
            zipf.writestr(f"{protein_of_interest}_{cond}_peptides.csv", csv_bytes)

        # ------------------------------------------------------------------
        # 2. PyMOL scripts (unchanged)
        # ------------------------------------------------------------------
        cmap = colormaps[cmap_name]
        for cond in conditions:
            pml = f"load {protein_of_interest}_protein.pdb\nhide everything\nshow cartoon\ncolor gray90, all\nzoom\n"
            mn, mx = min_max_logs[cond]
            for i in range(seq_len):
                if residue_data[cond][i] is not None:
                    norm = (residue_data[cond][i] - mn) / (mx - mn) if mx > mn else 0.5
                    col_hex = mcolors.rgb2hex(cmap(norm)[:3])
                    pml += f"color {col_hex}, resi {i+1}\n"
            # PTM spheres
            if ptm_data and ptm_data.get(cond):
                for uni, info in ptm_data[cond].items():
                    if info.get('selected', True):
                        col_hex = info.get('color', '#FF0000')
                        for pos0 in info.get('positions', []):
                            resi = pos0 + 1
                            pml += f"pseudoatom ptm_{uni}_{resi}, resi {resi} and name CA\n"
                            pml += f"show spheres, ptm_{uni}_{resi}\n"
                            pml += f"set sphere_scale, 5.0, ptm_{uni}_{resi}\n"
                            pml += f"color {col_hex}, ptm_{uni}_{resi}\n"
            zipf.writestr(f"{protein_of_interest}_{cond}_pymol_script.pml", pml)

        # ------------------------------------------------------------------
        # 3. Linear JPEGs (unchanged)
        # ------------------------------------------------------------------
        for cond in conditions:
            w = min(25, max(10, seq_len / 20))
            fig, ax = plt.subplots(figsize=(w, 1), dpi=600)
            ax.add_patch(patches.Rectangle((0, 0), seq_len, 1,
                                           facecolor=not_mapped_color, edgecolor='none'))
            mn, mx = min_max_logs[cond]
            for i in range(seq_len):
                if residue_data[cond][i] is not None:
                    norm = (residue_data[cond][i] - mn) / (mx - mn) if mx > mn else 0.5
                    ax.add_patch(patches.Rectangle((i, 0), 1, 1,
                                                   facecolor=cmap(norm)[:3], edgecolor='none'))
            ax.set_xlim(0, seq_len); ax.set_ylim(0, 1); ax.set_yticks([])
            ax.set_xlabel(f'Amino Acid Position ({cond})', fontsize=30)
            ax.tick_params(axis='x', labelsize=15)
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg', dpi=600, bbox_inches='tight')
            plt.close(fig)
            suffix = "_freq" if st.session_state.is_frequency else ""
            zipf.writestr(f"{protein_of_interest}_{cond}{suffix}_linear.jpeg", buf.getvalue())

        # ------------------------------------------------------------------
        # 4. PTM-position CSV  (the part that was broken)
        # ------------------------------------------------------------------
        if selected_df is not None and protein_seq is not None and metadata is not None:
            # ---- build a map: sample column → group name -----------------
            sample_to_group = dict(zip(metadata['File_Name'], metadata['Group']))

            ptm_rows = []
            for _, row in selected_df.iterrows():
                prot = row['Protein.Group']
                stripped = row.get('Stripped.Sequence', 'NA')
                if pd.isna(stripped):
                    stripped = 'NA'

                # ---- intensities per *selected* condition ----------------
                intens = [row.get(c, 'NA') for c in conditions]

                # ---- decide which column holds the modification ----------
                mod_seq = ''
                if 'Modified.Sequence' in row and pd.notna(row['Modified.Sequence']):
                    mod_seq = str(row['Modified.Sequence'])
                # (no else – if there is no Modified.Sequence we treat it as unmodified)

                # ---- PTM case ------------------------------------------------
                if mod_seq and '(UniMod:' in mod_seq:
                    cleaned, mods = clean_and_find_mods(mod_seq)
                    if cleaned != stripped:
                        ptm_rows.append([prot, stripped] + intens +
                                        [mod_seq, 'Mismatch', 'NA', 'NA', 'NA'])
                        continue

                    matches = list(re.finditer(re.escape(cleaned), protein_seq))
                    valid = False
                    for m in matches:
                        start = m.start()
                        if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                            continue
                        valid = True
                        pep_start = start + 1
                        pep_end   = start + len(cleaned)
                        for rel_pos, uni_num in mods:
                            abs_pos = start + rel_pos + 1
                            ptm_rows.append([prot, stripped] + intens +
                                            [mod_seq, abs_pos, f"UniMod:{uni_num}",
                                             pep_start, pep_end])
                    if not valid:
                        for rel_pos, uni_num in mods:
                            ptm_rows.append([prot, stripped] + intens +
                                            [mod_seq, 'No valid tryptic position',
                                             f"UniMod:{uni_num}", 'NA', 'NA'])
                # ---- non-PTM case -------------------------------------------
                else:
                    matches = list(re.finditer(re.escape(stripped), protein_seq)) if stripped != 'NA' else []
                    valid = False
                    for m in matches:
                        start = m.start()
                        if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                            continue
                        valid = True
                        pep_start = start + 1
                        pep_end   = start + len(stripped) if stripped != 'NA' else 'NA'
                        ptm_rows.append([prot, stripped] + intens +
                                        [mod_seq, 'NA', 'NA', pep_start, pep_end])
                    if not valid:
                        ptm_rows.append([prot, stripped] + intens +
                                        [mod_seq, 'No valid tryptic position',
                                         'NA', 'NA', 'NA'])

            # ---- column header (exact match with the rows) -------------
            cols = (['Protein.Group', 'Stripped.Sequence'] +
                    conditions +
                    ['Modified.Sequence', 'PTM_position', 'UniMod_Type',
                     'Peptide_Start', 'Peptide_End'])

            ptm_df = pd.DataFrame(ptm_rows, columns=cols)
            zipf.writestr(f"{protein_of_interest}_modification_positions.csv",
                          ptm_df.to_csv(index=False).encode())

        # ------------------------------------------------------------------
        # 5. Colour-bar PNG
        # ------------------------------------------------------------------
        vmin = min(min_max_logs[c][0] for c in conditions)
        vmax = max(min_max_logs[c][1] for c in conditions)
        fig, ax = plt.subplots(figsize=(8, 0.3))
        norm = Normalize(vmin=vmin, vmax=vmax)
        ScalarMappable(cmap=colormaps[cmap_name], norm=norm)
        cbar = fig.colorbar(
            ScalarMappable(cmap=colormaps[cmap_name], norm=norm),
            cax=ax, orientation='horizontal')
        cbar.set_label('Frequency-Z' if st.session_state.is_frequency else 'Z-Score Intensity',
                       fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        zipf.writestr(f"{protein_of_interest}_colorbar.png", buf.getvalue())
        plt.close(fig)

        # ------------------------------------------------------------------
        # 6. DisProt JSON + sequence TXT
        # ------------------------------------------------------------------
        base_id = protein_of_interest.split('-')[0]
        zipf.writestr(f"{protein_of_interest}_disprot_info.json",
                      json.dumps(get_disprot_info(base_id), indent=2).encode())
        if protein_seq:
            zipf.writestr(f"{protein_of_interest}_sequence.txt", protein_seq)

        # ------------------------------------------------------------------
        # 7. Example quantitative plots (intensity + frequency)
        # ------------------------------------------------------------------
        if selected_df is not None and metadata is not None:
            # pick the first non-NA stripped peptide as an example
            example_pep = selected_df['Stripped.Sequence'].dropna().iloc[0] if not selected_df.empty else None
            if example_pep:
                # ---- sample → group map (already built above) ------------
                sample_to_group = dict(zip(metadata['File_Name'], metadata['Group']))

                # ---- intensity box-plot (log10) -------------------------
                plot_rows = []
                for sample_col in sample_to_group.keys():
                    if sample_col not in selected_df.columns:
                        continue
                    vals = selected_df.loc[selected_df['Stripped.Sequence'] == example_pep, sample_col]
                    for v in vals.dropna():
                        plot_rows.append([example_pep, sample_to_group[sample_col], sample_col, v])
                if plot_rows:
                    df_int = pd.DataFrame(plot_rows,
                                          columns=['Peptide', 'Group', 'Sample', 'Intensity'])
                    df_int = df_int.groupby(['Peptide', 'Group', 'Sample'], as_index=False)['Intensity'].mean()
                    df_int['Intensity'] = np.log10(df_int['Intensity'] + 1)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(data=df_int, x='Group', y='Intensity', hue='Group', ax=ax,
                                palette='tab10')
                    ax.set_title(f'Example intensity – {example_pep}')
                    ax.set_ylabel('log10(Intensity+1)')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    zipf.writestr(f"{protein_of_interest}_example_intensity.png", buf.getvalue())
                    plt.close(fig)

                # ---- frequency bar-plot ---------------------------------
                freq_rows = []
                for grp in conditions:
                    samples = [s for s in sample_to_group.keys()
                               if sample_to_group.get(s) == grp and s in selected_df.columns]
                    if not samples:
                        freq_rows.append([grp, np.nan])
                        continue
                    detected = sum((selected_df.loc[
                        selected_df['Stripped.Sequence'] == example_pep, s] > 0).any()
                                   for s in samples)
                    freq = detected / len(samples)
                    freq_rows.append([grp, freq])
                if freq_rows:
                    df_freq = pd.DataFrame(freq_rows, columns=['Group', 'Frequency'])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(data=df_freq, x='Group', y='Frequency',hue='Group', ax=ax, palette='tab10')
                    ax.set_title(f'Example frequency – {example_pep}')
                    ax.set_ylim(0, 1)
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    zipf.writestr(f"{protein_of_interest}_example_frequency.png", buf.getvalue())
                    plt.close(fig)

    zip_buffer.seek(0)
    return zip_buffer
  
def render_single_3d_viewer(
    pdb_str,
    residue_vals,
    title,
    cmap_name_q='autumn',
    not_mapped_color='#d3d3d3',
    ptm_data=None,
    bg_color='black'
):
    """
    Single 3D viewer with:
    - Z-score / frequency coloring
    - PTM spheres with UniMod numbers
    - Click → highlights residue in linear plot (and vice versa via JS listener)
    - Same look as your multi-viewer version
    """
    # === 1. Compute colors with proper vmin/vmax ===
    if st.session_state.get("is_frequency", False):
        vals = [v for v in residue_vals if v is not None]
        vmin = min(vals) if vals else 0
        vmax = max(vals) if vals else 1
        margin = max(0.01, (vmax - vmin) * 0.05)
        vmin -= margin
        vmax += margin
    else:
        vmin, vmax = -3, 3

    hex_colors, _, _ = generate_colormap(
        residue_vals, cmap_name_q, not_mapped_color, vmin=vmin, vmax=vmax
    )

    # === 2. Create viewer ===
    view = py3Dmol.view(width=1000, height=600)

    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "lightgray"}})

    # Apply per-residue coloring
    for i, color in enumerate(hex_colors):
        view.setStyle({"resi": str(i + 1)}, {"cartoon": {"color": color}})

    # === 3. Extract CA coordinates for fallback PTM placement ===
    coords_map = {}
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("tmp", io.StringIO(pdb_str))
        for model in struct:
            for chain in model:
                for res in chain:
                    if "CA" in res:
                        idx = res.id[1] - 1
                        ca = res["CA"].coord
                        coords_map[idx] = (float(ca[0]), float(ca[1]), float(ca[2]))
    except:
        pass  # fallback will still work

    # === 4. Add PTM spheres (same logic as multi-viewer) ===
    if ptm_data:
        for unimod, info in ptm_data.items():
            positions = info.get("positions", []) if isinstance(info, dict) else info
            color = info.get("color", "#FF0000") if isinstance(info, dict) else "#FF0000"
            num_match = re.search(r"\d+", str(unimod))
            label = num_match.group() if num_match else "?"

            for pos0 in positions:
                resi_str = str(int(pos0) + 1)
                try:
                    # Try selector first
                    view.addSphere({
                        "center": {"resi": resi_str, "atom": "CA"},
                        "radius": 2.2,
                        "color": color,
                        "alpha": 0.95
                    })
                    view.addLabel(label, {
                        "fontSize": 14,
                        "fontColor": "white",
                        "backgroundColor": color,
                        "backgroundOpacity": 0.9
                    }, {"resi": resi_str})
                except:
                    # Fallback to coordinates
                    if pos0 in coords_map:
                        x, y, z = coords_map[pos0]
                        view.addSphere({
                            "center": {"x": x, "y": y, "z": z},
                            "radius": 2.5,
                            "color": color,
                            "alpha": 0.98
                        })
                        view.addLabel(label, {
                            "fontSize": 15,
                            "fontColor": "white",
                            "backgroundColor": color,
                            "backgroundOpacity": 0.9
                        }, {"position": {"x": x, "y": y, "z": z}})

    # === 5. Final style & zoom ===
    bg_hex = "#000000" if "black" in bg_color.lower() else "#FFFFFF"
    view.setBackgroundColor(bg_hex)
    view.zoomTo()
    view.zoom(1.4)

    # === 6. Enable clicking → send message to linear plot ===
    clickable_residues = json.dumps([i for i, v in enumerate(residue_vals) if v is not None])

    click_js = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        function initViewer() {{
            const container = document.querySelector('.viewer_3Dmoljs');
            if (!container || !container.viewer) {{
                setTimeout(initViewer, 300);
                return;
            }}
            const viewer = container.viewer;
            const residues = {clickable_residues};

            viewer.setClickable({{}}, true, function(atom) {{
                if (!atom) return;
                const resi = parseInt(atom.resi, 10) - 1;
                if (residues.includes(resi)) {{
                    window.parent.postMessage({{ type: 'SELECT_RESIDUE', residue: resi }}, '*');
                }}
            }});
        }}
        initViewer();
    }});
    </script>
    """

    # === 7. Render ===
    st.markdown(f"#### {title}")
    html_content = view._make_html() + click_js
    st.components.v1.html(html_content, height=650)

    # Listener (same for all single viewers — only one needed per page)
    if not hasattr(st.session_state, "single_viewer_listener_added"):
        st.session_state.single_viewer_listener_added = True
        st.components.v1.html("""
        <script>
        let previous_selected = null;
        window.addEventListener("message", (event) => {
            if (event.data && event.data.type === "SELECT_RESIDUE") {
                const residue = event.data.residue;

                // Clear previous
                if (previous_selected !== null) {
                    const prev_span = document.querySelector(`.aa[data-pos="${previous_selected}"]`);
                    if (prev_span) {
                        prev_span.style.backgroundColor = "";
                        prev_span.style.fontWeight = "";
                    }
                    document.querySelectorAll(`rect[data-pos="${previous_selected}"]`)
                        .forEach(r => { r.style.stroke = ""; r.style.strokeWidth = ""; });
                }

                // Highlight new
                const span = document.querySelector(`.aa[data-pos="${residue}"]`);
                if (span) {
                    span.style.backgroundColor = "yellow";
                    span.style.fontWeight = "bold";
                }
                document.querySelectorAll(`rect[data-pos="${residue}"]`)
                    .forEach(r => { r.style.stroke = "red"; r.style.strokeWidth = "3"; });

                previous_selected = residue;
            }
        });
        </script>
        """, height=0)
def render_zscore_colorbar(residue_vals,cmap_name="autumn",not_mapped_color='#d3d3d3'):
    """
    Beautiful fixed Z-score colorbar (-3 to +3) with integer ticks.
    Matches perfectly with render_linear_plot() and 3D viewer.
    Used for single-condition intensity analysis.
    """
    vmin, vmax = -3.0, 3.0

    fig = plt.figure(figsize=(8, 0.8))
    fig.patch.set_facecolor("#dfdfdf")
    ax = fig.add_axes([0.5, 0.8, 0.8, 0.5])  # [left, bottom, width, height]
    ax.set_facecolor('#0e1117')

    cmap = colormaps[cmap_name]
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal', ticks=[-3, -2, -1, 0, 1, 2, 3])

    # Styling
    cbar.set_label("Z-score Intensity", fontsize=15, fontweight='bold', color='white', labelpad=12)
    cbar.ax.tick_params(labelsize=12, colors='white', length=6, width=1.5)
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1.5)

    # Bold zero tick
    cbar.ax.set_xticklabels(['-3', '-2', '-1', '0', '1', '2', '3'],
                             fontweight='bold', fontsize=12)

    ax.axis('off')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
def extract_ptm_positions(stripped_seq, mod_seq):
    """
    Extract residue indices (0-based) of UniMod modifications within Modified.Sequence.
    Example: ACDEFGHIK -> ACDE(UniMod:21)FGHIK
    """
    positions = []
    seq_idx = 0
    i = 0
    while i < len(mod_seq):
        if mod_seq[i].isalpha():
            seq_idx += 1
            i += 1
        elif mod_seq[i] == '(':
            match = re.match(r'\(UniMod:(\d+)\)', mod_seq[i:])
            if match:
                unimod_id = match.group(1)
                positions.append((seq_idx - 1, f"UniMod:{unimod_id}"))
                i += len(match.group(0))
            else:
                i += 1
        else:
            i += 1
    return positions
import re

def find_peptide_position(protein_seq: str, peptide: str) -> str:
    """
    Return 'start-end' (1-based) or '-'.
    Handles:
      • Stripped peptides (plain AA)
      • Modified peptides with UniMod tags, e.g. 'AC(UniMod:1)DEF'
      • Lower-case modified letters (c, m, etc.)
    """
    # Normalise the peptide for searching
    # Remove the UniMod tag but KEEP the surrounding parentheses and the letter
    #   'AC(UniMod:1)DEF'  →  'AC()DEF'
    pep_no_tag = re.sub(r"UniMod:\d+", "", peptide)
    # Keep ONLY letters (A-Z, a-z) – this preserves lower-case modifications
    pep_clean = "".join(c for c in pep_no_tag if c.isalpha())
    if not pep_clean:
        return "-"
    # Normalise the protein sequence (upper-case, no gaps)
    prot_clean = "".join(c for c in protein_seq if c.isalpha()).upper()
    # Search – we look for the *exact* cleaned peptide (case-insensitive)
    pep_upper = pep_clean.upper()
    start = prot_clean.find(pep_upper)
    if start == -1:
        return "-"
    # 1-based positions
    start1 = start + 1
    end1   = start + len(pep_clean)
    return f"{start1}-{end1}"

def format_sequence_for_display(seq, residue_data, conditions, line_len=80, group=20):
    mapped_positions = set()
    for cond in conditions:
        mapped_positions.update(i for i, v in enumerate(residue_data[cond]) if v is not None)
    mapped_js = json.dumps(list(mapped_positions))
    lines = []
    seq_len = len(seq)
    # Render in fixed-width monospace using inline-block spans so numbering aligns exactly with residues.
    for start in range(0, seq_len, line_len):
        end = min(start + line_len, seq_len)
        segment = seq[start:end]
        # Number line: create blocks of width 'group' chars; place the group-start number at the left of each block
        num_line = "<div style='font-family: monospace; font-size: 10px; color: #888; display: block; margin-bottom:4px;'>"
        for i in range(0, len(segment), group):
            block_start = start + i + 1
            # width equals number of residues in this block
            block_size = min(group, len(segment) - i)
            # display the number at the start of the block and draw a small vertical tick under the number
            num_line += (
                f"<span style='display:inline-block; width:{block_size}ch; text-align:left; position:relative;'>"
                f"<span style='position:relative; display:inline-block;'>{block_start}</span>"
                f"<span style='position:absolute; left:0; top:100%; width:1px; height:8px; background:#888;'></span>"
                f"</span>"
            )
        num_line += "</div>"

        # Sequence line: each residue is a fixed-width inline-block so it lines up with the numbering above
        seq_line = "<div style='font-family: monospace; font-size: 12px; line-height: 1.5; display:block; position:relative;'>"
        for j, aa in enumerate(segment):
            abs_pos = start + j + 1
            style = "cursor:pointer;color:blue;" if abs_pos in mapped_positions else "color:gray;"
            seq_line += f"<span class='aa' data-pos='{abs_pos}' style='display:inline-block; width:1ch; {style}'>{aa}</span>"
        # place the end index at the right of the line so it's clearly aligned with the last residue
        seq_line += f"<span style='position:absolute; right:6px; top:0; font-weight:bold;'>{end}</span></div>"

        lines.append(num_line + seq_line)
    seq_html = "<div id='seq-panel' style='padding:10px; background:#fafafa; border-radius:6px; border:1px solid #ddd; white-space:nowrap; overflow:auto;'>" + "".join(lines) + "</div>"
    js = f"""
    <script>
    const mapped = {mapped_js};
    document.querySelectorAll('.aa').forEach(el => {{
        const pos = parseInt(el.getAttribute('data-pos'));
        if (mapped.includes(pos)) {{
            el.style.cursor = 'pointer';
            el.addEventListener('click', e => {{
                window.parent.postMessage({{ type: 'SELECT_RESIDUE', residue: pos }}, '*');
                console.log("Sequence click sent for pos:", pos);
            }});
        }}
    }});
    </script>
    """
    return seq_html + js

def sequence_copy_component(seq):
    seq_escaped = seq.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""
    <div style="display:flex; gap:10px; align-items:center;">
      <button id="copySeqBtn" style="padding:6px 10px; background:#2b8cff; color:white; border-radius:6px; border:none; cursor:pointer;">Copy sequence</button>
      <span id="copyMsg" style="color:green; font-size:13px; display:none;">Copied!</span>
    </div>
    <pre id="seqText" style="display:none;">{seq_escaped}</pre>
    <script>
      const btn = document.getElementById('copySeqBtn');
      const msg = document.getElementById('copyMsg');
      btn.addEventListener('click', () => {{
        const text = document.getElementById('seqText').innerText;
        navigator.clipboard.writeText(text).then(() => {{
          msg.style.display = 'inline';
          setTimeout(() => msg.style.display = 'none', 1500);
        }});
      }});
    </script>
    """
    return html

# --- Main App UI ---
html_content = """
<div style="position: relative; width: 100%; overflow: hidden; background-color: #1a1a2e; padding: 20px 0;">
    <h1 id="animated-title" style="font-family: 'Arial', sans-serif; font-size: 48px; color: #e94560; margin: 0; text-align: center; 
           background: linear-gradient(to right, #e94560, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
           position: relative; z-index: 1;">
        Peptide3D Mapper
    </h1>
    <div id="paint-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(to right, #00d4ff, #e94560, #ffffff); 
           z-index: 0; animation: paintEffect 2s ease-out forwards;">
    </div>
    <style>
        @keyframes paintEffect {
            0% { width: 0; }
            100% { width: 100%; opacity: 0; }
        }
        #animated-title {
            display: inline-block;
        }
        #paint-overlay {
            animation-fill-mode: forwards;
        }
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            setTimeout(() => {
                document.getElementById("paint-overlay").style.display = "none";
            }, 2000);
        });
    </script>
</div>
"""
st.components.v1.html(html_content, height=100)

st.markdown(
    """
    <p style='text-align: justify; font-size: 16px; color: #87CEEB;'>
   The Peptide3D Mapper is a web-based tool that visualizes peptide intensity data from proteomics experiments on 3D protein structures, highlighting post-translational modifications (PTMs) with 
   customizable annotations. Upload peptide CSV and FASTA files to compare conditions (e.g., control vs. disease) using z-score intensity scales, with options to apply tryptic cleavage rules and 
   handle protein isoforms. Automatically fetch AlphaFold structures with pLDDT confidence scores or upload custom PDB files. Explore residue-level differences through interactive 3D and linear 
   sequence views with clickable residue selection, customizable color schemes, and PhosphoSitePlus integration for PTM exploration. Export comprehensive outputs, including PDB files, PyMOL 
   scripts, peptide CSVs, linear plots, and PTM position data, for further analysis.
    </p>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'conditions_confirmed' not in st.session_state:
    st.session_state.conditions_confirmed = {'single': False, 'multiple': False}
if 'processed' not in st.session_state:
    st.session_state.processed = {'single': False, 'multiple': False}
if 'selected_residue' not in st.session_state:
    st.session_state.selected_residue = None
if 'all_unimods' not in st.session_state:
    st.session_state.all_unimods = []
if "selected_unimods" not in st.session_state:
    st.session_state.selected_unimods = []
if 'ptm_enabled' not in st.session_state:
    st.session_state.ptm_enabled = False
if 'ptm_configs' not in st.session_state:
    st.session_state.ptm_configs = {}
if 'apply_tryptic' not in st.session_state:
    st.session_state.apply_tryptic = False
if 'pdb_source' not in st.session_state:
    st.session_state.pdb_source = 'AlphaFold'  # Default to AlphaFold
if 'uploaded_pdb' not in st.session_state:
    st.session_state.uploaded_pdb = None

# ----------------------------------------------------------------------
# Main tabs (Qualitative vs Quantitative)
# ----------------------------------------------------------------------
main_tab = st.tabs(["Qualitative Analysis", "Quantitative Analysis"])

# Determine which main tab is active
if main_tab[0].is_selected:
    st.session_state.main_tab = "Qualitative Analysis"
elif main_tab[1].is_selected:
    st.session_state.main_tab = "Quantitative Analysis"
# ----------------------------------------------------------------------
# QUALITATIVE ANALYSIS
# ----------------------------------------------------------------------
with main_tab[0]:
    # ----- description (always visible) -----
    st.markdown(
        """
        <p style='text-align:center; font-size:16px; color:#87CEbB;'>
        The Qualitative Analysis tab allows users to visualize peptide intensity data on 3D protein structures, 
        focusing on the presence and locations of post-translational modifications (PTMs). 
        Users can upload peptide CSV and FASTA files, select proteins of interest, and customize PTM annotations. 
        The tool fetches AlphaFold structures or accepts user-uploaded PDB files, providing interactive 3D and 
        linear sequence views for detailed exploration of residue-level modifications.
        </p>
        """,
        unsafe_allow_html=True,
    )
    # ----- sub-tabs inside Qualitative -----
    st.session_state.qual_sub_tab = "Single Condition"
    st.subheader("Single Condition")
    st.write("Upload peptide and FASTA files for a single condition.")

    csv_file = st.file_uploader("Peptide CSV (Single Condition)", type="csv", key="qual_csv")
    fasta_file = st.file_uploader("FASTA File", type="fasta", key="qual_fasta")

    if csv_file and fasta_file:
        df = pd.read_csv(csv_file)
        fasta_str = fasta_file.getvalue().decode("utf-8")
        records = list(SeqIO.parse(io.StringIO(fasta_str), "fasta"))

        # Detect intensity column
        num_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in ['Protein.Group', 'Stripped.Sequence']]
        if not num_cols:
            st.error("No numeric intensity columns found in CSV.")
            st.stop()

        intensity_col = st.selectbox("Select Intensity Column (Condition Name)", num_cols, key="qual_intensity_col")

        proteins = sorted(df['Protein.Group'].dropna().unique())
        selected_protein = st.selectbox("Select Protein", proteins, key="qual_protein_select")

        col1, col2 = st.columns(2)
        with col1:
            combine_isoforms = st.selectbox("Combine Isoforms?", ["yes", "no"], key="qual_combine_isoforms")
        with col2:
            overlap_strategy = st.selectbox("Overlap Strategy", ["none", "merge", "highest", "last"], key="qual_overlap_strategy")

        st.session_state.ptm_enabled = st.checkbox("Enable PTM Annotation", value=st.session_state.get("ptm_enabled", False), key="qual_ptm_enable")
        st.session_state.apply_tryptic = st.checkbox("Apply Tryptic Rule (K/R cleavage)", value=st.session_state.get("apply_tryptic", True), key="qual_tryptic_rule")

        # Force Z-score style (beautiful!)
        st.session_state.is_frequency = False

        if st.button("Process Protein", key="process_protein_1", use_container_width=True):
            with st.spinner("Processing protein..."):
                base_id = selected_protein.split("-")[0]
                protein_seq = next((str(r.seq) for r in records if base_id in r.id), str(records[0].seq))
                seq_len = len(protein_seq)

                selected_df = df[df['Protein.Group'] == selected_protein]
                ptm_col = 'Modified.Sequence' if st.session_state.ptm_enabled and 'Modified.Sequence' in df.columns else None

                residue_vals, ptm_dict = map_peptides_to_residues(
                    selected_df, protein_seq, intensity_col,
                    overlap_strategy=overlap_strategy,
                    ptm_col=ptm_col,
                    apply_tryptic=st.session_state.apply_tryptic
                )

                # === Build PTM data ===
                final_ptm = {}
                if st.session_state.ptm_enabled and ptm_dict:
                    all_unimods = sorted(ptm_dict.keys(), key=lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else 999999)
                    if all_unimods:
                        st.markdown("### PTM Configuration")
                        for um in all_unimods:
                            positions = ptm_dict[um]
                            default_color = "#0024FF"
                            match=re.search(r"\d+", str(um))
                            label = f"UniMod:{match.group()}" if match else str(um)

                            colA, colB= st.columns([2, 1])
                            with colA:
                                st.write(f"**{label}** at positions: {', '.join(map(str, sorted(positions)[:10]))}{', ...' if len(positions)>10 else ''}")
                            with colB:
                                color = st.color_picker(f"Color##{um}", default_color, key=f"ptmcol_{um}")
                            #with colC:
                                #st.checkbox("Show", value=True, key=f"ptmshow_{um}")

                            final_ptm[um] = {
                                "positions": positions,
                                "color": color,
                                "label": label,
                                "selected": True
                            }
                    else:
                        st.info("No PTMs detected.")
                else:
                    final_ptm = None

                # === Load PDB ===
                if st.session_state.pdb_source == "AlphaFold":
                    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v6.pdb"
                    try:
                        pdb_str = requests.get(pdb_url, timeout=30).text
                    except:
                        st.error("Failed to download AlphaFold structure.")
                        st.stop()
                else:
                    if not st.session_state.uploaded_pdb:
                        st.error("Please upload a PDB file.")
                        st.stop()
                    pdb_str = st.session_state.uploaded_pdb.getvalue().decode()

                plddt_list, _, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)

                st.success(f"Loaded **{base_id}** • Sequence length: {seq_len} • Mean pLDDT: **{mean_plddt:.1f}**")

                # === External Links ===
                st.markdown("### External Resources")
                col1, col2, col3,col4 = st.columns(4)
                with col1:
                    st.markdown(f"[AlphaFold DB](https://alphafold.ebi.ac.uk/entry/{base_id})")
                with col2:
                    st.markdown(f"[UniProt](https://www.uniprot.org/uniprotkb/{base_id}/entry)")
                with col3:
                    st.markdown(f"[PeptideAtlas](https://db.systemsbiology.net/sbeams/cgi/PeptideAtlas/GetProtein?protein_name={base_id})")
                with col4:
                    st.markdown(f"[ESM Metagenomic Atlas](https://esmatlas.com/resources?action=fold&id={base_id})",
                                unsafe_allow_html=True
                    )
                # === 3D Viewer ===
                st.subheader("3D Structure Visualization")
                render_single_3d_viewer(
                    pdb_str,
                    residue_vals,
                    intensity_col,
                    "autumn",
                    "#A7A5A5",
                    final_ptm
                )

                # === Linear Plot — Beautiful Z-score Style ===
                st.subheader("Linear Sequence Visualization")
                render_linear_plot(
                    residue_vals=residue_vals,
                    title=intensity_col,                    # ← THIS IS THE CONDITION NAME!
                    seq_len=seq_len,
                    vmin=-3,
                    vmax=3,
                    protein_seq=protein_seq,
                    model_name=intensity_col,
                    plddt_list=plddt_list,
                    mean_plddt=mean_plddt,
                    cmap_name="autumn",                     # Best for intensity
                    not_mapped_color="#A7A5A5",
                    ptm_data=final_ptm
                )

                # Optional colorbar
                st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# QUANTITATIVE ANALYSIS
# ----------------------------------------------------------------------
with main_tab[1]:
    # ----- description (always visible) -----
    st.markdown(
        """
        <p style='text-align:center; font-size:16px; color:#87CEbB;'>
        The Quantitative Analysis tab enables users to compare peptide intensity data between two conditions 
        (e.g., control vs. disease) on 3D protein structures. Users can upload peptide CSV and FASTA files, 
        define experimental conditions, and apply tryptic cleavage rules. The tool visualizes intensity 
        differences using z-score scales, highlights PTMs, and provides interactive 3D and linear sequence 
        views for in-depth analysis of residue-level changes between conditions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ----- sub-tabs inside Quantitative -----
    quant_single_tab, quant_multi_tab,quant_diff_tab = st.tabs(["Single Condition", "Multiple Conditions","Differential Analysis"])

    # ----- SINGLE CONDITION (Quantitative) -----
    with quant_single_tab:
        st.session_state.active_quant_tab = "single"
        # Clean multi-tab leftovers
        st.session_state.processed_multi = {"processed": False}
        st.subheader("Single Condition Quantitative Analysis")
        st.markdown(
            """
            <p style='text-align:center; font-size:18px; color:#32CD32;'>
            Available Now! The Single Condition Quantitative Analysis allows you to analyze peptide intensities 
            and visualize them on 3D structures. Upload your data and explore the features.
            </p>
            """,
            unsafe_allow_html=True,
        )
        csv_file = st.file_uploader("Peptide CSV (Single Condition)", type="csv", key="quant_single_csv")
        fasta_file = st.file_uploader("FASTA File", type="fasta", key="quant_single_fasta")

        if csv_file and fasta_file:
            df = pd.read_csv(csv_file)
            fasta_str = fasta_file.getvalue().decode("utf-8")
            records = list(SeqIO.parse(io.StringIO(fasta_str), "fasta"))

            # Detect intensity column
            num_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in ['Protein.Group', 'Stripped.Sequence']]

            proteins = sorted(df['Protein.Group'].dropna().unique())
            selected_protein = st.selectbox("Select Protein", proteins,key="quant_protein_select")
            intensity_col = st.selectbox("Select Intensity Column (Condition Name)", num_cols, key="quant_intensity_col")
            # Detect intensity column
            if not num_cols:
                st.error("No numeric intensity columns found in CSV.")
                st.stop()
            col1, col2 = st.columns(2)
            with col1:
                combine_isoforms = st.selectbox("Combine Isoforms?", ["yes", "no"], key="quant_combine_isoforms")
            with col2:
                overlap_strategy = st.selectbox("Overlap Strategy", ["none","merge", "highest"], key="quant_overlap_strategy")

            st.session_state.ptm_enabled = st.checkbox("Enable PTM Annotation", value=st.session_state.get("ptm_enabled", False), key="quant_ptm_enable")
            st.session_state.apply_tryptic = st.checkbox("Apply Tryptic Rule (K/R cleavage)", value=st.session_state.get("apply_tryptic", True), key="quant_tryptic_rule")


            st.session_state.is_frequency = False

            if st.button("Process Protein", key="process_single_quant", use_container_width=True):
                with st.spinner("Processing protein..."):
                    base_id = selected_protein.split("-")[0]
                    protein_seq = next((str(r.seq) for r in records if base_id in r.id), str(records[0].seq))
                    seq_len = len(protein_seq)

                    selected_df = df[df['Protein.Group'] == selected_protein]
                    ptm_col = 'Modified.Sequence' if st.session_state.ptm_enabled and 'Modified.Sequence' in df.columns else None

                    residue_vals, ptm_dict = map_peptides_to_residues(
                        selected_df, protein_seq, intensity_col,
                        overlap_strategy=overlap_strategy,
                        ptm_col=ptm_col,
                        apply_tryptic=st.session_state.apply_tryptic
                    )
                    if residue_vals is None:
                        st.error("map_peptides_to_residues returned None → check input data / function logic")
                        st.stop()

                    # Make sure it's a list/array with numbers
                    if not hasattr(residue_vals, '__len__') or len(residue_vals) == 0:
                        st.warning("No residue values were mapped. Color will be uniform.")
                        residue_vals = [0.0] * len(protein_seq)   # or whatever fallback makes sense
                    # === Build PTM data ===
                    final_ptm = {}
                    if st.session_state.ptm_enabled and ptm_dict:
                        all_unimods = sorted(ptm_dict.keys(), key=lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else 999999)
                        if all_unimods:
                            st.markdown("### PTM Configuration")
                            for um in all_unimods:
                                positions = ptm_dict[um]
                                default_color = "#0024FF"
                                match=re.search(r"\d+", str(um))
                                label = f"UniMod:{match.group()}" if match else str(um)

                                colA, colB,colC = st.columns([2, 1, 1])
                                with colA:
                                    st.write(f"**{label}** at positions: {', '.join(map(str, sorted(positions)[:10]))}{', ...' if len(positions)>10 else ''}")
                                with colB:
                                    color = st.color_picker(f"Color##{um}", default_color, key=f"ptmcol_{um}")
                                with colC:
                                    st.checkbox("Show", value=True, key=f"ptmshow_{um}")

                                final_ptm[um] = {
                                    "positions": positions,
                                    "color": color,
                                    "label": label,
                                    "selected": True
                                }
                        else:
                            st.info("No PTMs detected.")
                    else:
                        final_ptm = None

                    # === Load PDB ===
                    if st.session_state.pdb_source == "AlphaFold":
                        pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v6.pdb"
                        try:
                            pdb_str = requests.get(pdb_url, timeout=30).text
                        except:
                            st.error("Failed to download AlphaFold structure.")
                            st.stop()
                    else:
                        if not st.session_state.uploaded_pdb:
                            st.error("Please upload a PDB file.")
                            st.stop()
                        pdb_str = st.session_state.uploaded_pdb.getvalue().decode()

                    plddt_list, _, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)

                    st.success(f"Loaded **{base_id}** • Sequence length: {seq_len} • Mean pLDDT: **{mean_plddt:.1f}**")

                    # === External Links ===
                    st.markdown("### External Resources")
                    col1, col2, col3,col4 = st.columns(4)
                    with col1:
                        st.markdown(f"[AlphaFold DB](https://alphafold.ebi.ac.uk/entry/{base_id})")
                    with col2:
                        st.markdown(f"[UniProt](https://www.uniprot.org/uniprotkb/{base_id}/entry)")
                    with col3:
                        st.markdown(f"[PeptideAtlas](https://db.systemsbiology.net/sbeams/cgi/PeptideAtlas/GetProtein?protein_name={base_id})")
                    with col4:
                        st.markdown(f"[ESM Metagenomic Atlas](https://esmatlas.com/resources?action=fold&id={base_id})",
                                unsafe_allow_html=True
                    )
                
                st.session_state.processed_single = {
                    "selected_df": selected_df,
                    "protein_seq": protein_seq,
                    "intensity_col": intensity_col,
                    "base_id": base_id,
                    "residue_vals": residue_vals,
                    "ptm_dict": final_ptm,
                    "pdb_str": pdb_str,
                    "plddt_list": plddt_list,
                    "mean_plddt": mean_plddt,
                    "processed": True
                }
                    # === 3D Viewer ===
                st.subheader("3D Structure Visualization")
                render_single_3d_viewer(
                        pdb_str,
                        residue_vals,
                        intensity_col,
                        "autumn",
                        "#A7A5A5",
                        final_ptm
                    )

                # === Linear Plot —  Z-score Style ===
                st.subheader("Linear Sequence Visualization")
                render_linear_plot(
                        residue_vals=residue_vals,
                        title=intensity_col,                    # ← THIS IS THE CONDITION NAME!
                        seq_len=seq_len,
                        vmin=-3,
                        vmax=3,
                        protein_seq=protein_seq,
                        model_name=intensity_col,
                        plddt_list=plddt_list,
                        mean_plddt=mean_plddt,
                        cmap_name="autumn",                   
                        not_mapped_color="#A7A5A5",
                        ptm_data=final_ptm
                )

                if (st.session_state.get("processed_single", {}).get("processed", False) and
                    st.session_state.active_quant_tab == "single"):

                    data = st.session_state.processed_single
                    base_id       = data.get("base_id")
                    intensity_col = data.get("intensity_col")
                    residue_vals  = data.get("residue_vals")

                    st.markdown(f"**Protein:** `{base_id}` | **Intensity column:** `{intensity_col}`")

                    if residue_vals is None or not hasattr(residue_vals, '__iter__'):
                        st.info("No processed intensity data yet. Using default color scale.")
                        vmin, vmax = -3.0, 3.0

                    else:
                        valid_vals = [
                            v for v in residue_vals
                            if v is not None and isinstance(v, (int, float))
                        ]

                        if valid_vals:
                            vmin = min(valid_vals)
                            vmax = max(valid_vals)
                        else:
                            st.warning("No valid (non-None) intensity values were mapped.")
                            vmin, vmax = -3.0, 3.0

                    # ─────────────────────────────────────────────
                    #   Z-score Colorbar
                    # ─────────────────────────────────────────────
                    st.subheader("Z-score Colorbar")

                    fig, ax = plt.subplots(figsize=(8, 0.35))
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    sm = ScalarMappable(cmap="autumn", norm=norm)
                    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
                    cbar.set_label('Z-Score Intensity', fontsize=10)
                    cbar.ax.tick_params(labelsize=9)

                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=220, transparent=False)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close(fig)

                    html_content = f"""
                    <div style="display: flex; justify-content: center; width: 100%; max-width: 620px; margin: 12px auto;">
                        <img src="data:image/png;base64,{img_str}" style="width: 100%; max-width: 520px; height: auto; border: 1px solid #e0e0e0; border-radius: 6px;">
                    </div>
                    """
                    st.components.v1.html(html_content, height=90)

                    
                    # Optional colorbar
                #st.markdown("<br>", unsafe_allow_html=True)
                #st.subheader("Z-score Colorbar")
                #render_zscore_colorbar(residue_vals, "autumn", "#A5A5A5")
                #qunatitative single condition visualization panel
            #if (st.session_state.processed_single.get("processed", False) and
                #st.session_state.active_quant_tab == "single"):   # ← lowercase!
                    #data = st.session_state.processed_single
                    #selected_df   = data.get("selected_df")
                    #protein_seq   = data.get("protein_seq")
                    ##intensity_col = data.get("intensity_col")
                    # base_id       = data.get("base_id")   
                    #st.markdown(f"**Protein:** `{base_id}` | **Intensity column:** `{intensity_col}`")

                    # Choose peptide column
                    #available_cols = [c for c in ["Stripped.Sequence", "Modified.Sequence"] if c in selected_df.columns]
                    #if not available_cols:
                     #   st.error("No peptide sequence column found.")
                     #   st.stop()

                    #peptide_column = st.selectbox(
                           # "Peptide sequence column",
                           # available_cols,
                           # index=0,
                           # key="vis_peptide_col"
                        #)
                    #unique_peptides = selected_df[peptide_column].dropna().unique()
                    # Build list of peptides with positions
                    #peptides_with_pos = []
                    #for pep in unique_peptides:
                        #clean_pep = re.sub(r"\(UniMod:\d+\)", "", str(pep))
                        #pos = find_peptide_position(protein_seq, clean_pep)
                        #if pos is not None:
                        #    peptides_with_pos.append((pep, pos))
                    # Sort by position
                    #peptides_with_pos.sort(key=lambda x: x[1] if x[1] else 999999)

                    #if not peptides_with_pos:
                    #    st.warning("No mappable peptides found.")
                    #    st.stop()
                
                    # === TWO-COLUMN LAYOUT ===
                    #col_left, col_right = st.columns([1.2, 1])  # Left slightly wider

                    #with col_left:
                        #st.markdown("**Select Peptide**")
                        #options = [f"{pep} (pos: {pos})" for pep, pos in peptides_with_pos]
                        #selected_option = st.selectbox(
                         #   "Peptide sequence and position",
                         #   options,
                          #  key="vis_selected_peptide",
                          #  label_visibility="collapsed"
                        #)
                        # Extract selected peptide
                        #selected_peptide = selected_option.split(" (pos:")[0]

                    #with col_right:
                        #st.markdown("**Summary Statistics**")

                        # Get intensities for selected peptide
                        #subset = selected_df[selected_df[peptide_column] == selected_peptide]
                        #intensities = pd.to_numeric(subset[intensity_col], errors="coerce").dropna()

                        #if intensities.empty:
                            #st.info("No intensity values for this peptide.")
                        #else:
                            #stats = intensities.describe()
                            #stats_df = stats.to_frame().T
                            #stats_df = stats_df.round(4)

                            # Beautiful styled table
                            #st.dataframe(
                                #stats_df.style.format({
                                 #   "count": "{:.0f}",
                                  #  "mean": "{:.4f}",
                               # }).background_gradient(cmap="viridis", low=0, high=1),
                               # use_container_width=True
                #)
    # ----- MULTIPLE CONDITIONS (Quantitative) – ONLY HERE we show uploads -----
with quant_multi_tab:
        st.session_state.active_quant_tab = "multi"
        # OPTIONAL: Clean any leftover single data
        st.session_state.processed_single = {"processed": False}
        st.subheader("Multiple Conditions Quantitative Analysis")
        st.markdown(
            """
            <p style='text-align:center; font-size:18px; color:#32CD32;'>
            Available Now! The Multiple Conditions Quantitative Analysis is ready with exciting features 
            to compare peptide intensity data on 3D protein structures.
            </p>
            """,
            unsafe_allow_html=True,
        )
        
        # File upload
        csv_file_q = st.file_uploader("Upload Peptide Intensity CSV (e.g., Input_Two_Condition.csv)", type=["csv"],key="quant_multi_csv")
        fasta_file_q = st.file_uploader("Upload FASTA", type=["fasta"],key="quant_multi_fasta")
        metadata_file = st.file_uploader("Upload Metadata CSV (File_Name, Group, Sample_Name)", type=["csv"],key="quant_multi_metadata")

        if csv_file_q and fasta_file_q and metadata_file:
            # Read files
            try:
                df = pd.read_csv(csv_file_q)
            except Exception as e:
                st.error(f"Error reading Peptide CSV: {e}")
                st.stop()

            try:
                metadata = pd.read_csv(metadata_file)
            except Exception as e:
                st.error(f"Error reading Metadata CSV: {e}")
                st.stop()

            required_meta = {"File_Name", "Group", "Sample_Name"}
            if not required_meta.issubset(set(metadata.columns)):
                st.error(f"Metadata CSV must contain columns: {required_meta}")
                st.stop()

            fasta_str = fasta_file_q.getvalue().decode("utf-8")
            fasta_handle = io.StringIO(fasta_str)
            seq_records = list(SeqIO.parse(fasta_handle, "fasta"))
            if not seq_records:
                st.error("No sequences found in FASTA file.")
                st.stop()
            def extract_uniprot_id(header: str) -> str:
                # Try to use UniProt format sp|ID|... else fallback first token
                parts = header.split('|')
                if len(parts) >= 2 and parts[0] in ('sp', 'tr'):
                    return parts[1]
                return header.split()[0]

            fasta_ids = [extract_uniprot_id(rec.id) for rec in seq_records]
            fasta_with_isoform = len(fasta_ids)
            fasta_without_isoform = len(set([fid.split('-')[0] for fid in fasta_ids]))

            # Metadata: samples & groups
            num_samples = metadata.shape[0]
            num_groups = metadata['Group'].nunique()
    
            # Peptide CSV: proteins with/without isoform
            if 'Protein.Group' not in df.columns:
                st.error("Peptide CSV must contain 'Protein.Group' column.")
                st.stop()
            prot_ids = df['Protein.Group'].dropna().astype(str).unique().tolist()
            prot_with_isoform = len(prot_ids)
            prot_without_isoform = len(set([p.split('-')[0] for p in prot_ids]))

            # Peptide CSV: stripped sequences
            if 'Stripped.Sequence' not in df.columns:
                st.error("Peptide CSV must contain 'Stripped.Sequence' column.")
                st.stop()
            num_peptides_raw = df['Stripped.Sequence'].notna().sum()
            num_peptides_unique = df['Stripped.Sequence'].dropna().nunique()
        
            # PTM detection (supports either 'PTM' or 'Modified.Sequence')
            ptm_col_candidates = [c for c in [ 'Modified.Sequence','PTM'] if c in df.columns]
            ptm_detected = False
            unimods = ["NA"]
            num_modseq_raw = "NA"
            num_modseq_unique = "NA"
            if ptm_col_candidates:
                ptm_col_for_detection = ptm_col_candidates[0]
                col_series = df[ptm_col_for_detection].dropna().astype(str)
                ptm_detected = col_series.str.contains('UniMod:', case=False).any()
                if ptm_detected:
                    all_text = ' '.join(col_series.tolist())
                    unimods = sorted(set(re.findall(r'UniMod:\d+', all_text)))
                    num_modseq_raw = df[ptm_col_for_detection].notna().sum()
                    num_modseq_unique = df[ptm_col_for_detection].dropna().nunique()

            st.markdown("### 🧾 File Information Summary")
            st.info(
                f"**FASTA File**\n"
                f"- Total Proteins (without isoform): {fasta_without_isoform}\n"
                f"- Total Proteins (with isoform): {fasta_with_isoform}\n\n"
                f"**Metadata File**\n"
                f"- Samples Detected: {num_samples}\n"
                f"- Groups Detected: {num_groups}\n\n"
                f"**Peptide CSV File**\n"
                f"- Proteins Detected (without isoform): {prot_without_isoform}\n"
                f"- Proteins Detected (with isoform): {prot_with_isoform}\n"
                f"- Stripped Sequences: {num_peptides_raw} (Unique: {num_peptides_unique})\n"
                f"- PTM Detected: {'Yes' if ptm_detected else 'No'}\n"
                f"- PTM Types: {', '.join(unimods)}\n"
                f"- Modified Sequences: {num_modseq_raw} (Unique: {num_modseq_unique})"
            )
            # --------------------------------------
            # Identify sample intensity columns using Metadata["File_Name"]
            # --------------------------------------
            sample_candidates = metadata['File_Name'].astype(str).tolist()
            df_cols = set(map(str, df.columns))
            sample_cols = [c for c in sample_candidates if c in df_cols]
            if len(sample_cols) == 0:
                st.error("No sample columns found in the Peptide CSV that match Metadata['File_Name']. Check column names.")
                st.stop()

            # Map each sample to its Group
            meta_map = metadata.set_index('File_Name')['Group'].to_dict()
            groups = metadata['Group'].unique().tolist()

            # Aggregation choice
            st.markdown("### Aggregation across replicates")
            agg_choice = st.radio( "Aggregate replicates by:",["Mean", "Median","Frequency"],horizontal=True,index=0,key="agg_choice_radio")
            #agg_method = st.radio("Method:", ["Mean", "Median"], horizontal=True, index=0,key="agg_method_radio")
            # store for later use
            st.session_state.is_frequency = (agg_choice == "Frequency")
            # Create group-intensity columns per row
            group_to_samples = {
                g: [s for s in sample_cols if meta_map.get(s, None) == g]
                for g in groups
            }
            empty_groups = [g for g, cols in group_to_samples.items() if len(cols) == 0]
            if empty_groups:
                st.warning(f"Groups with no matching sample columns in CSV: {empty_groups}")

            # Aggregation function
            def aggregate_group(df_group, samples, choice):
                """
                df_group – ONE ROW of the main DataFrame (as a DataFrame)
                samples  – list of column names belonging to the group
                choice   – "Mean" | "Median" | "Frequency"
                Returns a **single float** (or 0/1 for frequency)
                """
                # 1. Extract the values for the current row & the selected samples
                row_vals = df_group[samples].iloc[0]               # Series of length = #samples
                # Frequency → 0/1 detection per replicate → then aggregate
                if choice == "Frequency":
                    # any non-zero / non-NA → detected
                    detected = row_vals.gt(0) | row_vals.notna()   # True/False per replicate
                    if choice == "Mean":
                        return detected.mean()                    # 0.0 – 1.0
                    else:  # Median
                        return detected.median()                  # 0.0 or 1.0
                # Mean / Median of intensities
                # clean infinite / NaN
                clean = row_vals.replace([np.inf, -np.inf], np.nan).dropna()
                if clean.empty:
                    return np.nan

                if choice == "Mean":
                    return clean.mean()
                else:  # Median
                    return clean.median()
            group_to_samples = {
                g: [s for s in sample_cols if meta_map.get(s, None) == g]
                for g in groups
            }
            for g, cols in group_to_samples.items():
                if not cols:                                 # safety
                    df[g] = np.nan
                    continue
                df[g] = df.apply(
                    lambda row: aggregate_group(row.to_frame().T, cols, agg_choice),
                    axis=1
                )
            if ptm_detected:
                st.session_state.all_unimods = unimods
            else:
                st.session_state.all_unimods = []
                st.session_state.selected_unimods = []
            # PTM and tryptic options
            has_ptm = bool(st.session_state.all_unimods)
            ptm_checkbox_disabled = not has_ptm
            st.session_state.ptm_enabled = st.checkbox("Enable PTM Annotation", disabled=ptm_checkbox_disabled, value=False if ptm_checkbox_disabled else st.session_state.ptm_enabled,key="quant_multi_ptm_enable")
            st.session_state.apply_tryptic = st.checkbox("Apply Tryptic Rule (K/R cleavage)", value=st.session_state.apply_tryptic)
            st.session_state.proteotypic_only = st.checkbox(
                    "Proteotypic Peptides Only (correct start site + fully tryptic)",
                    value=st.session_state.get("proteotypic_only", True),
                    help="Only use peptides that are correctly cleaved (preceded by K/R or N-terminal) AND have no missed cleavages (fully tryptic, respects K-P/R-P rule)",
                    key="proteotypic_only_checkbox"
                )
          
            # PTM selection
            if st.session_state.all_unimods:
                st.markdown("### Detected PTM UniMod IDs")
                st.write("The following UniMod IDs were detected in the PTM column. Select the ones to include:")
                if 'selected_unimods' not in st.session_state:
                    st.session_state.selected_unimods = st.session_state.all_unimods.copy()
                selected_unimods = st.multiselect(
                    "Select UniMod IDs to Include",
                    options=st.session_state.all_unimods,
                    default=st.session_state.selected_unimods,
                    help="Choose the UniMod IDs you want to process for PTM annotation."
                )
                st.session_state.selected_unimods = selected_unimods
          
                if st.session_state.ptm_enabled and st.session_state.all_unimods and not st.session_state.selected_unimods:
                    st.session_state.selected_unimods = st.session_state.all_unimods.copy()
            else:
                st.session_state.selected_unimods = []
              
            intensity_cols = [g for g in groups if g in df.columns]
            if len(intensity_cols) < 2:
                st.error(f"At least two groups are required after aggregation. Found: {intensity_cols}")
                st.stop()

            # Select conditions (groups) to compare
            default_conditions = intensity_cols[:2] if len(intensity_cols) >= 2 else intensity_cols
            selected_conditions = st.multiselect(
                "Select Conditions to Compare (2-4)",
                options=intensity_cols,
                default=default_conditions,
                help="Select 2 to 4 groups/conditions for comparison."
            )

            if len(selected_conditions) < 2 or len(selected_conditions) > 4:
                st.error("Please select between 2 and 4 conditions.")
                st.stop()
            
            if st.button("Confirm Conditions", use_container_width=True, key="confirm_conditions"):
                st.session_state.conditions_confirmed = True
                st.session_state.processed = False
                st.rerun()

            if st.session_state.conditions_confirmed:
                with st.container():
                    st.info("✅ Conditions confirmed. Now select protein and options.")
                    protein_options = sorted(df['Protein.Group'].unique())
                    selected_protein = st.selectbox("Select Protein", protein_options)
                    col3,col4 = st.columns([1,1])
                    with col3:
                        combine_isoforms = st.selectbox("Combine Isoforms?", ["yes", "no"])
                    with col4:
                        overlap_strategy = st.selectbox("Overlap Strategy", ["none","merge", "highest"])
                    
                    # PDB source selection with hyperlinks
                    st.markdown(
                        f'<div style="text-align:left; margin-bottom:10px;">'
                        f'<span style="font-size:16px; color:#FFFFFF;">Databases: </span>'
                        f'<a href="https://alphafold.ebi.ac.uk/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">AlphaFold Database</a>'
                        f' | '
                        f'<a href="https://esmatlas.com/resources?action=fold" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">ESM Atlas</a>'
                        f' | '
                        f'<a href="https://build.nvidia.com/mit/boltz2" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">NVIDIA Boltz-2</a>'
                        f' | '
                        f'<a href="https://design-a-protein.com/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">SeqHub</a>'
                        f' | '
                        f'<a href="https://www.rcsb.org/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">RCSB PDB</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.pdb_source = st.selectbox(
                        "Select PDB Source",
                        ["AlphaFold", "Upload PDB"],
                        help="Choose to fetch the structure from AlphaFold or upload a PDB file named as UniProtID.pdb"
                    )
                    if st.session_state.pdb_source == "Upload PDB":
                        st.session_state.uploaded_pdb = st.file_uploader(
                            "Upload PDB File",
                            type=["pdb"],
                            help="Upload a PDB file named as {UniProt_ID}.pdb matching the selected protein"
                        )
                    
                    if st.button("Process Protein", use_container_width=True, key="process_protein_3"):
                        st.session_state.processed = True
                        st.rerun()

                if st.session_state.processed:
                    with st.container():
                        st.info("🔄 Processing... (This may take a moment for PDB fetch or upload.)")
                        base_id = selected_protein.split('-')[0]
                        protein_seq = None
                        matched_header="Not Found"
                        for rec in seq_records:
                           header = rec.id
                        
                           candidate = header.split('|')[1] if '|' in header and header.split('|')[0] in ('sp', 'tr') else header.split()[0]
                           if candidate.split('-')[0] == base_id:
                                protein_seq = str(rec.seq)
                                matched_header = header
                                st.success(f"Matched by ID: {header}")
                                break
                        if protein_seq is None:
                            st.info(f"No direct FASTA header match for {base_id}. Attempting peptide-based matching...")
                            peptides_unique = selected_df['Stripped.Sequence'].dropna().astype(str).str.strip().str.upper().unique().tolist()
                            peptides_unique = [p for p in peptides_unique if len(p) >= 7]

                            best_rec = None
                            best_count = 0

                            for rec in seq_records:
                                seq_str = str(rec.seq).upper()
                                count = sum(1 for p in peptides_unique if p in seq_str)
                                if count > best_count:
                                    best_count = count
                                    best_rec = rec

                            if best_count >= 2:  # at least 2 peptides = reliable
                                protein_seq = str(best_rec.seq)
                                matched_header = best_rec.id
                                st.success(f"Best match: {matched_header} ({best_count} peptides matched)")
                            elif len(seq_records) == 1:
                                protein_seq = str(seq_records[0].seq)
                                matched_header = seq_records[0].id
                                st.warning(f"Only one FASTA entry → using: {matched_header}")
                            else:
                                st.error(f"Could not match protein sequence. Check FASTA headers or peptide data.")
                                st.stop()

                        seq_len = len(protein_seq)
                        # CRITICAL: SAVE TO SESSION STATE SO SUMMARY CAN SEE IT!
                        st.session_state.protein_seq = protein_seq
                        st.session_state.selected_protein = selected_protein
                        st.session_state.matched_fasta_header = matched_header
                        # Isoform handling
                        isoforms = df[df['Protein.Group'].str.contains(selected_protein + r'(?:-\d+)?$', regex=True)]['Protein.Group'].unique()
                        if len(isoforms) > 1 and combine_isoforms == "yes":
                            st.info("Isoforms Detected")
                            selected_groups = list(isoforms)
                        elif len(isoforms) > 1 and combine_isoforms == "no":
                            selected_groups = st.multiselect("Select Isoforms", options=list(isoforms), default=list(isoforms))
                        else:
                            selected_groups = list(isoforms)
                        
                        if not selected_groups:
                            st.error("No isoforms selected.")
                            st.stop()
                        
                        selected_df = df[df['Protein.Group'].isin(selected_groups)]
                        conditions = selected_conditions
                        peptide_data = {}
                        residue_data = {cond: [None] * seq_len for cond in conditions}
                        ptm_data = {cond: {} for cond in conditions}
                        min_max_logs = {}
                        ptm_col = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None

                        st.session_state.selected_df = selected_df

                        for condition in conditions:
                            intensity_col = condition
                            # ← PASS ptm_col only if enabled!
                            ptm_col_to_use = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None
                            residues, ptms = map_peptides_to_residues(
                                selected_df, protein_seq, intensity_col, overlap_strategy,
                                ptm_col=ptm_col_to_use, apply_tryptic=st.session_state.apply_tryptic
                            )
                            residue_data[condition] = residues
                            # Build full PTM dict with config
                            ptm_data[condition] = {}
                            if st.session_state.ptm_enabled:
                                for um, positions in ptms.items():
                                    if um in st.session_state.selected_unimods:
                                        config = st.session_state.ptm_configs.get(um, {})
                                        ptm_data[condition][um] = {
                                            'positions': sorted(list(positions)),
                                            'selected': config.get('selected', True),
                                            'color': st.session_state.ptm_configs.get(um,{}).get('color', '#3700FF'),
                                            'label': config.get('label', um)
                                        }
                           
                            covered = [v for v in residues if v is not None]
                            if not covered:
                                st.error(f"No peptides mapped for {condition}.")
                                st.stop()
                            min_max_logs[condition] = (min(covered), max(covered))
                            peptides = selected_df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
                            peptide_data[condition] = peptides
                       
                        # PTM configuration with hyperlinks
                        if st.session_state.ptm_enabled and st.session_state.selected_unimods:
                            st.subheader("PTM Configuration")
                            st.write(f"Selected UniMods: {st.session_state.selected_unimods}")
                            # After mapping peptides → residues and PTMs
                            for condition in conditions:
                                detected_ptms = ptms  # from map_peptides_to_residues()

                                # Initialize full dict with ALL selected UniMods (even if empty positions)
                                ptm_data[condition] = {}
                                for um in st.session_state.selected_unimods:
                                    positions = detected_ptms.get(um, set())  # Use detected, or empty set
                                    config = st.session_state.ptm_configs.get(um, {
                                        'selected': True,
                                        'label': um,
                                        'color': '#3700FF'
                                    })

                                    ptm_data[condition][um] = {
                                        'positions': sorted(list(positions)),
                                        'selected': config['selected'],
                                        'label': config['label'],
                                        'color': config['color']
                                    }

                                # Optional: clean up unselected ones at the end
                                ptm_data[condition] = {
                                    um: info for um, info in ptm_data[condition].items()
                                    if info['selected']
                                }
                                for condition in ptm_data:
                                        ptm_data[condition] = normalize_ptm_data(ptm_data[condition])
                            
                        displayed_ptms = [um for cond in ptm_data.values() for um in cond.keys() if cond[um]['selected']]
                        if not displayed_ptms:
                            st.warning("No PTMs will be displayed: either none selected or none detected in this protein.")
                        else:
                            st.success(f"Displaying {len(displayed_ptms)} PTM type(s) where detected.")
                            # Ensure ptm_configs exist for selected UniMods and render PTM configuration UI
                            if ('ptm_configs' not in st.session_state or set(st.session_state.ptm_configs.keys()) != set(st.session_state.selected_unimods)):
                                st.session_state.ptm_configs = {um: {'selected': True, 'label': f"{um}", 'color': "#3700FF"} for um in st.session_state.selected_unimods}
                            # Debug to confirm selected UniMods
                            #st.write(f"Rendering PTM config for UniMods: {st.session_state.selected_unimods}")
                            for um in st.session_state.selected_unimods:
                                col_ptm1, col_ptm2, col_ptm3 = st.columns([1, 1, 1])
                                with col_ptm1:
                                    # UniMod website expects a numeric accession (e.g. 21) not the 'UniMod:21' prefix.
                                    m = re.search(r"(\d+)", str(um))
                                    unimod_id = m.group(1) if m else str(um)
                                    direct_url = f"https://www.unimod.org/modifications_view.php?editid1={unimod_id}"
                                    st.markdown(
                                        f'<a href="{direct_url}" target="_blank" '
                                        f'style="color:#2b8cff; text-decoration:underline; font-weight:bold;" '
                                        f'title="Click to view UniMod record {unimod_id} (phosphorylation, etc.)">'
                                        f'UniMod:{unimod_id}</a>',
                                        unsafe_allow_html=True
                                    )
                                    st.session_state.ptm_configs[um]['selected'] = st.checkbox(
                                        f"Include UniMod:{um}", value=st.session_state.ptm_configs[um]['selected'], key=f"checkbox_{um}"
                                    )
                                with col_ptm2:
                                    st.session_state.ptm_configs[um]['label'] = st.text_input(
                                        f"Label for UniMod:{um}", value=st.session_state.ptm_configs[um]['label'], key=f"label_{um}"
                                    )
                                with col_ptm3:
                                    st.session_state.ptm_configs[um]['color'] = st.color_picker(
                                        f"Color for UniMod:{um}", value=st.session_state.ptm_configs[um]['color'], key=f"color_{um}"
                                    )  
                        st.subheader("Detected Sequence")
                        st.markdown(f"**FASTA header:** {matched_header}")
                        seq_html = format_sequence_for_display(protein_seq, residue_data, conditions, line_len=150, group=20)
                        copy_html = sequence_copy_component(protein_seq)
                        st.components.v1.html(copy_html + seq_html, height=320)
                        
                        # PDB fetching or upload
                        if st.session_state.pdb_source == "AlphaFold":
                            pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v6.pdb"
                            with st.spinner(f"Attempting to fetch AlphaFold v6 structure for {base_id}..."):
                                try:
                                    r = requests.get(pdb_url, timeout=30)
                                    if r.status_code == 200:
                                        pdb_str = r.text
                                    elif r.status_code == 404:
                                        st.error("Model_v6 not found. The protein may not have a v6 structure yet.")
                                        st.stop()
                                    else:
                                        st.error(f"PDB fetch failed (status {r.status_code}).")
                                        st.stop()
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Failed to fetch PDB for {base_id}: {str(e)}.")
                                    st.stop()
                        else:  # Upload PDB
                            if st.session_state.uploaded_pdb is None:
                                st.error("No PDB file uploaded.")
                                st.stop()
                            # Validate filename
                            pdb_filename = st.session_state.uploaded_pdb.name
                            if not pdb_filename.endswith('.pdb'):
                                st.error("Uploaded file must have a .pdb extension.")
                                st.stop()
                            filename_id = pdb_filename[:-4]  # Remove .pdb extension
                            if filename_id != base_id:
                                st.error(f"PDB filename ({pdb_filename}) must match the selected protein's UniProt ID ({base_id}).")
                                st.stop()
                            try:
                                pdb_str = st.session_state.uploaded_pdb.getvalue().decode("utf-8")
                            except Exception as e:
                                st.error(f"Error reading uploaded PDB file: {e}")
                                st.stop()
                        
                        st.success(f"Loaded {'AlphaFold' if st.session_state.pdb_source == 'AlphaFold' else 'uploaded'} structure for {base_id} ({len(pdb_str)} bytes)")
                        plddt_list, model_name, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)
                        mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
                        st.info(f"**Mean pLDDT:** {mean_plddt_display} (Overall Confidence)")
                        bg_color = st.selectbox("Background Color", ["black", "white", "darkgrey"], index=0)
                        cmap_options = ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                        selected_cmap = st.selectbox("Select Color Gradient", cmap_options, index=0)
                        selected_not_mapped_color = st.color_picker("Select Not Mapped Color", "#d3d3d3")

                        st.subheader("PhosphoSitePlus®")
                        try:
                            uniprot_url = f"https://rest.uniprot.org/uniprotkb/{base_id}"
                            response = requests.get(uniprot_url, timeout=10)
                            if response.status_code == 200:
                                uniprot_data = response.json()
                                gene_name = uniprot_data.get('genes', [{}])[0].get('geneName', {}).get('value', 'Unknown')
                            else:
                                gene_name = 'Unknown'
                        except Exception as e:
                            gene_name = 'Unknown'
                            st.warning(f"Failed to fetch gene name for {base_id}: {e}")
                        if gene_name != 'Unknown':
                            phosphosite_url = f"https://www.phosphosite.org/simpleSearchSubmitAction.action?searchStr={gene_name}"
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore PhosphoSitePlus® : </span>'
                                f'<a href="{phosphosite_url}" target="_blank" style="font-size:20px; color:#87CEEB; text-decoration:underline;">{base_id}|{gene_name}</a>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore PhosphoSitePlus® : </span>'
                                f'<span style="font-size:20px; color:#FFFFFF;">No gene name found for {base_id}. Unable to generate PhosphoSitePlus link.</span>',
                                unsafe_allow_html=True
                            )
                        
                        with st.container():
                            st.subheader("3D Structure Visualizations")

                            # === Extract peptides and compute positions ===

                            # Compute peptide start positions directly from protein sequence
                            def compute_positions_with_label(peptide_list):
                                result = {}
                                for pep in peptide_list:
                                    pep = str(pep).strip()
                                    if pep == "":
                                        continue
                                    pos = find_peptide_position(protein_seq, pep)
                                    label = f"{pep} (pos: {pos})"
                                    result[pep] = (pos,label)
                                return result
                            # ---- Unique peptides for dropdowns (FIXED & ROBUST) ----
                            stripped_unique = selected_df["Stripped.Sequence"].dropna().astype(str).unique().tolist()
                            # === MODIFIED SEQUENCE: ONLY THOSE WITH ACTUAL PTMs ===
                            if (st.session_state.ptm_enabled and 
                                st.session_state.selected_unimods and 
                                "Modified.Sequence" in selected_df.columns):

                                # Build regex pattern for selected UniMods only
                                pattern = '|'.join([re.escape(f"UniMod:{um.split(':')[-1]}") for um in st.session_state.selected_unimods])
                                
                                mod_mask = selected_df["Modified.Sequence"].astype(str).str.contains(pattern, case=False, na=False)
                                filtered_modified_df = selected_df.loc[mod_mask]

                                if filtered_modified_df.empty:
                                    st.warning("No peptides found with the selected UniMod(s) — falling back to stripped sequences for dropdown.")
                                    modified_unique = stripped_unique
                                else:
                                    modified_unique = filtered_modified_df["Modified.Sequence"].dropna().astype(str).unique().tolist()
                            else:
                                # PTM disabled or no UniMod selected → fallback
                                modified_unique = stripped_unique

                            # Now compute positions exactly like you do in the quant tab
                            def compute_positions_with_label(peptide_list):
                                result = {}
                                for pep in peptide_list:
                                    pep_str = str(pep).strip()
                                    if not pep_str:
                                        continue
                                    pos = find_peptide_position(protein_seq, pep_str)
                                    label = f"{pep_str} (pos: {pos})"
                                    result[pep_str] = (pos, label)
                                return result

                            stripped_positions = compute_positions_with_label(stripped_unique)
                            modified_positions  = compute_positions_with_label(modified_unique)  # ← now only real modified peptides!

                            # Sort by position (same as before)
                            def sort_labels_by_position(pos_dict):
                                sorted_items = sorted(pos_dict.items(), key=lambda x: x[1][0] if x[1][0] != -1 else 999999)
                                return [label for _, (_, label) in sorted_items]

                            stripped_sorted_labels  = sort_labels_by_position(stripped_positions)
                            modified_sorted_labels  = sort_labels_by_position(modified_positions)  # ← clean & correct!
                          
                            # === Session state initialization ===
                            if "view_mode" not in st.session_state:
                                st.session_state.view_mode = "Full Structure (All Peptides)"
                            if "selected_peptide" not in st.session_state:
                                st.session_state.selected_peptide = None

                            st.markdown("#### Peptide View Mode")
                            view_mode = st.radio(
                                "Choose visualization mode:",
                                options=[
                                    "Full Structure (All Peptides)",
                                    "View by Stripped Sequence",
                                    "View by Modified Sequence"
                                ],
                                index=["Full Structure (All Peptides)", "View by Stripped Sequence", "View by Modified Sequence"]
                                    .index(st.session_state.view_mode),
                                key="view_mode_radio",
                                horizontal=True
                            )

                            if view_mode != st.session_state.view_mode:
                                st.session_state.view_mode = view_mode
                                st.session_state.selected_peptide = None
                                st.rerun()

                            selected_peptide = None
                            show_full = (st.session_state.view_mode == "Full Structure (All Peptides)")

                            # === Peptide selector ===
                            if st.session_state.view_mode == "View by Stripped Sequence":
                                peptide_options = stripped_sorted_labels
                                selected_label = st.selectbox(
                                    "Select a stripped peptide to highlight:",
                                    options=peptide_options,
                                    index=0 if st.session_state.selected_peptide not in stripped_positions else peptide_options.index(next(label for pep,(pos,label) in stripped_positions.items()if pep==st.session_state.selected_peptide)),
                                    key="stripped_selector"
                                )
                                selected_peptide = re.split(r"\s*\(pos:", selected_label)[0].strip()
                                st.session_state.selected_peptide = selected_peptide
                                st.info(f"Showing peptide: **{selected_label}**")
                                
                            elif st.session_state.view_mode == "View by Modified Sequence":
                                peptide_options = modified_sorted_labels
                                selected_label = st.selectbox(
                                    "Select a modified peptide to highlight:",
                                    options=peptide_options,
                                    index=0 if st.session_state.selected_peptide not in modified_positions else
                                        peptide_options.index(next(label for pep, (pos, label) in modified_positions.items() if pep == st.session_state.selected_peptide)),
                                    key="modified_selector"
                                )
                                selected_peptide = re.split(r"\s*\(pos:", selected_label)[0].strip()
                                st.session_state.selected_peptide = selected_peptide
                                st.info(f"Showing modified peptide: **{selected_label}**")

                            else:
                                st.success("Showing all mapped peptides on the protein structure")

                            # === BUILD RESIDUE DATA FOR VISUALIZATION ===
                            if show_full:
                                viewer_residue_data = [residue_data[cond] for cond in conditions]
                                viewer_ptm_data = [ptm_data[cond] for cond in conditions]  # Use full PTM data for all viewers
                            else:
                                target_peptide = selected_peptide
                                use_modified = (st.session_state.view_mode == "View by Modified Sequence")
                                filter_col = 'Modified.Sequence' if use_modified else 'Stripped.Sequence'

                                peptide_rows = selected_df[selected_df[filter_col] == target_peptide]

                                if peptide_rows.empty:
                                    st.error(f"Selected peptide not found in current data.")
                                    st.stop()

                                viewer_residue_data = []
                                viewer_ptm_data = []  # Clear & build fresh

                                for condition in conditions:
                                    single_residue_list = [None] * len(protein_seq)
                                    intensity_col = condition
                                    ptm_col_to_use = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None

                                    residues_temp, ptms_temp = map_peptides_to_residues(
                                        peptide_rows,
                                        protein_seq,
                                        intensity_col,
                                        overlap_strategy,
                                        ptm_col=ptm_col_to_use,
                                        apply_tryptic=st.session_state.apply_tryptic
                                    )

                                    # Copy residues (only if intensity non-NaN for this condition)
                                    if not peptide_rows[intensity_col].isna().all():
                                        for i, val in enumerate(residues_temp):
                                            if val is not None:
                                                single_residue_list[i] = val
                                    else:
                                        # For missing conditions: use gray/low-intensity fallback if needed, but keep PTMs
                                        pass

                                    viewer_residue_data.append(single_residue_list)

                                    # Handle PTMs: build condition_ptm even if intensity missing (PTMs are peptide-intrinsic)
                                    condition_ptm = {}
                                    if st.session_state.ptm_enabled and ptms_temp:
                                        for um, positions in ptms_temp.items():
                                            if um in st.session_state.selected_unimods:
                                                config = st.session_state.ptm_configs.get(um, {})
                                                condition_ptm[um] = {
                                                    'positions': sorted(list(positions)),
                                                    'selected': config.get('selected', True),
                                                    'color': config.get('color', '#3700FF'),
                                                    'label': config.get('label', um)
                                                }

                                    viewer_ptm_data.append(condition_ptm)  # ONLY append once per condition!
                            if not show_full:
                                st.info(f"**Single peptide view mode active**: Only the selected peptide '{selected_peptide}' is colored/highlighted in 3D and linear plots. The rest of the protein remains unmapped (grey).")
                            peptide_atlas_url = f"https://db.systemsbiology.net/sbeams/cgi/PeptideAtlas/GetProtein?atlas_build_id=592&protein_name={base_id}&action=QUERY"
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore PeptideAtlas : </span>'
                                f'<a href="{peptide_atlas_url}" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">Explore Peptides of {base_id} in Peptide Atlas</a>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            # Above "3D Structure Visualizations" header
                            alphafold_url = f"https://alphafold.ebi.ac.uk/search/text/{base_id}"
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore AlphaFold DB : </span>'
                                f'<a href="{alphafold_url}" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">View {base_id} in AlphaFold Database</a>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            # ---------- DisProt ----------
                            disprot_info = get_disprot_info(base_id)

                            # Safely extract disorder percent
                            disorder_percent = disprot_info.get("disorder_percent")
                            disorder_txt = (
                                f"{disorder_percent:.1f} %" if disorder_percent is not None else "unknown"
                            )

                            # Determine if it's an IDP: has DisProt ID + disorder >= 30%
                            is_idp = (
                                disprot_info.get("found", False) and
                                disorder_percent is not None and
                                disorder_percent >= 30.0
                            )

                            # Define the links
                            if disprot_info.get("found", False):
                                disprot_link = f"https://disprot.org/{disprot_info['disprot_id']}"
                                view_text = f"View {disprot_info['disprot_id']} ({base_id}) in DisProt"
                            else:
                                disprot_link = f"https://disprot.org/browse?sort_field=disprot_id&sort_value=asc&page_size=20&page=0&release=current&show_ambiguous=true&show_obsolete=false&acc={base_id}"
                                view_text = f"View {base_id} in DisProt"

                            if is_idp:
                                st.success(f"**{base_id}** is considered an IDP (Disorder content: **{disorder_txt}**)")
                                # Normal clickable link
                                st.markdown(
                                    f'Explore DisProt: <a href="{disprot_link}" target="_blank">{view_text}</a>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                # ---- NOT an IDP or insufficient disorder ----------------
                                if not disprot_info.get("found", False):
                                    st.warning(f"**{base_id}** not found in DisProt.")
                                else:
                                    st.warning(
                                        f"**{base_id}** is **not** considered an IDP "
                                        f"(Disorder content: **{disorder_txt}**)."
                                    )
                                
                                # Blurred, non-clickable link (still shows URL)
                                st.markdown(
                                    f'Explore DisProt: '
                                    f'<span style="color: #888; text-decoration: none;">{view_text}</span> '
                                    f'<span style="font-size: 0.8em; color: #aaa;"></span>',
                                    unsafe_allow_html=True,
                                )
                            
                            # DEBUG: print PTM structures passed to the viewer so we can inspect them when PTMs don't appear
                            #try:
                                #st.write("DEBUG: ptm_data (left):", ptm_data[condition1_name])
                                #st.write("DEBUG: ptm_data (right):", ptm_data[condition2_name])
                                #st.write("DEBUG: protein_ptms:", protein_ptms if 'protein_ptms' in locals() else None)
                            #except Exception:
                                # don't break rendering if debug printing fails
                                #pass
                            
                            # ---- NEW: also normalize the protein-level fallback ----
                            if 'protein_ptms' in locals() and protein_ptms:
                                protein_ptms = normalize_ptm_data(protein_ptms)

                            render_synced_viewers(pdb_str, viewer_residue_data, bg_color, conditions, selected_cmap, selected_not_mapped_color, viewer_ptm_data)
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                            
                            # Calculate coverage for each condition
                            coverages = {}
                            for cond in conditions:
                                coverages[cond] = (sum(1 for v in residue_data[cond] if v is not None) / seq_len * 100) if seq_len > 0 else 0
                            
                            # Linear Sequence Visualizations
                            # ======================================================
                            #              LINEAR SEQUENCE VISUALIZATIONS
                            # ======================================================
                            st.subheader("Linear Sequence Visualizations")

                            view_mode = st.session_state.view_mode
                            selected_peptide = st.session_state.selected_peptide

                            # Decide what to show based on 3D selection
                            if view_mode == "Full Structure (All Peptides)" or selected_peptide is None:
                                # Show all peptides — default behavior
                                linear_residue_data = residue_data
                                linear_ptm_data = ptm_data
                                st.info("Linear plots show all mapped peptides.")

                            else:
                                # SHOW ONLY *ONE* PEPTIDE
                                st.success(f"Linear plots show only peptide: **{selected_peptide}**")

                                # Determine whether to filter on stripped or modified sequence
                                filter_col = (
                                    "Modified.Sequence" if view_mode == "View by Modified Sequence"
                                    else "Stripped.Sequence"
                                )

                                peptide_rows = selected_df[selected_df[filter_col] == selected_peptide]

                                if peptide_rows.empty:
                                    st.error(f"Selected peptide '{selected_peptide}' not found in data.")
                                    st.stop()

                                # Build linear data for each condition
                                linear_residue_data = {}
                                linear_ptm_data = {}

                                for cond in conditions:
                                    # List of residues for linear plot
                                    residues_linear = [None] * len(protein_seq)

                                    # Choose if PTM column is active
                                    ptm_col_to_use = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None

                                    # Map only this peptide
                                    temp_residues, temp_ptms = map_peptides_to_residues(
                                        peptide_rows,
                                        protein_seq,
                                        intensity_col=cond,
                                        overlap_strategy=overlap_strategy,
                                        ptm_col=ptm_col_to_use,
                                        apply_tryptic=st.session_state.apply_tryptic,
                                        proteotypic_only=st.session_state.proteotypic_only
                                    )

                                    # Copy mapped residues only
                                    for i, v in enumerate(temp_residues):
                                        if v is not None:
                                            residues_linear[i] = v

                                    linear_residue_data[cond] = residues_linear

                                    # Rebuild PTM data ONLY for this peptide
                                    cond_ptm = {}
                                    if st.session_state.ptm_enabled and temp_ptms:
                                        for um, positions in temp_ptms.items():
                                            if um in st.session_state.selected_unimods:
                                                cfg = st.session_state.ptm_configs.get(um, {})
                                                cond_ptm[um] = {
                                                    "positions": sorted(list(positions)),
                                                    "selected": cfg.get("selected", True),
                                                    "color": cfg.get("color", "#3700FF"),
                                                    "label": cfg.get("label", um)
                                                }
                                    linear_ptm_data[cond] = cond_ptm

                            # --------------------------------------------------------------
                            # RENDER LINEAR PLOTS FOR EACH CONDITION
                            # --------------------------------------------------------------
                            first_condition = conditions[0] if conditions else None
                            for cond in conditions:
                                mapped_count = sum(1 for v in linear_residue_data[cond] if v is not None)
                                coverage_pct = (mapped_count / len(protein_seq) * 100) if len(protein_seq) > 0 else 0
                                short_title = f"{cond} (Coverage: {coverage_pct:.1f}%)"
                                #label_style = "font-size:16px; font-weight:bold; color:#87CEEB;" if cond == first_condition else "font-size:15px; color:#666;"
                                st.markdown(
                                    f'<div style="text-align:center; margin:12px 0 6px 0; {short_title}</div>',
                                    unsafe_allow_html=True
                                )
                                is_first = (cond == first_condition)
                                render_linear_plot(
                                    linear_residue_data[cond],
                                    short_title,
                                    seq_len,
                                    min_max_logs[cond][0],
                                    min_max_logs[cond][1],
                                    protein_seq,
                                    model_name,
                                    plddt_list,
                                    mean_plddt,
                                    cmap_name=selected_cmap,
                                    not_mapped_color=selected_not_mapped_color,
                                    ptm_data=linear_ptm_data[cond],
                                    show_full_header=is_first,
                                    show_ptm_legend=is_first
                                )

                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

                            # Colorbar
                            st.subheader("Colorbar")
                            overall_vmin = min(min_max_logs[cond][0] for cond in conditions)
                            overall_vmax = max(min_max_logs[cond][1] for cond in conditions)
                            cbar_fig, cbar_ax = plt.subplots(figsize=(8, 0.3))
                            norm = Normalize(vmin=overall_vmin, vmax=overall_vmax)
                            sm = ScalarMappable(cmap=colormaps[selected_cmap], norm=norm)
                            cbar = cbar_fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                            if st.session_state.is_frequency:
                                cbar.set_label('Frequency- Z-score', fontsize=10)
                            else:
                                cbar.set_label('Z-Score Intensity', fontsize=10)
                            cbar.ax.tick_params(labelsize=9)
                            #cbar.outline.set_visible(False)
                            #cbar_fig.patch.set_alpha(0.0)
                            #cbar_ax.set_facecolor((0,0,0,0))
                            plt.tight_layout()
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=False)
                            buf.seek(0)
                            img_str = base64.b64encode(buf.getvalue()).decode()
                            plt.close(cbar_fig)
                            html_content = f"""
                            <div style="display: flex; justify-content: center; align-items: center; width: 100%; max-width: 600px; margin: 10px auto;">
                                <img src="data:image/png;base64,{img_str}" style="width: 100%; max-width: 500px; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                            </div>
                            """
                            st.components.v1.html(html_content, height=80)
                            
                            # --- Quantitative Visualization ---
                            st.subheader("📈 Quantitative Visualization")

                            # --- Select peptide type column ---
                            available_cols = [c for c in ["Stripped.Sequence", "Modified.Sequence"] if c in selected_df.columns]
                            if not available_cols:
                                st.warning("No peptide columns found (expected 'Stripped.Sequence' or 'Modified.Sequence').")
                            else:
                                peptide_type = st.selectbox("Type of Peptide", available_cols, index=0, key="peptide_type_selectbox")
                            # --- Build peptide list based on selected type ---
                                if peptide_type == "Modified.Sequence":
                                    # Only keep peptides containing any UniMod tag
                                    mod_mask = selected_df["Modified.Sequence"].astype(str).str.contains(r"UniMod:\d+", regex=True, na=False)
                                    filtered_df = selected_df.loc[mod_mask].copy()

                                    # Extract all UniMod numbers present
                                    all_text = " ".join(filtered_df["Modified.Sequence"].dropna().astype(str).tolist())
                                    unimods = sorted(set(re.findall(r"UniMod:\d+", all_text)))

                                    # Show detected UniMods to user
                                    st.markdown(f"### UniMod types detected: `{', '.join(unimods) if unimods else 'None found'}`")

                                    # Optional: allow user to filter by UniMod type
                                    selected_unimod = st.selectbox("Filter by UniMod type (optional)", ["All"] + unimods, key="unimod_selectbox")
                                    if selected_unimod != "All":
                                        filtered_df = filtered_df[
                                            filtered_df["Modified.Sequence"].astype(str).str.contains(selected_unimod, regex=False)
                                        ]

                                    # *** NEW: add position to every entry ***
                                    peptide_with_pos = []
                                    for pep in filtered_df["Modified.Sequence"].dropna().unique():
                                        pos = find_peptide_position(protein_seq, pep)
                                        peptide_with_pos.append(f"{pep} (pos: {pos})")

                                    def extract_position(p):
                                        m = re.search(r"pos:\s*(\d+)", p)
                                        return int(m.group(1)) if m else 999999

                                    available_peptides = sorted(peptide_with_pos, key=extract_position)    # <-- dropdown shows pos
                                else:
                                    # Stripped sequence – same logic
                                    peptide_with_pos = []
                                    for pep in selected_df["Stripped.Sequence"].dropna().unique():
                                        pos = find_peptide_position(protein_seq, pep)
                                        peptide_with_pos.append(f"{pep} (pos: {pos})")

                                    def extract_position(p):
                                        m = re.search(r"pos:\s*(\d+)", p)
                                        return int(m.group(1)) if m else 999999

                                    available_peptides = sorted(peptide_with_pos, key=extract_position)

                                # --- Continue if we have peptides ---
                                if not available_peptides:
                                    st.warning(f"No peptides found in '{peptide_type}' column.")
                                else:
                                    col_plot, col_opts = st.columns([1, 1])

                                    with col_opts:
                                        st.markdown("### ⚙️ Plot Settings")
                                        selected_peptide_with_pos = st.selectbox(f"Select Peptide ({peptide_type})", available_peptides, index=0, key="peptide_selectbox")
                                        selected_peptide = re.split(r"\s*\(pos:",selected_peptide_with_pos)[0].strip()  # Remove position for matching")
                                        plot_type = st.selectbox("Plot Type", ["Box", "Violin"], index=0, key="plot_type_selectbox")
                                        add_swarm = st.radio("Add Swarm Overlay", ["Yes", "No"], horizontal=True, index=0, key="swarm_radio")

                                        # Define condition1_name and condition2_name based on metadata['Group']
                                        unique_groups = metadata['Group'].dropna().unique()
                                        if len(unique_groups) >= 2:
                                            # Default to first two groups, or let user select
                                            condition1_name = st.selectbox("Select Group 1", unique_groups, index=0, key="condition1_selectbox")
                                            condition2_name = st.selectbox("Select Group 2", unique_groups, index=1 if len(unique_groups) > 1 else 0, key="condition2_selectbox")
                                        else:
                                            st.warning("Not enough unique groups in metadata. At least two groups are required.")
                                            condition1_name = unique_groups[0] if unique_groups else "Group1"
                                            condition2_name = unique_groups[0] if unique_groups else "Group2"

                                        color_group1 = st.color_picker(f"Color for {condition1_name}", "#1f77b4", key="color_group1")
                                        color_group2 = st.color_picker(f"Color for {condition2_name}", "#d62728", key="color_group2")
                                        swarm_color = st.color_picker("Swarm Dot Color", "#87CEEB", key="swarm_color")
                                        log10_checkbox = st.checkbox("Convert Intensities to log10 scale", value=True, key="log10_checkbox")
                                        show_grid = st.checkbox("Show Grid Lines", value=True, key="grid_checkbox")
                                        # --------------------------------------------------------------
                                        # 1. GLOBAL PALETTE – works for BOTH Intensity & Frequency
                                        # --------------------------------------------------------------
                                        unique_groups = metadata["Group"].dropna().unique()
                                        default_palette = sns.color_palette("tab10", n_colors=max(3, len(unique_groups)))
                                        group_colors = [default_palette[i % len(default_palette)] for i in range(len(unique_groups))]

                                        # override user-picked colours
                                        if condition1_name in unique_groups:
                                            group_colors[list(unique_groups).index(condition1_name)] = color_group1
                                        if condition2_name in unique_groups:
                                            group_colors[list(unique_groups).index(condition2_name)] = color_group2

                                        palette = dict(zip(unique_groups, group_colors))
                                    with col_plot:
                                        import numpy as np
                                        import pandas as pd

                                        # Prepare plotting data -----------------------------------------------
                                        groups_to_samples = metadata.groupby("Group")["File_Name"].apply(list).to_dict()
                                        plot_data = []

                                        for g, samples in groups_to_samples.items():
                                            for s in samples:
                                                if s not in df.columns:
                                                    continue
                                                df_subset = selected_df[selected_df[peptide_type] == selected_peptide]
                                                if not df_subset.empty:
                                                    vals = df_subset[s].dropna().tolist()
                                                    for v in vals:
                                                        plot_data.append([selected_peptide, g, s, v])

                                        df_plot = pd.DataFrame(plot_data, columns=["Peptide", "Group", "Sample", "Intensity"])

                                        # Prepare a default palette mapping for all known groups (from metadata)
                                        try:
                                            all_groups = list(groups) if 'groups' in locals() else []
                                        except Exception:
                                            all_groups = []
                                        default_colors_for_groups = sns.color_palette("tab10", n_colors=max(3, len(all_groups)))
                                        colors_for_groups_global = [default_colors_for_groups[i % len(default_colors_for_groups)] for i in range(len(all_groups))]
                                        # override colors for condition1/condition2 if present
                                        if 'condition1_name' in locals() and condition1_name in all_groups:
                                            idx = all_groups.index(condition1_name)
                                            colors_for_groups_global[idx] = color_group1
                                        if 'condition2_name' in locals() and condition2_name in all_groups:
                                            idx = all_groups.index(condition2_name)
                                            colors_for_groups_global[idx] = color_group2
                                        palette = {g: c for g, c in zip(all_groups, colors_for_groups_global)}

                                    # --------------------------------------------------------------
                                        # 2. PLOT MODE (Intensity vs Frequency)
                                        # --------------------------------------------------------------
                                        plot_mode = st.radio("Plot mode:", ["Intensity", "Frequency"], index=0)

                                        plot_container = st.container()
                                        plot_slot      = plot_container.empty()

                                        # ------------------------------------------------------------------
                                        # 2-a  FREQUENCY MODE
                                        # ------------------------------------------------------------------
                                        if plot_mode == "Frequency":
                                            # ----- frequency calculation (unchanged) -----
                                            freq_mode   = st.radio(
                                                "Frequency mode:",
                                                ["Include zeros (NA->0 counted)", "Non-zero only (exclude blanks)"],
                                                index=0,
                                            )
                                            show_percent = st.checkbox("Show as percent (0-100)", value=False,key="show_percent_checkbox")

                                            freq_records = []
                                            pep_rows = selected_df[selected_df[peptide_type] == selected_peptide]

                                            for grp, sample_cols in groups_to_samples.items():
                                                if not sample_cols:
                                                    freq_records.append({"Group": grp, "Frequency": np.nan})
                                                    continue

                                                per_sample = []
                                                for s in sample_cols:
                                                    s_vals = pd.to_numeric(pep_rows[s], errors='coerce') if s in pep_rows.columns else pd.Series(dtype=float)
                                                    orig_non_blank = s_vals.notna().any()
                                                    any_nonzero    = s_vals.fillna(0).astype(float).ne(0).any()
                                                    per_sample.append({"sample": s, "any_nonzero": any_nonzero, "orig_non_blank": orig_non_blank})

                                                num_nonzero = sum(r["any_nonzero"] for r in per_sample)
                                                denom = len(per_sample) if freq_mode.startswith("Include") else \
                                                        (sum(r["orig_non_blank"] for r in per_sample) or len(per_sample))

                                                freq = float(num_nonzero) / denom if denom > 0 else np.nan
                                                if show_percent and not pd.isna(freq):
                                                    freq *= 100.0
                                                freq_records.append({"Group": grp, "Frequency": freq})

                                            df_freq = pd.DataFrame(freq_records).set_index("Group")
                                            st.write("Frequency of summary Peptide:", selected_peptide_with_pos)
                                            st.dataframe(df_freq.style.format("{:.2f}" if not show_percent else "{:.1f}%"))

                                            # ----- NEW FIGURE / AXIS (never reuse the intensity one) -----
                                            freq_fig, freq_ax = plt.subplots(figsize=(11, 6))
                                            freq_fig.patch.set_alpha(0.0)
                                            freq_ax.set_facecolor((0, 0, 0, 0))

                                            # ----- colours for the bar plot -----
                                            bar_colors = [palette.get(g, default_palette[0]) for g in df_freq.index]

                                            # ----- bar plot -----
                                            bars = df_freq["Frequency"].plot(
                                                kind="bar", color=bar_colors, ax=freq_ax,
                                                edgecolor="white", linewidth=1.2
                                            )

                                            ylabel = "Frequency (%)" if show_percent else "Frequency"
                                            freq_ax.set_ylabel(ylabel, fontsize=14, color="white")
                                            freq_ax.set_xlabel("Group", fontsize=14, color="white")
                                            freq_ax.set_ylim(0, 105 if show_percent else 1.05)
                                            freq_ax.set_title(selected_peptide, fontsize=14, color="white",
                                                            weight="bold", pad=12)
                                            freq_ax.tick_params(axis="x", rotation=0, colors="white", labelsize=16)
                                            freq_ax.tick_params(axis="y", colors="white", labelsize=16)

                                            # value labels on top of bars
                                            for p in bars.patches:
                                                h = p.get_height()
                                                if pd.isna(h):
                                                    continue
                                                label = f"{h:.1f}%" if show_percent else f"{h:.2f}"
                                                y_pos = min(h, (105 if show_percent else 1.05) * 0.98)
                                                freq_ax.annotate(
                                                    label,
                                                    (p.get_x() + p.get_width() / 2, y_pos),
                                                    ha="center", va="bottom", fontsize=14, color="white",
                                                    xytext=(0, 3), textcoords="offset points",
                                                )

                                            freq_ax.grid(show_grid, color="white", alpha=0.3,
                                                        linestyle="--", linewidth=0.5)
                                            sns.despine(left=False, bottom=False)
                                            plt.subplots_adjust(top=0.88, bottom=0.15, left=0.15, right=0.95)
                                            plt.tight_layout()

                                            # ----- render + download -----
                                            with plot_container:
                                                plot_slot.pyplot(freq_fig)

                                                buf = io.BytesIO()
                                                freq_fig.savefig(buf, format="png", dpi=300,
                                                                bbox_inches="tight", transparent=True)
                                                buf.seek(0)

                                                st.download_button(
                                                    label="Download Frequency Plot (PNG)",
                                                    data=buf.getvalue(),
                                                    file_name=f"frequency_{selected_peptide.replace(' ', '_')}.png",
                                                    mime="image/png",
                                                )

                                            plt.close(freq_fig)

                                        # ------------------------------------------------------------------
                                        # 2-b  INTENSITY MODE (Box / Violin)
                                        # ------------------------------------------------------------------
                                        else:
                                            # ----- average duplicates per sample -----
                                            if not df_plot.empty:
                                                df_plot = df_plot.groupby(
                                                    ["Peptide", "Group", "Sample"], as_index=False
                                                ).agg({"Intensity": "mean"})

                                            # ----- log10 transform (optional) -----
                                            if log10_checkbox and not df_plot.empty:
                                                df_plot["Intensity"] = np.log10(df_plot["Intensity"] + 1)
                                                ylabel = "log10(Intensity + 1)"
                                            else:
                                                ylabel = "Intensity"

                                            # ----- NEW FIGURE / AXIS -----
                                            int_fig, int_ax = plt.subplots(figsize=(8, 6))
                                            int_fig.patch.set_alpha(0.0)
                                            int_ax.set_facecolor((0, 0, 0, 0))
                                            sns.set_style("whitegrid" if show_grid else "white")
                                            sns.set_context("talk")

                                            # ----- plot -----
                                            if plot_type == "Box":
                                                sns.boxplot(
                                                    data=df_plot, x="Group", y="Intensity", hue="Group",
                                                    palette=palette, ax=int_ax,
                                                    boxprops=dict(alpha=0.5, linewidth=1.2, edgecolor="white"),
                                                    medianprops=dict(color="black", linewidth=1.5),
                                                    linewidth=1.2,
                                                )
                                            else:   # Violin
                                                sns.violinplot(
                                                    data=df_plot, x="Group", y="Intensity", hue="Group",
                                                    palette=palette, ax=int_ax, inner=None,
                                                    linewidth=1.2, cut=0, legend=False,
                                                )

                                            if add_swarm == "Yes":
                                                sns.swarmplot(
                                                    data=df_plot, x="Group", y="Intensity",
                                                    color=swarm_color, edgecolor="white",
                                                    linewidth=0.6, alpha=0.9, size=5, ax=int_ax
                                                )

                                            int_ax.set_title(f"Peptide: {selected_peptide_with_pos}",
                                                            fontsize=12, color="white", weight="bold", pad=10)
                                            int_ax.set_ylabel(ylabel, fontsize=12, color="white")
                                            int_ax.set_xlabel("Group", fontsize=12, color="white")
                                            int_ax.tick_params(colors="white", labelsize=9)
                                            int_ax.grid(show_grid, alpha=0.3, linestyle="--",
                                                        linewidth=0.5, color="white")
                                            sns.despine(left=False, bottom=False)
                                            if not df_plot.empty:
                                                y_data = df_plot["Intensity"]
                                                y_min, y_max = y_data.min(), y_data.max()
                                                padding = (y_max - y_min) * 0.05
                                                if padding == 0:  # flat data
                                                    padding = 0.1
                                                int_ax.set_ylim(y_min - padding, y_max + padding)
                                            #plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

                                            # ----- render + download -----
                                            with plot_container:
                                                plot_slot.pyplot(int_fig)

                                                buf = io.BytesIO()
                                                int_fig.savefig(buf, format="png", dpi=300,
                                                                bbox_inches="tight", transparent=True)
                                                buf.seek(0)

                                                st.download_button(
                                                    label="Download Intensity Plot (PNG)",
                                                    data=buf.getvalue(),
                                                    file_name=f"intensity_{selected_peptide.replace(' ', '_')}_{plot_type.lower()}.png",
                                                    mime="image/png",
                                                )

                                            plt.close(int_fig)
                                        # Render the figure into the shared plot placeholder to avoid duplicate stacked outputs
with quant_diff_tab:
        st.session_state.active_quant_tab = "diff"
        st.session_state.processed_single = {"processed": False}
        if 'confirmed' not in st.session_state:
            st.session_state.confirmed = False
        if 'processed' not in st.session_state:
            st.session_state.processed = False
        if 'apply_log2' not in st.session_state:
            st.session_state.apply_log2 = False
        if 'apply_neglog10' not in st.session_state:
            st.session_state.apply_neglog10 = False
        if 'auto_link_transforms' not in st.session_state:
            st.session_state.auto_link_transforms = True
        st.subheader("Differential Quantitative Analysis")
        st.markdown(
            """
            <p style='text-align:center; font-size:18px; color:#32CD32;'>
            Available Now! The Differential Quantitative Analysis is ready with exciting features 
            to compare peptide intensity data on 3D protein structures using fold changes and p-values.
            </p>
            """,
            unsafe_allow_html=True,
        )
        
        # File upload
        csv_file_q = st.file_uploader("Upload Peptide Intensity CSV (e.g., Input_Two_Condition.csv)", type=["csv"], key="diff_peptide_csv_uploader")
        fasta_file_q = st.file_uploader("Upload FASTA", type=["fasta"], key="diff_fasta_uploader")
        metadata_file = st.file_uploader("Upload Metadata CSV (File_Name, Group, Sample_Name)", type=["csv"], key="diff_metadata_csv_uploader")

        if csv_file_q and fasta_file_q and metadata_file:
            # Read files
            try:
                df = pd.read_csv(csv_file_q)
            except Exception as e:
                st.error(f"Error reading Peptide CSV: {e}")
                st.stop()

            try:
                metadata = pd.read_csv(metadata_file)
            except Exception as e:
                st.error(f"Error reading Metadata CSV: {e}")
                st.stop()

            required_meta = {"File_Name", "Group", "Sample_Name"}
            if not required_meta.issubset(set(metadata.columns)):
                st.error(f"Metadata CSV must contain columns: {required_meta}")
                st.stop()

            fasta_str = fasta_file_q.getvalue().decode("utf-8")
            fasta_handle = io.StringIO(fasta_str)
            seq_records = list(SeqIO.parse(fasta_handle, "fasta"))
            if not seq_records:
                st.error("No sequences found in FASTA file.")
                st.stop()

            def extract_uniprot_id(header: str) -> str:
                parts = header.split('|')
                if len(parts) >= 2 and parts[0] in ('sp', 'tr'):
                    return parts[1]
                return header.split()[0]

            fasta_ids = [extract_uniprot_id(rec.id) for rec in seq_records]
            fasta_with_isoform = len(fasta_ids)
            fasta_without_isoform = len(set([fid.split('-')[0] for fid in fasta_ids]))

            # Metadata: samples & groups
            num_samples = metadata.shape[0]
            num_groups = metadata['Group'].nunique()
        
            # Peptide CSV validation
            if 'Protein.Group' not in df.columns:
                st.error("Peptide CSV must contain 'Protein.Group' column.")
                st.stop()
            if 'Stripped.Sequence' not in df.columns:
                st.error("Peptide CSV must contain 'Stripped.Sequence' column.")
                st.stop()

            prot_ids = df['Protein.Group'].dropna().astype(str).unique().tolist()
            prot_with_isoform = len(prot_ids)
            prot_without_isoform = len(set([p.split('-')[0] for p in prot_ids]))
            num_peptides_raw = df['Stripped.Sequence'].notna().sum()
            num_peptides_unique = df['Stripped.Sequence'].dropna().nunique()
        
            # PTM detection
            ptm_col_candidates = [c for c in ['Modified.Sequence', 'PTM'] if c in df.columns]
            ptm_detected = False
            unimods = ["NA"]
            if ptm_col_candidates:
                col_series = df[ptm_col_candidates[0]].dropna().astype(str)
                ptm_detected = col_series.str.contains('UniMod:', case=False).any()
                if ptm_detected:
                    all_text = ' '.join(col_series.tolist())
                    unimods = sorted(set(re.findall(r'UniMod:\d+', all_text)))
            
            st.markdown("### 🧾 File Information Summary")
            st.info(
                f"**FASTA File**\n"
                f"- Total Proteins (without isoform): {fasta_without_isoform}\n"
                f"- Total Proteins (with isoform): {fasta_with_isoform}\n\n"
                f"**Metadata File**\n"
                f"- Samples Detected: {num_samples}\n"
                f"- Groups Detected: {num_groups}\n\n"
                f"**Peptide CSV File**\n"
                f"- Proteins Detected (without isoform): {prot_without_isoform}\n"
                f"- Proteins Detected (with isoform): {prot_with_isoform}\n"
                f"- Stripped Sequences: {num_peptides_raw} (Unique: {num_peptides_unique})\n"
                f"- PTM Detected: {'Yes' if ptm_detected else 'No'}\n"
                f"- PTM Types: {', '.join(unimods)}"
            )

            # Identify sample intensity columns
            sample_candidates = metadata['File_Name'].astype(str).tolist()
            df_cols = set(map(str, df.columns))
            sample_cols = [c for c in sample_candidates if c in df_cols]
            if len(sample_cols) == 0:
                st.error("No sample columns found in the Peptide CSV that match Metadata['File_Name'].")
                st.stop()

            # Map samples to groups
            meta_map = metadata.set_index('File_Name')['Group'].to_dict()
            groups = metadata['Group'].unique().tolist()
            group_to_samples = {g: [s for s in sample_cols if meta_map.get(s) == g] for g in groups}
            empty_groups = [g for g, cols in group_to_samples.items() if len(cols) == 0]
            if empty_groups:
                st.warning(f"Groups with no matching sample columns: {empty_groups}")

            # PTM options
            if ptm_detected:
                st.session_state.all_unimods = unimods
            else:
                st.session_state.all_unimods = []
                st.session_state.selected_unimods = []
            has_ptm = bool(st.session_state.all_unimods)
            ptm_checkbox_disabled = not has_ptm
            st.session_state.ptm_enabled = st.checkbox("Enable PTM Annotation", disabled=ptm_checkbox_disabled, value=False if ptm_checkbox_disabled else st.session_state.get("ptm_enabled", False), key="quant_diff_ptm_enable")
            st.session_state.apply_tryptic = st.checkbox("Apply Tryptic Rule (K/R cleavage)", value=st.session_state.get("apply_tryptic", False))
            st.session_state.proteotypic_only = st.checkbox(
                "Proteotypic Peptides Only",
                value=st.session_state.get("proteotypic_only", True),
                key="proteotypic_only_checkbox_diff"
            )

            # PTM selection
            if st.session_state.all_unimods:
                st.markdown("### Detected PTM UniMod IDs")
                if 'selected_unimods' not in st.session_state:
                    st.session_state.selected_unimods = st.session_state.all_unimods.copy()
                selected_unimods = st.multiselect(
                    "Select UniMod IDs to Include",
                    options=st.session_state.all_unimods,
                    default=st.session_state.selected_unimods,
                    key="diff_unimod_multiselect"
                )
                st.session_state.selected_unimods = selected_unimods

            # Condition selection
            if len(groups) < 2:
                st.error(f"At least two groups required. Found: {groups}")
                st.stop()

            # ─── Detect available differential columns ───────────────────────────────
            import re

            fc_patterns = r'(_log2FC|_FC|_logFC|FoldChange|log2FoldChange)$'
            pval_patterns = r'(_padj|_pvalue|_adjp|_FDR|P\.Value|adj\.P\.Val)$'

            all_cols = df.columns.tolist()

            fc_columns = sorted([c for c in all_cols if re.search(fc_patterns, c, re.IGNORECASE)])
            pval_columns = sorted([c for c in all_cols if re.search(pval_patterns, c, re.IGNORECASE)])

            #available_diff_cols = sorted(set(fc_columns + pval_columns))

            if not fc_columns and not pval_columns:
                st.error("""
                No differential result columns detected in the uploaded CSV.
                
                Please include at least one column matching these patterns:
                • Fold Change:   ..._log2FC, ..._FC, ..._logFC
                • P-value:       ..._padj, ..._pvalue, ..._adjp
                Please uppload a file with pre-computed differential results.""")
                st.stop()

            st.markdown("### Detected Differential Columns")
            if fc_columns:
                st.info("Fold change columns:\n" + "\n".join(f"- {c}" for c in fc_columns) if fc_columns else "None found")
            if pval_columns:
                st.info("P-value columns:\n" + "\n".join(f"- {c}" for c in pval_columns) if pval_columns else "None found")

            # Let user choose which ones to visualize
            #selected_columns = st.multiselect(
                #"Select differential columns to visualize (1–5 recommended)",
                #options=available_diff_cols,
                #default=available_diff_cols[:min(3, len(available_diff_cols))],
                #key="diff_selected_columns"
            #)

            #if not selected_columns:
                #st.warning("Please select at least one differential column to proceed.")
                #st.stop()

            # This becomes your working list — real column names!
            #conditions = selected_columns
            if st.button("Confirm & Proceed", use_container_width=True, key="confirm_diff_conditions"):
                st.session_state.confirmed = True
                st.session_state.processed = False
                st.rerun()

            if st.session_state.confirmed:
                with st.container():
                    st.info("✅ Conditions confirmed. Using pre-computed fold changes from CSV.")

                    # Check and compute p-values only if missing
              
                    visualize_by = st.radio("Visualize by:", ["Fold Change", "P-value"], horizontal=True, key="visualize_by_radio_diff")
                    # Set the active list of columns based on choice
                    if visualize_by == "Fold Change":
                        conditions = fc_columns
                        if not conditions:
                            st.error("No fold change columns detected. Cannot proceed in Fold Change mode.")
                            st.stop()
                    else:
                        conditions = pval_columns
                        if not conditions:
                            st.error("No p-value columns detected. Cannot proceed in P-value mode.")
                            st.stop()

                    st.info(f"Will visualize **{len(conditions)}** {visualize_by.lower()} comparisons:")
                    st.write(", ".join(conditions))
                    apply_log2 = False
                    show_log2_checkbox = False

                    if visualize_by == "Fold Change" and conditions:
                        # Heuristic: check if ANY column name suggests it's already log₂ transformed
                        log2_indicators = [
                            '_log2fc', '_log2FC', 'log2foldchange', 'log2_foldchange',
                            'log2_fc', 'log2-foldchange', 'log2 fold change'
                        ]
                        
                        is_already_log2 = any(
                            any(indicator in col.lower() for indicator in log2_indicators)
                            for col in conditions
                        )
                        
                        if is_already_log2:
                            show_log2_checkbox = False
                            apply_log2 = False
                            st.info("Detected **log₂-transformed** fold changes → no additional transformation needed.")
                        else:
                            show_log2_checkbox = True
                            apply_log2 = True  # default: apply log2 (most common case)
                            st.info("Fold changes appear to be **raw ratios** → log₂ transformation option enabled.")
                    apply_neglog10 = False
                    show_neglog10_checkbox = False

                    if visualize_by == "P-value" and conditions:
                        neglog10_indicators = [
                            '-log10p', 'neglog10p', 'log10pvalue', 'log10pval', 
                            '-log10_p', 'neg_log10p', '-log10(p)', 'log10(pvalue)'
                        ]
                            
                        is_already_neglog10 = any(
                            any(ind in col.lower() for ind in neglog10_indicators)
                            for col in conditions
                        )
                            
                        if is_already_neglog10:
                            show_neglog10_checkbox = False
                            apply_neglog10 = False
                            st.info("Detected **-log₁₀-transformed** p-values → no transformation needed.")
                        else:
                            show_neglog10_checkbox = True
                            apply_neglog10 = True  # default = apply -log10
                            st.info("P-values appear to be **raw probabilities** → -log₁₀ transformation option enabled.")
                    
                    apply_log2 = st.session_state.apply_log2
                    apply_neglog10 = st.session_state.apply_neglog10
                    if  show_log2_checkbox:
                        st.session_state.apply_log2 = st.checkbox(
                            "Apply log₂ transformation to Fold Changes",
                            value=st.session_state.apply_log2,
                            key="apply_log2_checkbox_diff",
                            help="Check if your fold changes are raw ratios (e.g. 2.0 = 2-fold up). Uncheck if already log₂-transformed."
                        )
                        apply_log2 = st.session_state.apply_log2
                    if  show_neglog10_checkbox:
                        st.session_state.apply_neglog10 = st.checkbox(
                            "Apply -log₁₀ transformation to P-values",
                            value=st.session_state.apply_neglog10,
                            key="apply_neglog10_checkbox_diff",
                            help="Check if your p-values are raw probabilities (e.g. 0.05). Uncheck if already -log₁₀-transformed."
                        )
                        apply_neglog10 = st.session_state.apply_neglog10
                    # ─── Automatic linking (run EARLY to set values before checkboxes) ────────────────────────────────
                    auto_link = st.checkbox(
                        "Automatically link transformations (log₂ FC ↔ -log₁₀ P-value)",
                        value=st.session_state.auto_link_transforms,
                        key="auto_link_checkbox_diff",
                        help="When enabled, checking one auto-applies the other."
                    )
                    st.session_state.auto_link_transforms = auto_link

                    if auto_link:
                        # Bidirectional: FC log₂ → p-value -log₁₀, and vice versa
                        if st.session_state.apply_log2:
                            st.session_state.apply_neglog10 = True
                            st.info("Auto-linked: Applied -log₁₀ to P-values (scatter plot updated).")

                        if st.session_state.apply_neglog10:
                            st.session_state.apply_log2 = True
                            st.info("Auto-linked: Applied log₂ to Fold Changes (scatter plot updated).")        
                    st.info("Now select protein and options.")
                    protein_options = sorted(df['Protein.Group'].unique())
                    selected_protein = st.selectbox("Select Protein", protein_options, key="diff_protein_select")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        combine_isoforms = st.selectbox("Combine Isoforms?", ["yes", "no"], key="diff_combine_isoforms")
                    with col2:
                        overlap_strategy = st.selectbox("Overlap Strategy", ["none", "merge", "highest"], key="diff_overlap_strategy")
                    
                    # PDB source selection with hyperlinks (same as multi)
                    st.markdown(
                        f'<div style="text-align:left; margin-bottom:10px;">'
                        f'<span style="font-size:16px; color:#FFFFFF;">Databases: </span>'
                        f'<a href="https://alphafold.ebi.ac.uk/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">AlphaFold Database</a>'
                        f' | '
                        f'<a href="https://esmatlas.com/resources?action=fold" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">ESM Atlas</a>'
                        f' | '
                        f'<a href="https://build.nvidia.com/mit/boltz2" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">NVIDIA Boltz-2</a>'
                        f' | '
                        f'<a href="https://design-a-protein.com/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">SeqHub</a>'
                        f' | '
                        f'<a href="https://www.rcsb.org/" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">RCSB PDB</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.pdb_source = st.selectbox(
                        "Select PDB Source",
                        ["AlphaFold", "Upload PDB"],
                        help="Choose to fetch the structure from AlphaFold or upload a PDB file named as UniProtID.pdb"
                    )
                    if st.session_state.pdb_source == "Upload PDB":
                        st.session_state.uploaded_pdb = st.file_uploader(
                            "Upload PDB File",
                            type=["pdb"],
                            help="Upload a PDB file named as {UniProt_ID}.pdb matching the selected protein"
                        )
                    
                    if st.button("Process Protein", use_container_width=True, key="process_protein_diff"):
                        st.session_state.processed = True
                        st.rerun()

                if st.session_state.processed:
                    with st.container():
                        st.info("🔄 Processing... (This may take a moment for PDB fetch or upload.)")
                        base_id = selected_protein.split('-')[0]
                        protein_seq = None
                        matched_header="Not Found"
                        for rec in seq_records:
                            header = rec.id
                        
                            candidate = header.split('|')[1] if '|' in header and header.split('|')[0] in ('sp', 'tr') else header.split()[0]
                            if candidate.split('-')[0] == base_id:
                                    protein_seq = str(rec.seq)
                                    matched_header = header
                                    st.success(f"Matched by ID: {header}")
                                    break
                        if protein_seq is None:
                            st.info(f"No direct FASTA header match for {base_id}. Attempting peptide-based matching...")
                            peptides_unique = selected_df['Stripped.Sequence'].dropna().astype(str).str.strip().str.upper().unique().tolist()
                            peptides_unique = [p for p in peptides_unique if len(p) >= 7]

                            best_rec = None
                            best_count = 0

                            for rec in seq_records:
                                seq_str = str(rec.seq).upper()
                                count = sum(1 for p in peptides_unique if p in seq_str)
                                if count > best_count:
                                    best_count = count
                                    best_rec = rec

                            if best_count >= 2:  # at least 2 peptides = reliable
                                protein_seq = str(best_rec.seq)
                                matched_header = best_rec.id
                                st.success(f"Best match: {matched_header} ({best_count} peptides matched)")
                            elif len(seq_records) == 1:
                                protein_seq = str(seq_records[0].seq)
                                matched_header = seq_records[0].id
                                st.warning(f"Only one FASTA entry → using: {matched_header}")
                            else:
                                st.error(f"Could not match protein sequence. Check FASTA headers or peptide data.")
                                st.stop()

                        seq_len = len(protein_seq)
                        # CRITICAL: SAVE TO SESSION STATE SO SUMMARY CAN SEE IT!
                        st.session_state.protein_seq = protein_seq
                        st.session_state.selected_protein = selected_protein
                        st.session_state.matched_fasta_header = matched_header
                        # Isoform handling
                        isoforms = df[df['Protein.Group'].str.contains(selected_protein + r'(?:-\d+)?$', regex=True)]['Protein.Group'].unique()
                        if len(isoforms) > 1 and combine_isoforms == "yes":
                            st.info("Isoforms Detected")
                            selected_groups = list(isoforms)
                        elif len(isoforms) > 1 and combine_isoforms == "no":
                            selected_groups = st.multiselect("Select Isoforms", options=list(isoforms), default=list(isoforms))
                        else:
                            selected_groups = list(isoforms)
                        
                        if not selected_groups:
                            st.error("No isoforms selected.")
                            st.stop()
                        
                        selected_df = df[df['Protein.Group'].isin(selected_groups)]
                        
                        peptide_data = {}
                        residue_data = {cond: [None] * seq_len for cond in conditions}
                        ptm_data = {cond: {} for cond in conditions}
                        min_max_logs = {}
                        ptm_col = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None

                        st.session_state.selected_df = selected_df

                        for condition in conditions:
                            intensity_col = condition
                            # ← PASS ptm_col only if enabled!
                            ptm_col_to_use = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None
                            residues, ptms = map_peptides_to_residues(
                                selected_df, protein_seq, intensity_col, overlap_strategy,
                                ptm_col=ptm_col_to_use, apply_tryptic=st.session_state.apply_tryptic
                            )
                            # Transform values
                            if visualize_by == "Fold Change":
                                if apply_log2:
                                    residues = [np.log2(v) if v is not None and np.isfinite(v) and v > 0 else None for v in residues]
                            else:#p-value
                                if apply_neglog10:
                                    residues = [-np.log10(v) if v is not None and np.isfinite(v) and v > 0 else None for v in residues]
                            residue_data[condition] = residues
                            # Build full PTM dict with config
                            ptm_data[condition] = {}
                            if st.session_state.ptm_enabled:
                                for um, positions in ptms.items():
                                    if um in st.session_state.selected_unimods:
                                        config = st.session_state.ptm_configs.get(um, {})
                                        ptm_data[condition][um] = {
                                            'positions': sorted(list(positions)),
                                            'selected': config.get('selected', True),
                                            'color': st.session_state.ptm_configs.get(um,{}).get('color', '#3700FF'),
                                            'label': config.get('label', um)
                                        }
                        
                            covered = [v for v in residues if v is not None]
                            if not covered:
                                st.error(f"No peptides mapped for {condition}.")
                                st.stop()
                            min_max_logs[condition] = (min(covered), max(covered))
                            peptides = selected_df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
                            peptide_data[condition] = peptides
                    
                        # PTM configuration with hyperlinks (same as multi)
                        if st.session_state.ptm_enabled and st.session_state.selected_unimods:
                            st.subheader("PTM Configuration")
                            st.write(f"Selected UniMods: {st.session_state.selected_unimods}")
                            # After mapping peptides → residues and PTMs
                            for condition in conditions:
                                detected_ptms = ptms  # from map_peptides_to_residues()

                                # Initialize full dict with ALL selected UniMods (even if empty positions)
                                ptm_data[condition] = {}
                                for um in st.session_state.selected_unimods:
                                    positions = detected_ptms.get(um, set())  # Use detected, or empty set
                                    config = st.session_state.ptm_configs.get(um, {
                                        'selected': True,
                                        'label': um,
                                        'color': '#3700FF'
                                    })

                                    ptm_data[condition][um] = {
                                        'positions': sorted(list(positions)),
                                        'selected': config['selected'],
                                        'label': config['label'],
                                        'color': config['color']
                                    }

                                # Optional: clean up unselected ones at the end
                                ptm_data[condition] = {
                                    um: info for um, info in ptm_data[condition].items()
                                    if info['selected']
                                }
                                for condition in ptm_data:
                                        ptm_data[condition] = normalize_ptm_data(ptm_data[condition])
                            
                        displayed_ptms = [um for cond in ptm_data.values() for um in cond.keys() if cond[um]['selected']]
                        if not displayed_ptms:
                            st.warning("No PTMs will be displayed: either none selected or none detected in this protein.")
                        else:
                            st.success(f"Displaying {len(displayed_ptms)} PTM type(s) where detected.")
                            # Ensure ptm_configs exist for selected UniMods and render PTM configuration UI
                            if ('ptm_configs' not in st.session_state or set(st.session_state.ptm_configs.keys()) != set(st.session_state.selected_unimods)):
                                st.session_state.ptm_configs = {um: {'selected': True, 'label': f"{um}", 'color': "#3700FF"} for um in st.session_state.selected_unimods}
                            # Debug to confirm selected UniMods
                            #st.write(f"Rendering PTM config for UniMods: {st.session_state.selected_unimods}")
                            for um in st.session_state.selected_unimods:
                                col_ptm1, col_ptm2, col_ptm3 = st.columns([1, 1, 1])
                                with col_ptm1:
                                    # UniMod website expects a numeric accession (e.g. 21) not the 'UniMod:21' prefix.
                                    m = re.search(r"(\d+)", str(um))
                                    unimod_id = m.group(1) if m else str(um)
                                    direct_url = f"https://www.unimod.org/modifications_view.php?editid1={unimod_id}"
                                    st.markdown(
                                        f'<a href="{direct_url}" target="_blank" '
                                        f'style="color:#2b8cff; text-decoration:underline; font-weight:bold;" '
                                        f'title="Click to view UniMod record {unimod_id} (phosphorylation, etc.)">'
                                        f'UniMod:{unimod_id}</a>',
                                        unsafe_allow_html=True
                                    )
                                    st.session_state.ptm_configs[um]['selected'] = st.checkbox(
                                        f"Include UniMod:{um}", value=st.session_state.ptm_configs[um]['selected'], key=f"checkbox_diff_{um}"
                                    )
                                with col_ptm2:
                                    st.session_state.ptm_configs[um]['label'] = st.text_input(
                                        f"Label for UniMod:{um}", value=st.session_state.ptm_configs[um]['label'], key=f"label_diff_{um}"
                                    )
                                with col_ptm3:
                                    st.session_state.ptm_configs[um]['color'] = st.color_picker(
                                        f"Color for UniMod:{um}", value=st.session_state.ptm_configs[um]['color'], key=f"color_diff_{um}"
                                    )  
                        st.subheader("Detected Sequence")
                        st.markdown(f"**FASTA header:** {matched_header}")
                        seq_html = format_sequence_for_display(protein_seq, residue_data, conditions, line_len=150, group=20)
                        copy_html = sequence_copy_component(protein_seq)
                        st.components.v1.html(copy_html + seq_html, height=320)
                        
                        # PDB fetching or upload (same as multi)
                        if st.session_state.pdb_source == "AlphaFold":
                            pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v6.pdb"
                            with st.spinner(f"Attempting to fetch AlphaFold v6 structure for {base_id}..."):
                                try:
                                    r = requests.get(pdb_url, timeout=30)
                                    if r.status_code == 200:
                                        pdb_str = r.text
                                    elif r.status_code == 404:
                                        st.error("Model_v6 not found. The protein may not have a v6 structure yet.")
                                        st.stop()
                                    else:
                                        st.error(f"PDB fetch failed (status {r.status_code}).")
                                        st.stop()
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Failed to fetch PDB for {base_id}: {str(e)}.")
                                    st.stop()
                        else:  # Upload PDB
                            if st.session_state.uploaded_pdb is None:
                                st.error("No PDB file uploaded.")
                                st.stop()
                            # Validate filename
                            pdb_filename = st.session_state.uploaded_pdb.name
                            if not pdb_filename.endswith('.pdb'):
                                st.error("Uploaded file must have a .pdb extension.")
                                st.stop()
                            filename_id = pdb_filename[:-4]  # Remove .pdb extension
                            if filename_id != base_id:
                                st.error(f"PDB filename ({pdb_filename}) must match the selected protein's UniProt ID ({base_id}).")
                                st.stop()
                            try:
                                pdb_str = st.session_state.uploaded_pdb.getvalue().decode("utf-8")
                            except Exception as e:
                                st.error(f"Error reading uploaded PDB file: {e}")
                                st.stop()
                        
                        st.success(f"Loaded {'AlphaFold' if st.session_state.pdb_source == 'AlphaFold' else 'uploaded'} structure for {base_id} ({len(pdb_str)} bytes)")
                        plddt_list, model_name, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)
                        mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
                        st.info(f"**Mean pLDDT:** {mean_plddt_display} (Overall Confidence)")
                        bg_color = st.selectbox("Background Color", ["black", "white", "darkgrey"], index=0)
                        
                        # Adjust cmap options based on visualize_by
                        if visualize_by == "Fold Change":
                            cmap_options = ['autumn','coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                        else:
                            cmap_options = ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                        selected_cmap = st.selectbox("Select Color Gradient", cmap_options, index=0)
                        selected_not_mapped_color = st.color_picker("Select Not Mapped Color", "#d3d3d3")

                        st.subheader("PhosphoSitePlus®")
                        try:
                            uniprot_url = f"https://rest.uniprot.org/uniprotkb/{base_id}"
                            response = requests.get(uniprot_url, timeout=10)
                            if response.status_code == 200:
                                uniprot_data = response.json()
                                gene_name = uniprot_data.get('genes', [{}])[0].get('geneName', {}).get('value', 'Unknown')
                            else:
                                gene_name = 'Unknown'
                        except Exception as e:
                            gene_name = 'Unknown'
                            st.warning(f"Failed to fetch gene name for {base_id}: {e}")
                        if gene_name != 'Unknown':
                            phosphosite_url = f"https://www.phosphosite.org/simpleSearchSubmitAction.action?searchStr={gene_name}"
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore PhosphoSitePlus® : </span>'
                                f'<a href="{phosphosite_url}" target="_blank" style="font-size:20px; color:#87CEEB; text-decoration:underline;">{base_id}|{gene_name}</a>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore PhosphoSitePlus® : </span>'
                                f'<span style="font-size:20px; color:#FFFFFF;">No gene name found for {base_id}. Unable to generate PhosphoSitePlus link.</span>',
                                unsafe_allow_html=True
                            )
                        
                        with st.container():
                            st.subheader("3D Structure Visualizations")

                            # === Extract peptides and compute positions === (same as multi)

                            # Compute peptide start positions directly from protein sequence
                            def compute_positions_with_label(peptide_list):
                                result = {}
                                for pep in peptide_list:
                                    pep = str(pep).strip()
                                    if pep == "":
                                        continue
                                    pos = find_peptide_position(protein_seq, pep)
                                    label = f"{pep} (pos: {pos})"
                                    result[pep] = (pos,label)
                                return result
                            # ---- Unique peptides for dropdowns (FIXED & ROBUST) ----
                            stripped_unique = selected_df["Stripped.Sequence"].dropna().astype(str).unique().tolist()
                            # === MODIFIED SEQUENCE: ONLY THOSE WITH ACTUAL PTMs ===
                            if (st.session_state.ptm_enabled and 
                                st.session_state.selected_unimods and 
                                "Modified.Sequence" in selected_df.columns):

                                # Build regex pattern for selected UniMods only
                                pattern = '|'.join([re.escape(f"UniMod:{um.split(':')[-1]}") for um in st.session_state.selected_unimods])
                                
                                mod_mask = selected_df["Modified.Sequence"].astype(str).str.contains(pattern, case=False, na=False)
                                filtered_modified_df = selected_df.loc[mod_mask]

                                if filtered_modified_df.empty:
                                    st.warning("No peptides found with the selected UniMod(s) — falling back to stripped sequences for dropdown.")
                                    modified_unique = stripped_unique
                                else:
                                    modified_unique = filtered_modified_df["Modified.Sequence"].dropna().astype(str).unique().tolist()
                            else:
                                # PTM disabled or no UniMod selected → fallback
                                modified_unique = stripped_unique

                            # Now compute positions exactly like you do in the quant tab
                            def compute_positions_with_label(peptide_list):
                                result = {}
                                for pep in peptide_list:
                                    pep_str = str(pep).strip()
                                    if not pep_str:
                                        continue
                                    pos = find_peptide_position(protein_seq, pep_str)
                                    label = f"{pep_str} (pos: {pos})"
                                    result[pep_str] = (pos, label)
                                return result

                            stripped_positions = compute_positions_with_label(stripped_unique)
                            modified_positions  = compute_positions_with_label(modified_unique)  # ← now only real modified peptides!

                            # Sort by position (same as before)
                            def sort_labels_by_position(pos_dict):
                                sorted_items = sorted(pos_dict.items(), key=lambda x: x[1][0] if x[1][0] != -1 else 999999)
                                return [label for _, (_, label) in sorted_items]

                            stripped_sorted_labels  = sort_labels_by_position(stripped_positions)
                            modified_sorted_labels  = sort_labels_by_position(modified_positions)  # ← clean & correct!
                        
                            # === Session state initialization ===
                            if "view_mode" not in st.session_state:
                                st.session_state.view_mode = "Full Structure (All Peptides)"
                            if "selected_peptide" not in st.session_state:
                                st.session_state.selected_peptide = None

                            st.markdown("#### Peptide View Mode")
                            view_mode = st.radio(
                                "Choose visualization mode:",
                                options=[
                                    "Full Structure (All Peptides)",
                                    "View by Stripped Sequence",
                                    "View by Modified Sequence"
                                ],
                                index=["Full Structure (All Peptides)", "View by Stripped Sequence", "View by Modified Sequence"]
                                    .index(st.session_state.view_mode),
                                key="view_mode_radio_diff",
                                horizontal=True
                            )

                            if view_mode != st.session_state.view_mode:
                                st.session_state.view_mode = view_mode
                                st.session_state.selected_peptide = None
                                st.rerun()

                            selected_peptide = None
                            show_full = (st.session_state.view_mode == "Full Structure (All Peptides)")

                            # === Peptide selector ===
                            if st.session_state.view_mode == "View by Stripped Sequence":
                                peptide_options = stripped_sorted_labels
                                selected_label = st.selectbox(
                                    "Select a stripped peptide to highlight:",
                                    options=peptide_options,
                                    index=0 if st.session_state.selected_peptide not in stripped_positions else peptide_options.index(next(label for pep,(pos,label) in stripped_positions.items()if pep==st.session_state.selected_peptide)),
                                    key="stripped_selector_diff"
                                )
                                selected_peptide = re.split(r"\s*\(pos:", selected_label)[0].strip()
                                st.session_state.selected_peptide = selected_peptide
                                st.info(f"Showing peptide: **{selected_label}**")
                                
                            elif st.session_state.view_mode == "View by Modified Sequence":
                                peptide_options = modified_sorted_labels
                                selected_label = st.selectbox(
                                    "Select a modified peptide to highlight:",
                                    options=peptide_options,
                                    index=0 if st.session_state.selected_peptide not in modified_positions else
                                        peptide_options.index(next(label for pep, (pos, label) in modified_positions.items() if pep == st.session_state.selected_peptide)),
                                    key="modified_selector_diff"
                                )
                                selected_peptide = re.split(r"\s*\(pos:", selected_label)[0].strip()
                                st.session_state.selected_peptide = selected_peptide
                                st.info(f"Showing modified peptide: **{selected_label}**")

                            else:
                                st.success("Showing all mapped peptides on the protein structure")

                            # === BUILD RESIDUE DATA FOR VISUALIZATION ===
                            if show_full:
                                viewer_residue_data = [residue_data[cond] for cond in conditions]
                                viewer_ptm_data = [ptm_data[cond] for cond in conditions]  # Use full PTM data for all viewers
                            else:
                                target_peptide = selected_peptide
                                use_modified = (st.session_state.view_mode == "View by Modified Sequence")
                                filter_col = 'Modified.Sequence' if use_modified else 'Stripped.Sequence'

                                #peptide_rows = selected_df[selected_df[filter_col] == target_peptide]

                                #if peptide_rows.empty:
                                    #st.error(f"Selected peptide not found in current data.")
                                    #st.stop()
                                # Normalize both sides to avoid whitespace/case problems
                                # ─── PUT DEBUG HERE ────────────────────────────────────────────────
                                st.markdown("**─ Debug: Peptide Selection ─**")
                                st.markdown(f"**Selected peptide:** `{target_peptide}`")
                                st.markdown(f"**Filter column used:** `{filter_col}`")
                                target_clean = str(target_peptide).strip()

                                # Create a temporary normalized column (do this once after creating selected_df)
                                if 'Stripped.Sequence_clean' not in selected_df.columns:
                                    selected_df['Stripped.Sequence_clean'] = selected_df['Stripped.Sequence'].astype(str).str.strip()
                                if 'Modified.Sequence_clean' not in selected_df.columns:
                                    selected_df['Modified.Sequence_clean'] = selected_df['Modified.Sequence'].astype(str).str.strip() if 'Modified.Sequence' in selected_df.columns else None

                                # Choose clean column
                                clean_col = 'Stripped.Sequence_clean' if filter_col == 'Stripped.Sequence' else 'Modified.Sequence_clean'

                                peptide_rows = selected_df[selected_df[clean_col] == target_clean]

                                if peptide_rows.empty:
                                    # Fallback: contains (useful when modifications are present)
                                    peptide_rows = selected_df[selected_df[clean_col].str.contains(target_clean, na=False)]

                                    if peptide_rows.empty:
                                        st.error(f"No data found for peptide: **{target_peptide}** (even after fallback search)")
                                        st.stop()
                                    else:
                                        st.warning(f"Using partial match for peptide: **{target_peptide}** ({len(peptide_rows)} rows found)")
                                # Debug matches
                                st.markdown(f"**Exact matches after clean:** {len(peptide_rows)}")
                                if len(peptide_rows) == 0:
                                    st.markdown("**Sample values from clean column (first 10):**")
                                    st.write(selected_df[clean_col].dropna().head(10).tolist())
                                viewer_residue_data = []
                                viewer_ptm_data = []  # Clear & build fresh

                                for condition in conditions:
                                    single_residue_list = [None] * len(protein_seq)
                                    #suffix = "_FC" if visualize_by == "Fold Change" else "_pvalue"
                                    intensity_col = condition

                                    ptm_col_to_use = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None

                                    residues_temp, ptms_temp = map_peptides_to_residues(
                                        peptide_rows,
                                        protein_seq,
                                        intensity_col,
                                        overlap_strategy,
                                        ptm_col=ptm_col_to_use,
                                        apply_tryptic=st.session_state.apply_tryptic
                                    )

                                    # Transform
                                    #if visualize_by == "Fold Change":
                                        #residues_temp = [np.log2(v) if v is not None and np.isfinite(v) and v > 0 else None for v in residues_temp]
                                    #else:
                                        #residues_temp = [-np.log10(v) if v is not None and np.isfinite(v) and v > 0 else None for v in residues_temp]
                                    # Transform only if user requested it
                                    if visualize_by == "Fold Change" and st.session_state.apply_log2:
                                        residues_temp = [np.log2(v) if v is not None and np.isfinite(v) and v > 0 else None for v in residues_temp]
                                    elif visualize_by == "P-value" and st.session_state.apply_neglog10:
                                        residues_temp = [-np.log10(v) if v is not None and np.isfinite(v) and 0 < v <= 1 else None for v in residues_temp]
                                    # else: keep original values (already transformed or user unchecked)
                                    # Copy residues (only if value non-None for this condition)
                                    for i, val in enumerate(residues_temp):
                                        if val is not None:
                                            single_residue_list[i] = val

                                    viewer_residue_data.append(single_residue_list)

                                    # Handle PTMs: build condition_ptm even if intensity missing (PTMs are peptide-intrinsic)
                                    condition_ptm = {}
                                    if st.session_state.ptm_enabled and ptms_temp:
                                        for um, positions in ptms_temp.items():
                                            if um in st.session_state.selected_unimods:
                                                config = st.session_state.ptm_configs.get(um, {})
                                                condition_ptm[um] = {
                                                    'positions': sorted(list(positions)),
                                                    'selected': config.get('selected', True),
                                                    'color': config.get('color', '#3700FF'),
                                                    'label': config.get('label', um)
                                                }

                                    viewer_ptm_data.append(condition_ptm)  # ONLY append once per condition!
                            
                            peptide_atlas_url = f"https://db.systemsbiology.net/sbeams/cgi/PeptideAtlas/GetProtein?atlas_build_id=592&protein_name={base_id}&action=QUERY"
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore PeptideAtlas : </span>'
                                f'<a href="{peptide_atlas_url}" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">Explore Peptides of {base_id} in Peptide Atlas</a>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            # Above "3D Structure Visualizations" header
                            alphafold_url = f"https://alphafold.ebi.ac.uk/search/text/{base_id}"
                            st.markdown(
                                f'<span style="font-size:20px; color:#FFFFFF;">Explore AlphaFold DB : </span>'
                                f'<a href="{alphafold_url}" target="_blank" style="font-size:16px; color:#87CEEB; text-decoration:underline;">View {base_id} in AlphaFold Database</a>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            # ---------- DisProt ---------- (same as multi)
                            disprot_info = get_disprot_info(base_id)

                            # Safely extract disorder percent
                            disorder_percent = disprot_info.get("disorder_percent")
                            disorder_txt = (
                                f"{disorder_percent:.1f} %" if disorder_percent is not None else "unknown"
                            )

                            # Determine if it's an IDP: has DisProt ID + disorder >= 30%
                            is_idp = (
                                disprot_info.get("found", False) and
                                disorder_percent is not None and
                                disorder_percent >= 30.0
                            )

                            # Define the links
                            if disprot_info.get("found", False):
                                disprot_link = f"https://disprot.org/{disprot_info['disprot_id']}"
                                view_text = f"View {disprot_info['disprot_id']} ({base_id}) in DisProt"
                            else:
                                disprot_link = f"https://disprot.org/browse?sort_field=disprot_id&sort_value=asc&page_size=20&page=0&release=current&show_ambiguous=true&show_obsolete=false&acc={base_id}"
                                view_text = f"View {base_id} in DisProt"

                            if is_idp:
                                st.success(f"**{base_id}** is considered an IDP (Disorder content: **{disorder_txt}**)")
                                # Normal clickable link
                                st.markdown(
                                    f'Explore DisProt: <a href="{disprot_link}" target="_blank">{view_text}</a>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                # ---- NOT an IDP or insufficient disorder ----------------
                                if not disprot_info.get("found", False):
                                    st.warning(f"**{base_id}** not found in DisProt.")
                                else:
                                    st.warning(
                                        f"**{base_id}** is **not** considered an IDP "
                                        f"(Disorder content: **{disorder_txt}**)."
                                    )
                                
                                # Blurred, non-clickable link (still shows URL)
                                st.markdown(
                                    f'Explore DisProt: '
                                    f'<span style="color: #888; text-decoration: none;">{view_text}</span> '
                                    f'<span style="font-size: 0.8em; color: #aaa;"></span>',
                                    unsafe_allow_html=True,
                                )
                            
                            # ---- NEW: also normalize the protein-level fallback ----
                            if 'protein_ptms' in locals() and protein_ptms:
                                protein_ptms = normalize_ptm_data(protein_ptms)
                            
                            render_synced_viewers(pdb_str, viewer_residue_data, bg_color, conditions, selected_cmap, selected_not_mapped_color, viewer_ptm_data)
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                            
                            # Calculate coverage for each condition
                            coverages = {}
                            for cond in conditions:
                                coverages[cond] = (sum(1 for v in residue_data[cond] if v is not None) / seq_len * 100) if seq_len > 0 else 0
                            
                            # ======================================================
                            #              LINEAR SEQUENCE VISUALIZATIONS
                            # ======================================================
                            st.subheader("Linear Sequence Visualizations")

                            view_mode = st.session_state.view_mode
                            selected_peptide = st.session_state.selected_peptide

                            # Decide what to show based on 3D selection
                            if view_mode == "Full Structure (All Peptides)" or selected_peptide is None:
                                # Show all peptides — default behavior
                                linear_residue_data = residue_data
                                linear_ptm_data = ptm_data
                                st.info("Linear plots show all mapped peptides.")

                            else:
                                # SHOW ONLY *ONE* PEPTIDE
                                st.success(f"Linear plots show only peptide: **{selected_peptide}**")

                                # Determine whether to filter on stripped or modified sequence
                                filter_col = (
                                    "Modified.Sequence" if view_mode == "View by Modified Sequence"
                                    else "Stripped.Sequence"
                                )

                                peptide_rows = selected_df[selected_df[filter_col] == selected_peptide]

                                if peptide_rows.empty:
                                    st.error(f"Selected peptide '{selected_peptide}' not found in data.")
                                    st.stop()

                                # Build linear data for each condition
                                linear_residue_data = {}
                                linear_ptm_data = {}

                                for cond in conditions:
                                    # List of residues for linear plot
                                    residues_linear = [None] * len(protein_seq)

                                    # Choose if PTM column is active
                                    ptm_col_to_use = 'Modified.Sequence' if st.session_state.ptm_enabled and has_ptm else None

                                    #suffix = "_FC" if visualize_by == "Fold Change" else "_pvalue"
                                    intensity_col = condition

                                    # Map only this peptide
                                    temp_residues, temp_ptms = map_peptides_to_residues(
                                        peptide_rows,
                                        protein_seq,
                                        intensity_col,
                                        overlap_strategy,
                                        ptm_col=ptm_col_to_use,
                                        apply_tryptic=st.session_state.apply_tryptic,
                                        proteotypic_only=st.session_state.proteotypic_only
                                    )

                                    # Transform
                                    #if visualize_by == "Fold Change":
                                       # temp_residues = [np.log2(v) if v is not None and np.isfinite(v) and v > 0 else None for v in temp_residues]
                                    #else:
                                        #temp_residues = [-np.log10(v) if v is not None and np.isfinite(v) and v > 0 else None for v in temp_residues]
                                    # Transform only if user requested it
                                    if visualize_by == "Fold Change" and st.session_state.apply_log2:
                                        residues_temp = [np.log2(v) if v is not None and np.isfinite(v) and v > 0 else None for v in residues_temp]
                                    elif visualize_by == "P-value" and st.session_state.apply_neglog10:
                                        residues_temp = [-np.log10(v) if v is not None and np.isfinite(v) and 0 < v <= 1 else None for v in residues_temp]
                                    # else: keep original values (already transformed or user unchecked)
                                    # Copy mapped residues only
                                    for i, v in enumerate(temp_residues):
                                        if v is not None:
                                            residues_linear[i] = v

                                    linear_residue_data[cond] = residues_linear

                                    # Rebuild PTM data ONLY for this peptide
                                    cond_ptm = {}
                                    if st.session_state.ptm_enabled and temp_ptms:
                                        for um, positions in temp_ptms.items():
                                            if um in st.session_state.selected_unimods:
                                                cfg = st.session_state.ptm_configs.get(um, {})
                                                cond_ptm[um] = {
                                                    "positions": sorted(list(positions)),
                                                    "selected": cfg.get("selected", True),
                                                    "color": cfg.get("color", "#3700FF"),
                                                    "label": cfg.get("label", um)
                                                }
                                    linear_ptm_data[cond] = cond_ptm

                            # --------------------------------------------------------------
                            # RENDER LINEAR PLOTS FOR EACH CONDITION
                            # --------------------------------------------------------------
                            first_condition = conditions[0] if conditions else None
                            for cond in conditions:
                                mapped_count = sum(1 for v in linear_residue_data[cond] if v is not None)
                                coverage_pct = (mapped_count / len(protein_seq) * 100) if len(protein_seq) > 0 else 0
                                short_title = f"{cond} (Coverage: {coverage_pct:.1f}%)"
                                #label_style = "font-size:16px; font-weight:bold; color:#87CEEB;" if cond == first_condition else "font-size:15px; color:#666;"
                                st.markdown(
                                    f'<div style="text-align:center; margin:10px 0 6px 0; {short_title}</div>',
                                    unsafe_allow_html=True
                                )
                                is_first = (cond == first_condition)
                                #ptm_to_pass = linear_ptm_data[cond] if is_first else None
                                render_linear_plot(
                                    linear_residue_data[cond],
                                    short_title,
                                    seq_len,
                                    min_max_logs[cond][0],
                                    min_max_logs[cond][1],
                                    protein_seq,
                                    model_name,
                                    plddt_list,
                                    mean_plddt,
                                    cmap_name=selected_cmap,
                                    not_mapped_color=selected_not_mapped_color,
                                    ptm_data=linear_ptm_data[cond],
                                    show_full_header=is_first,
                                    show_ptm_legend=is_first
                                )
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

                            # Colorbar (adjusted label)
                            st.subheader("Colorbar")
                            overall_vmin = min(min_max_logs[cond][0] for cond in conditions)
                            overall_vmax = max(min_max_logs[cond][1] for cond in conditions)
                            cbar_fig, cbar_ax = plt.subplots(figsize=(8, 0.3))
                            norm = Normalize(vmin=overall_vmin, vmax=overall_vmax)
                            sm = ScalarMappable(cmap=colormaps[selected_cmap], norm=norm)
                            cbar = cbar_fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                            if visualize_by == "Fold Change":
                                cbar.set_label('Log2 Fold Change', fontsize=10)
                            else:
                                cbar.set_label('-Log10 P-value', fontsize=10)
                            cbar.ax.tick_params(labelsize=9)
                            plt.tight_layout()
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=False)
                            buf.seek(0)
                            img_str = base64.b64encode(buf.getvalue()).decode()
                            plt.close(cbar_fig)
                            html_content = f"""
                            <div style="display: flex; justify-content: center; align-items: center; width: 100%; max-width: 600px; margin: 10px auto;">
                                <img src="data:image/png;base64,{img_str}" style="width: 100%; max-width: 500px; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                            </div>
                            """
                            st.components.v1.html(html_content, height=80) 
                              
                            # ─── Interactive Volcano Scatter Plot for selected protein ────────────────────────
                            st.subheader("Peptide Volcano Plot")

                            col_left, col_right = st.columns(2)

                            with col_left:
                                
                                # ─── 1. Find all FC and p-value columns ────────────────────────────────
                                fc_cols = [c for c in fc_columns if c in selected_df.columns]
                                pval_cols = [c for c in pval_columns if c in selected_df.columns]

                                if not fc_cols or not pval_cols:
                                    st.warning("Cannot build volcano plot: missing FC or p-value columns.")
                                else:
                                    # ─── 2. Melt FC values ────────────────────────────────
                                    df_fc = selected_df.melt(
                                        id_vars=['Stripped.Sequence', 'Modified.Sequence'],
                                        value_vars=fc_cols,
                                        var_name='FC_Condition',
                                        value_name='FC_value'
                                    )

                                    # ─── 3. Melt p-value values ────────────────────────────────
                                    df_pval = selected_df.melt(
                                        id_vars=['Stripped.Sequence', 'Modified.Sequence'],
                                        value_vars=pval_cols,
                                        var_name='Pval_Condition',
                                        value_name='P_value'
                                    )

                                    # ─── 4. Create clean condition key for joining (remove suffixes)
                                    df_fc['Cond_key'] = df_fc['FC_Condition'].str.replace(r'(_log2FC|_FC|_logFC)$', '', regex=True)
                                    df_pval['Cond_key'] = df_pval['Pval_Condition'].str.replace(r'(_pvalue|_padj|_adjp|_FDR)$', '', regex=True)

                                    # ─── 5. Merge on peptide + condition key
                                    plot_df = df_fc.merge(
                                        df_pval,
                                        on=['Stripped.Sequence', 'Modified.Sequence', 'Cond_key'],
                                        how='inner'
                                    )

                                    if plot_df.empty:
                                        st.warning("No matching peptide-condition pairs found for volcano plot.")
                                    else:
                                        # ─── 6. Apply transformations (using linked values) ────────────────────────────────
                                        if apply_log2:
                                            plot_df['FC_value'] = np.log2(plot_df['FC_value'].clip(lower=1e-10))
                                        if apply_neglog10:
                                            plot_df['P_value'] = -np.log10(plot_df['P_value'].clip(lower=1e-300, upper=1))

                                        # ─── 7. Clean condition name for legend ────────────────────────────────
                                        plot_df['Condition'] = plot_df['Cond_key']

                                        # ─── 8. Axis labels ────────────────────────────────
                                        x_label = "log₂ Fold Change" if apply_log2 else "Fold Change"
                                        y_label = "-log₁₀ P-value" if apply_neglog10 else "P-value"
# Apply transformations using session state values
                                        if st.session_state.apply_log2:
                                            plot_df['FC_value'] = np.log2(plot_df['FC_value'].clip(lower=1e-10))
                                            y_axis_label = "log₂ Fold Change"
                                        else:
                                            y_axis_label = "Fold Change"

                                        if st.session_state.apply_neglog10:
                                            plot_df['P_value'] = -np.log10(plot_df['P_value'].clip(lower=1e-300))
                                            x_axis_label = "-log₁₀ P-value"
                                        else:
                                            x_axis_label = "P-value"
                                        # ─── 9. Plotly figure ────────────────────────────────
                                        fig = px.scatter(
                                            plot_df,
                                            x='P_value',
                                            y='FC_value',
                                            color='Condition',
                                            hover_name='Stripped.Sequence',
                                            hover_data={
                                                'Modified.Sequence': True,
                                                'Condition': True,
                                                'FC_value': ':.3f',
                                                'P_value': ':.3e'
                                            },
                                            color_discrete_sequence=px.colors.qualitative.Plotly,
                                            opacity=0.8,
                                            title=f"Volcano Plot – {selected_protein} (n={len(plot_df)} peptides)",
                                            labels={'FC_value': y_label, 'P_value': x_label}
                                        )

                                        fig.update_traces(marker=dict(size=9, line=dict(width=0.8, color='Black')))
                                        fig.update_layout(
                                            legend=dict(
                                                title="Condition",
                                                orientation="v",
                                                yanchor="top",
                                                y=0.99,
                                                xanchor="left",
                                                x=1.02,
                                                bgcolor="rgba(255,255,255,0.8)"
                                            ),
                                            hovermode="closest",
                                            clickmode='event+select',
                                            height=600
                                        )

                                        col_left.plotly_chart(fig, use_container_width=True)
                            # ────────────────────────────────────────────────────────────────
                            #                   SAMPLE-LEVEL PCA (Peptide Intensities)
                            # ────────────────────────────────────────────────────────────────
                            with col_right:
                                st.info("Sample Seration PCA based on selected peptides.")
                                st.markdown("""
                                This PCA uses **log₂-transformed peptide intensities** from the currently selected protein  
                                (or combined isoforms). It helps you see whether samples from different groups separate well.
                                """)

                                if len(sample_cols) < 1:
                                    st.warning("Too few samples to make PCA meaningful (need ≥1).")
                                else:
                                    # Let user control a few options
                                    pca_use_log2 = st.checkbox("Apply log₂ transformation", value=True, key="pca_log2_checkbox")
                                    pca_impute_method = st.selectbox("Imputation method for missing values", 
                                                                    ["row median (recommended)", "zero", "skip peptide"], 
                                                                    index=0, key="pca_impute_select")

                                    if st.button("Compute & Show PCA", type="primary", key="btn_compute_pca"):
                                        with st.spinner("Preparing data and running PCA..."):

                                            # ─── 1. Create wide matrix: peptides × samples ───────────────────────
                                            intensity_wide = selected_df.pivot_table(
                                                index='Stripped.Sequence',
                                                values=sample_cols,
                                                aggfunc='mean'          # in case duplicate peptides
                                            ).reset_index(drop=False)

                                            # Keep only peptides with at least some data
                                            intensity_wide = intensity_wide[intensity_wide[sample_cols].notna().sum(axis=1) >= 2]

                                            if intensity_wide.empty:
                                                st.error("No usable peptides with intensities after filtering.")
                                            else:
                                                # ─── 2. Log₂ transform (if chosen) ────────────────────────────────
                                                intensity_mat = intensity_wide[sample_cols].to_numpy()
                                                if pca_use_log2:
                                                    intensity_mat = np.log2(np.clip(intensity_mat, 1e-10, None))

                                                # ─── 3. Imputation ────────────────────────────────────────────────
                                                if pca_impute_method == "row median (recommended)":
                                                    row_med = np.nanmedian(intensity_mat, axis=1, keepdims=True)
                                                    intensity_mat = np.where(np.isnan(intensity_mat), row_med, intensity_mat)
                                                elif pca_impute_method == "zero":
                                                    intensity_mat = np.nan_to_num(intensity_mat, nan=0.0)
                                                else:  # skip peptide = remove rows with any NaN
                                                    valid_rows = ~np.any(np.isnan(intensity_mat), axis=1)
                                                    intensity_mat = intensity_mat[valid_rows]
                                                    intensity_wide = intensity_wide[valid_rows]

                                                if intensity_mat.shape[0] == 0:
                                                    st.error("No peptides remain after imputation/filtering.")
                                                else:
                                                    # ─── 4. Transpose → samples × peptides ────────────────────────
                                                    X = intensity_mat.T

                                                    # Sample names and groups
                                                    samples_used = intensity_wide.columns[1:] if 'Stripped.Sequence' in intensity_wide.columns else sample_cols
                                                    groups_used = [meta_map.get(s, "Unknown") for s in samples_used]

                                                    # ─── 5. Scale & PCA ──────────

                                                    scaler = StandardScaler()
                                                    X_scaled = scaler.fit_transform(X)
                                                    n_features = X.shape[1]
                                                    if n_features == 0:
                                                        st.error("No features available for PCA.")
                                                    elif n_features == 1:
                                                        # Fallback: 1D plot — use the scaled intensity directly
                                                        pc_values = X_scaled[:, 0]

                                                        pca_df = pd.DataFrame({
                                                            'Value': pc_values,
                                                            'Sample': samples_used,
                                                            'Group': groups_used
                                                        })

                                                        fig = px.strip(
                                                            pca_df,
                                                            x='Group',
                                                            y='Value',
                                                            color='Group',
                                                                      # different markers per group
                                                            hover_name='Sample',
                                                            title=f"Single Peptide View ({len(samples_used)} samples, 1 peptide)",
                                                            labels={'Value': 'Scaled Intensity (only dimension)'},
                                                            height=550
                                                        )
                                                        fig.update_traces(marker=dict(size=12,opacity=0.9))
                                                        st.plotly_chart(fig, use_container_width=True)
                                                        st.info("Only one peptide available → showing 1D distribution instead of 2D PCA.")
                                                    else:
                                                        pca_model = PCA(n_components=2)
                                                        pcs = pca_model.fit_transform(X_scaled)

                                                        var_expl = pca_model.explained_variance_ratio_ * 100
                                                        
                                                        pca_df = pd.DataFrame({
                                                            'PC1': pcs[:, 0],
                                                            'PC2': pcs[:, 1],
                                                            'Sample': samples_used,
                                                            'Group': groups_used
                                                        })

                                                        # ─── 6. Plot ──────────────────────────────────────────────────
                                                        fig = px.scatter(
                                                            pca_df,
                                                            x='PC1', y='PC2',
                                                            color='Group',
                                                            symbol='Group',
                                                            hover_name='Sample',
                                                            title=f"PCA – Peptide Intensities ({len(samples_used)} samples, {len(intensity_mat)} peptides)",
                                                            labels={
                                                                'PC1': f"PC1 ({var_expl[0]:.1f}%)",
                                                                'PC2': f"PC2 ({var_expl[1]:.1f}%)"
                                                            },
                                                            opacity=0.9,
                                                            color_discrete_sequence=px.colors.qualitative.Bold,
                                                            symbol_sequence = ['circle', 'square', 'diamond', 'triangle-up','cross', 'x'],
                                                            height=550
                                                        )

                                                        fig.update_traces(
                                                            marker=dict(size=10, line=dict(width=1))
                                                        )

                                                        fig.update_layout(
                                                            legend=dict(
                                                                title="Group",
                                                                yanchor="top", y=0.99,
                                                                xanchor="left", x=1.02
                                                            ),
                                                            hovermode="closest"
                                                        )

                                                        st.plotly_chart(fig, use_container_width=True)

                                                        # Quick interpretation text
                                                        if var_expl[0] + var_expl[1] < 40:
                                                            st.caption("⚠ Low explained variance — PCA may not capture main patterns well.")
                                                        elif any(g == "Unknown" for g in groups_used):
                                                            st.caption("Note: Some samples could not be mapped to groups.")
                            # ===============================================
                            # 💾 Download & Reset Buttons
                            # ===============================================
                        
                            col_btn1, col_btn2 = st.columns(2)
                        
                            with col_btn1:
                                if st.button("Prepare Download (ZIP)", use_container_width=True):
                                    zip_buffer = create_download_zip(
                                        selected_protein, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len,
                                        selected_cmap, selected_not_mapped_color,
                                        ptm_data if st.session_state.ptm_enabled else None,
                                        selected_df if st.session_state.ptm_enabled else None,
                                        protein_seq if st.session_state.ptm_enabled else None,
                                        apply_tryptic=st.session_state.apply_tryptic,
                                        metadata=metadata
                                    )
                                    st.download_button(
                                        label="Download ZIP",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"{selected_protein}_files.zip",
                                        mime="application/zip"
                                    )
                        
                            with col_btn2:
                                if st.button("Reset & Re-Process", use_container_width=True):
                                    st.session_state.clear()
                                    st.session_state.conditions_confirmed = False
                                    st.session_state.processed = False
                                    st.session_state.selected_residue = None
                                    st.session_state.ptm_enabled = False
                                    st.session_state.ptm_configs = {}
                                    st.session_state.apply_tryptic = False
                                    st.session_state.pdb_source = 'AlphaFold'
                                    st.session_state.uploaded_pdb = None
                                    st.session_state.all_unimods = []
                                    st.session_state.selected_unimods = []
                                    st.rerun()
