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
from matplotlib.cm import ScalarMappable
import zipfile
from streamlit.components.v1 import html
import json
from Bio.PDB import PDBParser
import re
import time
import json

# Set wide layout
st.set_page_config(layout="wide", page_title="Peptide3D Mapper")

# --- Helper Functions ---
def z_score(intensities):
    log_int = np.log10(intensities + 1)
    mean_log = np.mean(log_int)
    std_log = np.std(log_int)
    return np.zeros_like(log_int) if std_log == 0 else (log_int - mean_log) / std_log

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

def map_peptides_to_residues(df, protein_seq, intensity_col, overlap_strategy='merge', ptm_col=None, apply_tryptic=False):
    seq_len = len(protein_seq)
    residue_vals = [None] * seq_len
    ptm_positions = {}
    peptides = df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
    z_scores = z_score(peptides[intensity_col])
    
    for idx, row in df.iterrows():
        pep = row['Stripped.Sequence']
        # Find all occurrences of the peptide
        matches = list(re.finditer(re.escape(pep), protein_seq))
        if not matches:
            st.write(f"Peptide {pep} not found in protein sequence")  # Debug
            continue
        
        valid_found = False
        for match in matches:
            start = match.start()
            # Apply tryptic rule if enabled
            if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                continue
            valid_found = True
            end = start + len(pep)
            # Intensity mapping
            pep_mean_intensity = peptides[peptides['Stripped.Sequence'] == pep][intensity_col].values[0]
            z_val = z_scores[peptides['Stripped.Sequence'] == pep].values[0]
            for i in range(start, end):
                if residue_vals[i] is None:
                    residue_vals[i] = [z_val]
                else:
                    residue_vals[i].append(z_val)
            # PTM mapping
            if ptm_col and ptm_col in row:
                ptm_seq = row[ptm_col]
                if pd.notna(ptm_seq) and '(UniMod:' in ptm_seq:
                    cleaned_pep, mods = clean_and_find_mods(ptm_seq)
                    #st.write(f"Row {idx}: Peptide={pep}, PTM Seq={ptm_seq}, Mods={mods}, Cleaned={cleaned_pep}")  # Debug
                    if cleaned_pep != pep:
                        st.write(f"Mismatch: Stripped.Sequence={pep}, Cleaned PTM={cleaned_pep}")
                        continue
                    for rel_pos, unismod in mods:
                        abs_pos = start + rel_pos
                        if 0 <= abs_pos < seq_len:
                            if unismod not in ptm_positions:
                                ptm_positions[unismod] = set()
                            ptm_positions[unismod].add(abs_pos)
                        else:
                            st.write(f"Invalid PTM position {abs_pos} for UniMod:{unismod} in peptide {pep}")
        
        if not valid_found:
            st.write(f"No valid tryptic position for peptide {pep}")
    
    # Resolve overlaps for intensities
    for i in range(seq_len):
        if residue_vals[i]:
            if overlap_strategy == 'merge':
                residue_vals[i] = np.mean(residue_vals[i])
            elif overlap_strategy == 'highest':
                residue_vals[i] = np.max(residue_vals[i])
            elif overlap_strategy in ['none', 'last']:
                residue_vals[i] = residue_vals[i][-1]
            else:
                residue_vals[i] = np.mean(residue_vals[i])
        else:
            residue_vals[i] = None
    
    # Convert PTM sets to lists
    for k in ptm_positions:
        ptm_positions[k] = sorted(list(ptm_positions[k]))
    
    st.write(f"Final PTM positions for {intensity_col}: {ptm_positions}")  # Debug
    return residue_vals, ptm_positions

def generate_colormap(residue_vals, cmap_name='autumn', not_mapped_color='#d3d3d3'):
    cmap = colormaps[cmap_name]
    vals = [v for v in residue_vals if v is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
    hex_colors = []
    for val in residue_vals:
        if val is None:
            hex_colors.append(not_mapped_color)
        else:
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            rgb = cmap(norm)[:3]
            hex_colors.append(mcolors.rgb2hex(rgb))
    return hex_colors, vmin, vmax

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

def render_linear_plot(residue_vals, title, seq_len, vmin, vmax, protein_seq, model_name, plddt_list, mean_plddt,
                       cmap_name='viridis', not_mapped_color='#d3d3d3', highlight_residues=[], ptm_data=None):
    hex_colors, _, _ = generate_colormap(residue_vals, cmap_name, not_mapped_color)
    mapped = [i for i, v in enumerate(residue_vals) if v is not None]
    mapped_js = str(mapped)
    pixel_per_res = max(2, min(12, 1200 / seq_len))
    total_width = pixel_per_res * seq_len
    bar_height = 35
    label_offset = bar_height + 15
    total_height = label_offset + 20
    bars = ""
    for i in range(seq_len):
        x = i * pixel_per_res
        color = hex_colors[i]
        width = pixel_per_res
        aa = protein_seq[i] if i < len(protein_seq) else 'X'
        z_val = f"{residue_vals[i]:.2f}" if residue_vals[i] is not None else "N/A"
        tooltip = f"Pos {i+1} ({aa}): Z-Score={z_val}"
        is_mapped = i in mapped
        mapped_attr = 'True' if is_mapped else 'False'
        bars += f'<rect x="{x}" y="0" width="{width}" height="{bar_height}" fill="{color}" '
        bars += f'stroke="#666" stroke-width="0.5" data-pos="{i}" data-mapped="{mapped_attr}" title="{tooltip}" />'
    # Add PTM vertical lines if provided (enhanced visibility)
    ptm_lines = ""
    #st.write(f"PTM data for linear plot '{title}': {ptm_data}")  # Debug to confirm UniMod:35 is present
    if ptm_data:
        for unismod, info in ptm_data.items():
            if info['selected']:
                color = info['color']
                for pos in info['positions']:
                    x = pos * pixel_per_res + (pixel_per_res / 2)
                    label = info['label']
                    #st.write(f"Adding PTM line for UniMod:{unismod} at pos {pos+1}, x={x}, color={color}")  # Debug position/calc
                    ptm_lines += f'<line x1="{x}" y1="0" x2="{x}" y2="{bar_height}" stroke="{color}" stroke-width="3" title="{label} at Pos {pos+1}" />'
                    # Add circle for better visibility
                    ptm_lines += f'<circle cx="{x}" cy="{bar_height / 2}" r="4" fill="{color}" stroke="black" stroke-width="1" title="{label} at Pos {pos+1}" />'
    label_step = max(1, int(50 / pixel_per_res))
    labels = ""
    for i in range(0, seq_len, label_step):
        x = i * pixel_per_res + (pixel_per_res / 2)
        labels += f'<text x="{x}" y="{label_offset}" font-size="12" text-anchor="middle" fill="#333">{i+1}</text>'
    mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
    title_html = f'<div style="text-align:center; font-size:18px; margin-bottom:5px; font-weight:bold; color:#2a2a2a;">{title}<br><span style="font-size:12px; color:#666;">{model_name} | Mean pLDDT: {mean_plddt_display}</span></div>'
    svg = f'<svg width="{total_width + 20}" height="{total_height + 20}" style="overflow:visible; background:#fff; border:1px solid #ddd; border-radius:6px; padding:10px;">{bars}{ptm_lines}{labels}</svg>'
    container_html = f'<div style="overflow-x:auto; max-width:100%; margin:10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{svg}</div>'
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
    }});
    </script>
    """
    html_output = title_html + container_html + js
    st.components.v1.html(html_output, height=150)
    return None, None

def add_ptm_spheres(viewer_idx, ptm_data, condition_name, view):
    if ptm_data:
        for unismod, info in ptm_data.items():
            if info['selected']:
                color = info['color']
                st.write(f"Adding PTM sphere for UniMod:{unismod} in {condition_name}, color={color}, positions={info['positions']}")
                for pos in info['positions']:
                    resi_str = str(pos + 1)
                    spec = {
                        'center': {'resi': resi_str, 'atom': 'CA', 'chain': 'A'},
                        'radius': 2.0,  # Larger for visibility
                        'color': color,
                        'alpha': 1.0,
                        'zOffset': 10.0  # Push forward
                    }
                    try:
                        view.addSphere(spec, viewer=(0, viewer_idx))
                        # Fallback label
                        view.addLabel(f"UniMod:{unismod}", {'fontSize': 12, 'fontColor': color, 'backgroundColor': 'white', 'backgroundOpacity': 0.5},
                                      {'resi': resi_str, 'chain': 'A'}, viewer=(0, viewer_idx))
                        #st.write(f"Successfully added sphere/label at resi {resi_str} with color {color}")
                    except Exception as e:
                        st.error(f"Failed to add sphere for UniMod:{unismod} at {resi_str}: {e}")
        view.render()

def render_synced_viewers(pdb_str, residue_vals1, residue_vals2, bg_color, title1, title2, cmap_name='autumn', not_mapped_color='#d3d3d3', ptm_data1=None, ptm_data2=None):
    hex_colors1, vmin1, vmax1 = generate_colormap(residue_vals1, cmap_name, not_mapped_color)
    hex_colors2, vmin2, vmax2 = generate_colormap(residue_vals2, cmap_name, not_mapped_color)
    residues_js1 = json.dumps([i for i, v in enumerate(residue_vals1) if v is not None])
    residues_js2 = json.dumps([i for i, v in enumerate(residue_vals2) if v is not None])
    
    view = py3Dmol.view(width='95vw', height='400px', viewergrid=(1,2), linked=True)
    view.addModel(pdb_str, 'pdb', viewer=(0,0))
    view.addModel(pdb_str, 'pdb', viewer=(0,1))
    bg_color_map = {'white': '#FFFFFF', 'black': '#000000', 'darkgrey': '#4A4A4A'}
    bg_color_hex = bg_color_map.get(bg_color.lower(), '#000000')
    view.setBackgroundColor(bg_color_hex, viewer=(0,0))
    view.setBackgroundColor(bg_color_hex, viewer=(0,1))
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}}, viewer=(0,0))
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}}, viewer=(0,1))
    
    for i, c in enumerate(hex_colors1):
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': c}}, viewer=(0,0))
    for i, c in enumerate(hex_colors2):
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': c}}, viewer=(0,1))
    
    add_ptm_spheres(0, ptm_data1, title1, view)
    add_ptm_spheres(1, ptm_data2, title2, view)
    
    view.zoomTo(viewer=(0,0))
    view.zoomTo(viewer=(0,1))
    view.render()
    
    hover_js = """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        function tryInitViewers(retryCount = 5, delay = 500) {
            try {
                const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
                if (viewerElems.length < 2) {
                    if (retryCount > 0) {
                        console.warn(`Not enough viewer elements found (${viewerElems.length}/2), retrying in ${delay}ms`);
                        setTimeout(() => tryInitViewers(retryCount - 1, delay), delay);
                    } else {
                        console.error("Failed to find enough viewer elements after retries");
                    }
                    return;
                }
                const viewer0 = viewerElems[0].querySelector('div > canvas').parentElement.viewer;
                const viewer1 = viewerElems[1].querySelector('div > canvas').parentElement.viewer;
                const residues1 = JSON.parse('{residues_js1}');
                const residues2 = JSON.parse('{residues_js2}');
                
                const container = viewerElems[0].parentElement;
                const divider = document.createElement('div');
                divider.id = 'viewerDivider';
                divider.style.position = 'absolute';
                divider.style.height = '400px';
                divider.style.width = '20px';
                divider.style.backgroundColor = '#666';
                divider.style.left = '50%';
                divider.style.top = '0';
                divider.style.zIndex = '100';
                divider.style.transform = 'translateX(-10px)';
                container.appendChild(divider);
                
                setTimeout(() => {
                    const rect0 = viewerElems[0].getBoundingClientRect();
                    const rect1 = viewerElems[1].getBoundingClientRect();
                    const midX = (rect0.right + rect1.left) / 2;
                    divider.style.left = midX + 'px';
                    divider.style.transform = 'translateX(-10px)';
                    console.log("Divider position set to:", midX);
                }, 500);
                
                function handlePick(viewer, residues) {
                    return function(atom, event) {
                        if (!atom) return;
                        const resi = parseInt(atom.resi, 10) - 1;
                        if (residues.includes(resi)) {
                            window.parent.postMessage({ type: "SELECT_RESIDUE", residue: resi }, "*");
                            console.log("3D click sent for pos:", resi);
                        }
                    }
                }
                viewer0.setClickable({}, true, handlePick(viewer0, residues1));
                viewer1.setClickable({}, true, handlePick(viewer1, residues2));
            } catch(e) {
                console.error("3Dmol pick init error", e);
                if (retryCount > 0) {
                    setTimeout(() => tryInitViewers(retryCount - 1, delay), delay);
                }
            }
        }
        tryInitViewers();
    });
    </script>
    """
    hover_js = hover_js.replace('{residues_js1}', residues_js1).replace('{residues_js2}', residues_js2)
    
    listener_js = """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        let previous_selected = null;
        const observer = new MutationObserver(() => {
            const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
            if (viewerElems.length >= 2) {
                observer.disconnect();
                console.log("3D viewers detected.");
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        
        window.addEventListener("message", (event) => {
            if (event.data && event.data.type === "SELECT_RESIDUE") {
                const residue = event.data.residue;
                console.log("Received SELECT_RESIDUE for pos:", residue);
                try {
                    const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
                    if (viewerElems.length < 2) {
                        console.warn("Viewers not ready - retrying in 100ms");
                        setTimeout(() => window.dispatchEvent(new MessageEvent("message", { data: event.data })), 100);
                        return;
                    }
                    const viewer0 = viewerElems[0].querySelector('div > canvas').parentElement.viewer;
                    const viewer1 = viewerElems[1].querySelector('div > canvas').parentElement.viewer;
                    
                    if (previous_selected !== null) {
                        const prev_span = document.querySelector(`.aa[data-pos="${previous_selected}"]`);
                        if (prev_span) {
                            prev_span.style.backgroundColor = "";
                            prev_span.style.fontWeight = "";
                        }
                        const prev_bars = document.querySelectorAll(`rect[data-pos="${previous_selected}"]`);
                        prev_bars.forEach(bar => {
                            bar.style.stroke = "none";
                            bar.style.strokeWidth = "0";
                        });
                        viewer0.removeAllShapes();
                        viewer1.removeAllShapes();
                        viewer0.render();
                        viewer1.render();
                    }
                    
                    const span = document.querySelector(`.aa[data-pos="${residue}"]`);
                    if (span) {
                        span.style.backgroundColor = "yellow";
                        span.style.fontWeight = "bold";
                    }
                    const bars = document.querySelectorAll(`rect[data-pos="${residue}"]`);
                    bars.forEach(bar => {
                        bar.style.stroke = "red";
                        bar.style.strokeWidth = "2";
                    });
                    
                    const resi_str = (residue + 1).toString();
                    const spec = {center: {resi: resi_str, atom: 'CA'}, radius: 5.0, color: 'red', alpha: 0.6};
                    viewer0.addSphere(spec);
                    viewer1.addSphere(spec);
                    viewer0.center({resi: resi_str, atom: 'CA'});
                    viewer1.center({resi: resi_str, atom: 'CA'});
                    viewer0.render();
                    viewer1.render();
                    previous_selected = residue;
                } catch (e) {
                    console.error("Error adding 3D highlight:", e);
                }
            }
        });
    });
    </script>
    """
    
    html = view._make_html()
    st.markdown(f"#### {title1} (Left) | {title2} (Right)")
    st.components.v1.html(html + hover_js, height=420)
    st.components.v1.html(listener_js, height=0)

def create_download_zip(protein_of_interest, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len, cmap_name='autumn', not_mapped_color='#d3d3d3', ptm_data=None, selected_df=None, protein_seq=None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(f"{protein_of_interest}_protein.pdb", pdb_str)
        for condition in conditions:
            peptide_csv = peptide_data[condition].to_csv(index=False)
            zipf.writestr(f"{protein_of_interest}_{condition}_peptides.csv", peptide_csv)
        
        cmap = colormaps[cmap_name]
        for condition in conditions:
            pml_content = f"load {protein_of_interest}_protein.pdb\nhide everything\nshow cartoon\ncolor gray90, all\nzoom\n"
            min_log, max_log = min_max_logs[condition]
            for i in range(seq_len):
                if residue_data[condition][i] is not None:
                    norm = (residue_data[condition][i] - min_log) / (max_log - min_log) if max_log > min_log else 0.5
                    color_hex = mcolors.rgb2hex(cmap(norm)[:3])
                    pml_content += f"color {color_hex}, resi {i+1}\n"
            # Add PTM spheres in PyMOL script
            if ptm_data and ptm_data[condition]:
                for unismod, info in ptm_data[condition].items():
                    if info['selected']:
                        color_hex = info['color']
                        for pos in info['positions']:
                            resi = pos + 1
                            pml_content += f"pseudoatom ptm_{unismod}_{resi}, resi {resi} and name CA\n"
                            pml_content += f"show spheres, ptm_{unismod}_{resi}\n"
                            pml_content += f"set sphere_scale, 5.0, ptm_{unismod}_{resi}\n"
                            pml_content += f"color {color_hex}, ptm_{unismod}_{resi}\n"
            zipf.writestr(f"{protein_of_interest}_{condition}_pymol_script.pml", pml_content)
        
        for condition in conditions:
            fig_width = min(25, max(10, seq_len / 20))
            fig, ax = plt.subplots(figsize=(fig_width, 1), dpi=600)
            ax.add_patch(patches.Rectangle((0, 0), seq_len, 1, facecolor=not_mapped_color, edgecolor='none'))
            min_log, max_log = min_max_logs[condition]
            for i in range(seq_len):
                if residue_data[condition][i] is not None:
                    norm = (residue_data[condition][i] - min_log) / (max_log - min_log) if max_log > min_log else 0.5
                    ax.add_patch(patches.Rectangle((i, 0), 1, 1, facecolor=cmap(norm)[:3], edgecolor='none'))
            ax.set_xlim(0, seq_len)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel(f'Amino Acid Position ({condition})', fontsize=30)
            ax.tick_params(axis='x', labelsize=15)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='jpeg', dpi=600, bbox_inches='tight')
            img_buffer.seek(0)
            zipf.writestr(f"{protein_of_interest}_{condition}_linear.jpeg", img_buffer.read())
            plt.close(fig)
        
        # Add PTM positions CSV (inspired by first code)
        if ptm_data and selected_df is not None and protein_seq is not None:
            ptm_rows = []
            for idx, row in selected_df.iterrows():
                protein = row['Protein.Group']
                stripped = row['Stripped.Sequence']
                ptm = row['PTM'] if 'PTM' in row and pd.notna(row['PTM']) else ''
                control = row[conditions[list(conditions.keys())[0]]] if list(conditions.keys())[0] in row else 'NA'
                disease = row[conditions[list(conditions.keys())[1]]] if list(conditions.keys())[1] in row else 'NA'
                
                if ptm and '(UniMod:' in ptm:
                    cleaned, mods = clean_and_find_mods(ptm)
                    if cleaned != stripped:
                        st.write(f"Warning: Cleaned PTM sequence '{cleaned}' does not match Stripped.Sequence '{stripped}' for protein {protein}")
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'Mismatch', 'NA', 'NA', 'NA'])
                        continue
                    matches = list(re.finditer(re.escape(cleaned), protein_seq))
                    valid_found = False
                    for match in matches:
                        start = match.start()
                        if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                            continue
                        valid_found = True
                        peptide_start = start + 1
                        peptide_end = start + len(cleaned)
                        for mod_pos, unismod in mods:
                            full_pos = start + mod_pos + 1  # 1-based for output
                            ptm_rows.append([protein, stripped, control, disease, ptm, full_pos, unismod, peptide_start, peptide_end])
                    if not valid_found:
                        for mod_pos, unismod in mods:
                            ptm_rows.append([protein, stripped, control, disease, ptm, 'No valid tryptic position', unismod, 'NA', 'NA'])
                else:
                    matches = list(re.finditer(re.escape(stripped), protein_seq))
                    valid_found = False
                    for match in matches:
                        start = match.start()
                        if apply_tryptic and start > 0 and protein_seq[start - 1] not in 'KR':
                            continue
                        valid_found = True
                        peptide_start = start + 1
                        peptide_end = start + len(stripped)
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'NA', 'NA', peptide_start, peptide_end])
                    if not valid_found:
                        ptm_rows.append([protein, stripped, control, disease, ptm, 'No valid tryptic position', 'NA', 'NA', 'NA'])
            
            ptm_df = pd.DataFrame(ptm_rows, columns=['Protein.Group', 'Stripped.Sequence', 'Control_Intensity', 'Disease_Intensity', 'PTM', 'PTM_position', 'UniMod_Type', 'Peptide_Start', 'Peptide_End'])
            ptm_csv = ptm_df.to_csv(index=False)
            zipf.writestr(f"{protein_of_interest}_modification_positions.csv", ptm_csv)
    
    zip_buffer.seek(0)
    return zip_buffer

def format_sequence_for_display(seq, residue_data, condition1_name, condition2_name, line_len=80, group=20):
    mapped_positions = set(i for i, v in enumerate(residue_data[condition1_name]) if v is not None)
    mapped_positions.update(i for i, v in enumerate(residue_data[condition2_name]) if v is not None)
    mapped_js = json.dumps(list(mapped_positions))
    lines = []
    seq_len = len(seq)
    for start in range(0, seq_len, line_len):
        end = min(start + line_len, seq_len)
        segment = seq[start:end]
        num_line = "<div style='display: flex; align-items: flex-start; font-family: monospace; font-size: 10px; color: #888;'>"
        seq_line = "<div style='display: flex; align-items: flex-start; font-family: monospace; font-size: 12px; line-height: 1.5;'>"
        if start > 0:
            num_line += f"<span style='position: relative;'><span style='margin-right: {group - 1}ch;'>{start}</span><span style='position: absolute; left: 0; right: 0; top: 100%; height: 10px; border-left: 1px dashed #888;'></span></span>"
        for i in range(0, len(segment), group):
            pos = start + i + 1
            num_span = f"<span style='margin-right: {group - 1}ch;'>{pos}</span>" if i + group <= len(segment) else f"<span>{pos}</span>"
            num_line += f"<span style='position: relative;'>{num_span}<span style='position: absolute; left: 0; right: 0; top: 100%; height: 10px; border-left: 1px dashed #888;'></span></span>"
            seq_segment = segment[i:i + group]
            segment_html = ""
            for j, aa in enumerate(seq_segment):
                abs_pos = start + i + j
                style = "cursor:pointer;color:blue;" if abs_pos in mapped_positions else "color:gray;"
                segment_html += f"<span class='aa' data-pos='{abs_pos}' style='{style}'>{aa}</span>"
            seq_line += f"<span>{segment_html}</span>"
        num_line += "</div>"
        seq_line += f"<span style='margin-left: auto;'>{end}</span></div>"
        lines.append(num_line + seq_line)
    seq_html = "<div id='seq-panel' style='padding:10px; background:#fafafa; border-radius:6px; border:1px solid #ddd;'>" + "".join(lines) + "</div>"
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
    <p style='text-align: justify; font-size: 16px; color: #4a4a4a;'>
   The Peptide3D Mapper is a web-based tool that visualizes peptide intensity data from proteomics experiments on AlphaFold 3D protein structures.
   Upload peptide CSV and FASTA files to compare conditions (e.g., control vs. disease) using z-score intensity scales.
   Explore residue-level differences in interactive 3D and linear sequence views, with customizable colors and exportable outputs.
    </p>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'conditions_confirmed' not in st.session_state:
    st.session_state.conditions_confirmed = False
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'selected_residue' not in st.session_state:
    st.session_state.selected_residue = None
if 'ptm_enabled' not in st.session_state:
    st.session_state.ptm_enabled = False
if 'ptm_configs' not in st.session_state:
    st.session_state.ptm_configs = {}
if 'apply_tryptic' not in st.session_state:
    st.session_state.apply_tryptic = False

# File upload
csv_file = st.file_uploader("Upload Peptide CSV", type=["csv"], help="CSV with Protein.Group, Stripped.Sequence, PTM, and intensity columns")
fasta_file = st.file_uploader("Upload FASTA", type=["fasta"], help="FASTA with matching UniProt IDs")

if csv_file and fasta_file:
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    fasta_str = fasta_file.getvalue().decode("utf-8")
    fasta_handle = io.StringIO(fasta_str)
    seq_records = list(SeqIO.parse(fasta_handle, "fasta"))
    if not seq_records:
        st.error("No sequences found in FASTA file.")
        st.stop()

    # Extract UniMod IDs
    if 'PTM' in df.columns:
        all_unimods = set()
        for ptm_seq in df['PTM'].dropna():
            if '(UniMod:' in ptm_seq:
                matches = re.finditer(r'UniMod:(\d+)', ptm_seq, re.IGNORECASE)
                for match in matches:
                    all_unimods.add(match.group(1))
        st.session_state.all_unimods = sorted(list(all_unimods)) if all_unimods else []
    else:
        st.session_state.all_unimods = []

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
    else:
        st.session_state.selected_unimods = []

    # PTM and tryptic options
    has_ptm = bool(st.session_state.all_unimods)
    ptm_checkbox_disabled = not has_ptm
    st.session_state.ptm_enabled = st.checkbox("Enable PTM Annotation", disabled=ptm_checkbox_disabled, value=False if ptm_checkbox_disabled else st.session_state.ptm_enabled)
    st.session_state.apply_tryptic = st.checkbox("Apply Tryptic Rule (K/R cleavage)", value=st.session_state.apply_tryptic)

    # Condition setup
    intensity_cols = [c for c in df.columns if 'intensity' in c.lower()]
    if len(intensity_cols) != 2:
        st.error(f"Expected exactly 2 intensity columns, found: {intensity_cols}")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        condition1_name = st.text_input("Name for Condition 1", value="Control")
        condition1_col = st.selectbox("Map Condition 1 to Column", intensity_cols, index=0)
    with col2:
        condition2_name = st.text_input("Name for Condition 2", value="Disease")
        condition2_col = st.selectbox("Map Condition 2 to Column", intensity_cols, index=1)
    
    if condition1_col == condition2_col:
        st.error("Intensity columns must be different.")
        st.stop()
    
    if st.button("Confirm Conditions", use_container_width=True):
        st.session_state.conditions_confirmed = True
        st.session_state.processed = False
        st.rerun()

    if st.session_state.conditions_confirmed:
        with st.container():
            st.info("✅ Conditions confirmed. Now select protein and options.")
            protein_options = sorted(df['Protein.Group'].unique())
            selected_protein = st.selectbox("Select Protein", protein_options)
            col3, col4 = st.columns([1, 1])
            with col3:
                combine_isoforms = st.selectbox("Combine Isoforms?", ["yes", "no"])
            with col4:
                overlap_strategy = st.selectbox("Overlap Strategy", ["none", "merge", "highest", "last"])
            if st.button("Process Protein", use_container_width=True):
                st.session_state.processed = True
                st.rerun()

        if st.session_state.processed:
            with st.container():
                st.info("🔄 Processing... (This may take a moment for PDB fetch or upload.)")
                base_id = selected_protein.split('-')[0]
                protein_seq = None
                for rec in seq_records:
                    parts = rec.id.split('|')
                    uniprot_candidate = None
                    if len(parts) >= 2:
                        if parts[0] in ['sp', 'tr']:
                            uniprot_candidate = parts[1]
                        else:
                            uniprot_candidate = parts[0]
                    else:
                        uniprot_candidate = rec.id.split()[0]
                    if uniprot_candidate == base_id:
                        protein_seq = str(rec.seq)
                        matched_header = rec.id
                        break
                if protein_seq is None:
                    st.info(f"No direct FASTA header match for {base_id}. Attempting peptide-based matching...")
                    peptides_unique = df[df['Protein.Group'] == selected_protein]['Stripped.Sequence'].dropna().unique().tolist()
                    if len(peptides_unique) == 0:
                        peptides_unique = df[df['Protein.Group'].str.contains(base_id)]['Stripped.Sequence'].dropna().unique().tolist()
                    best_count = -1
                    best_rec = None
                    for rec in seq_records:
                        rec_seq = str(rec.seq)
                        count = 0
                        for pep in peptides_unique:
                            if pep and pep in rec_seq:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_rec = rec
                    if best_count > 0 and best_rec is not None:
                        protein_seq = str(best_rec.seq)
                        matched_header = best_rec.id
                        st.info(f"Selected FASTA entry {matched_header} with {best_count} peptides matched.")
                    else:
                        if len(seq_records) == 1:
                            protein_seq = str(seq_records[0].seq)
                            matched_header = seq_records[0].id
                            st.info(f"No peptide matches found; using the single FASTA entry {matched_header}.")
                        else:
                            st.error("Protein sequence could not be unambiguously detected from FASTA (no header match and no peptide overlap).")
                            st.stop()

                seq_len = len(protein_seq)
                isoforms = df[df['Protein.Group'].str.contains(selected_protein + r'(?:-\d+)?$', regex=True)]['Protein.Group'].unique()
                if len(isoforms) > 1 and combine_isoforms == "no":
                    selected_groups = st.multiselect("Select Isoforms", options=list(isoforms), default=list(isoforms))
                else:
                    selected_groups = list(isoforms)
                if not selected_groups:
                    st.error("No isoforms selected.")
                    st.stop()
                
                selected_df = df[df['Protein.Group'].isin(selected_groups)]
                conditions = {condition1_name: condition1_col, condition2_name: condition2_col}
                peptide_data = {}
                residue_data = {condition1_name: [None] * seq_len, condition2_name: [None] * seq_len}
                ptm_data = {condition1_name: {}, condition2_name: {}}
                min_max_logs = {}
                ptm_col = 'PTM' if st.session_state.ptm_enabled and has_ptm else None
                
                for condition, intensity_col in conditions.items():
                    residues, ptms = map_peptides_to_residues(
                        selected_df, protein_seq, intensity_col, overlap_strategy, 
                        ptm_col, apply_tryptic=st.session_state.apply_tryptic
                    )
                    residue_data[condition] = residues
                    if st.session_state.ptm_enabled and st.session_state.selected_unimods:
                        ptm_data[condition] = {um: pos for um, pos in ptms.items() if um in st.session_state.selected_unimods}
                    else:
                        ptm_data[condition] = ptms
                    covered = [v for v in residues if v is not None]
                    if not covered:
                        st.error(f"No peptides mapped for {condition}.")
                        st.stop()
                    min_max_logs[condition] = (min(covered), max(covered))
                    peptides = selected_df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
                    peptide_data[condition] = peptides

                # PTM configuration
             # PTM configuration with hyperlinks
                if st.session_state.ptm_enabled and st.session_state.selected_unimods:
                    st.subheader("PTM Configuration")
                    st.write(f"Selected UniMods: {st.session_state.selected_unimods}")
                    if not st.session_state.selected_unimods:
                        st.warning("No selected UniMod annotations for this protein.")
                    else:
                        if 'ptm_configs' not in st.session_state or set(st.session_state.ptm_configs.keys()) != set(st.session_state.selected_unimods):
                            st.session_state.ptm_configs = {um: {'selected': True, 'label': f"UniMod:{um}", 'color': "#3700FF"} for um in st.session_state.selected_unimods}
                        # Debug to confirm selected UniMods
                        st.write(f"Rendering PTM config for UniMods: {st.session_state.selected_unimods}")
                        for um in st.session_state.selected_unimods:
                            col_ptm1, col_ptm2, col_ptm3 = st.columns([1, 1, 1])
                            with col_ptm1:
                                # Add hyperlink with explicit styling
                                st.markdown(
                                    f'<a href="https://www.unimod.org/modifications_view.php?editid1={um}" target="_blank" style="color: #2b8cff; text-decoration: underline; font-weight: bold;">UniMod:{um}</a>',
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
                            # Debug to confirm hyperlink generation
                            #print(f"Generated hyperlink for UniMod:{um}: https://www.unimod.org/modifications_view.php?editid1={um}")
                        for cond in ptm_data:
                            for um in list(ptm_data[cond].keys()):
                                if um in st.session_state.ptm_configs:
                                    ptm_data[cond][um] = {
                                        'positions': ptm_data[cond][um],
                                        'selected': st.session_state.ptm_configs[um]['selected'],
                                        'label': st.session_state.ptm_configs[um]['label'],
                                        'color': st.session_state.ptm_configs[um]['color']
                                    }
                                else:
                                    del ptm_data[cond][um]
                else:
                    ptm_data = {condition1_name: None, condition2_name: None}
                
                st.subheader("Detected Sequence")
                st.markdown(f"**FASTA header:** {matched_header}")
                seq_html = format_sequence_for_display(protein_seq, residue_data, condition1_name, condition2_name, line_len=150, group=20)
                copy_html = sequence_copy_component(protein_seq)
                st.components.v1.html(copy_html + seq_html, height=320)
                
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
                
                st.success(f"Loaded AlphaFold v6 structure for {base_id} ({len(pdb_str)} bytes)")
                plddt_list, model_name, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)
                mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
                st.info(f"**Mean pLDDT:** {mean_plddt_display} (Overall Confidence)")
                
                bg_color = st.selectbox("Background Color", ["black", "white", "darkgrey"], index=0)
                cmap_options = ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                selected_cmap = st.selectbox("Select Color Gradient", cmap_options, index=0)
                selected_not_mapped_color = st.color_picker("Select Not Mapped Color", "#d3d3d3")
                
                with st.container():
                    st.subheader("3D Structure Visualizations")
                    render_synced_viewers(pdb_str, residue_data[condition1_name], residue_data[condition2_name], bg_color, condition1_name, condition2_name, selected_cmap, selected_not_mapped_color, ptm_data[condition1_name], ptm_data[condition2_name])
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.subheader("Linear Sequence Visualizations")
                    st.markdown(f"#### {condition1_name}")
                    render_linear_plot(residue_data[condition1_name], condition1_name, seq_len,
                                       min_max_logs[condition1_name][0], min_max_logs[condition1_name][1], protein_seq, model_name, plddt_list, mean_plddt, cmap_name=selected_cmap, not_mapped_color=selected_not_mapped_color, ptm_data=ptm_data[condition1_name])
                    st.markdown(f"#### {condition2_name}")
                    render_linear_plot(residue_data[condition2_name], condition2_name, seq_len,
                                       min_max_logs[condition2_name][0], min_max_logs[condition2_name][1], protein_seq, model_name, plddt_list, mean_plddt, cmap_name=selected_cmap, not_mapped_color=selected_not_mapped_color, ptm_data=ptm_data[condition2_name])
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.subheader("Colorbar")
                    overall_vmin = min(min_max_logs[condition1_name][0], min_max_logs[condition2_name][0])
                    overall_vmax = max(min_max_logs[condition1_name][1], min_max_logs[condition2_name][1])
                    fig, ax = plt.subplots(figsize=(4, 0.2))
                    norm = Normalize(vmin=overall_vmin, vmax=overall_vmax)
                    sm = ScalarMappable(cmap=colormaps[selected_cmap], norm=norm)
                    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                    cbar.set_label('Z-Score Intensity', fontsize=8)
                    cbar.ax.tick_params(labelsize=6)
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
                    buf.seek(0)
                    plt.close(fig)
                    html_content = f"""
                    <div id="colorbar-container" style="width: 100%; height: auto; max-width: 500px;">
                        <img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="width: 100%; height: auto;">
                    </div>
                    """
                    st.components.v1.html(html_content, height=100)
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Download Files (ZIP)", use_container_width=True):
                        zip_buffer = create_download_zip(selected_protein, pdb_str, peptide_data, residue_data, conditions.keys(), min_max_logs, seq_len, selected_cmap, selected_not_mapped_color, ptm_data if st.session_state.ptm_enabled else None, selected_df if st.session_state.ptm_enabled else None, protein_seq if st.session_state.ptm_enabled else None)
                        st.download_button(
                            label="Download ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=f"{selected_protein}_files.zip",
                            mime="application/zip"
                        )
                with col_btn2:
                    if st.button("Reset & Re-Process", use_container_width=True):
                        st.session_state.processed = False
                        st.session_state.selected_residue = None
                        st.session_state.ptm_configs = {}
                        st.rerun()
