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
# Set wide layout at the top
st.set_page_config(layout="wide", page_title="Peptide3D Mapper")

# --- Helper functions ---
def z_score(intensities):
    log_int = np.log10(intensities + 1)
    mean_log = np.mean(log_int)
    std_log = np.std(log_int)
    if std_log == 0:
        return np.zeros_like(log_int)
    else:
        return (log_int - mean_log) / std_log

def map_peptides_to_residues(df, protein_seq, intensity_col, overlap_strategy='merge'):
    seq_len = len(protein_seq)
    residue_vals = [None] * seq_len
    peptides = df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
    z_scores = z_score(peptides[intensity_col])
    for idx, row in peptides.iterrows():
        pep = row['Stripped.Sequence']
        start = protein_seq.find(pep)
        if start == -1: 
            # peptide not found in this protein
            continue
        end = start + len(pep)
        for i in range(start, end):
            z_val = z_scores[idx]
            if residue_vals[i] is None:
                residue_vals[i] = [z_val]
            else:
                residue_vals[i].append(z_val)
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
    return residue_vals

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
    """
    Parses PDB string to extract per-residue pLDDT (from B-factors of CA atoms)
    and model name from header. Returns plddt_list (None if missing) and model_name.
    """
    # Parse PDB from string
    parser = PDBParser(QUIET=True)
    pdb_io = io.StringIO(pdb_str)  # Changed 'io' to 'pdb_io' to avoid conflict
    
    structure = parser.get_structure('model', pdb_io)
    
    plddt_list = [None] * len(protein_seq)
    model_name = "Unknown Model"  # Fallback
    
    # Extract model name from header (e.g., REMARK lines or first MODEL)
    if 'HEADER' in pdb_str:
        header_match = re.search(r'HEADER\s+\S+\s+\S+\s+(.+?)\s+\d{2}', pdb_str, re.IGNORECASE)
        if header_match:
            model_name = header_match.group(1).strip()
    else:
        # Fallback to URL-derived name
        model_name = "AF-" + base_id + "-F1-model_v6" if 'base_id' in globals() else "AlphaFold Model"
    
    # Extract pLDDT from B-factors (assuming single model/chain; adjust if multi-chain)
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    res_id = residue.id[1] - 1  # 1-based to 0-based index
                    if 0 <= res_id < len(protein_seq):
                        b_factor = residue['CA'].get_bfactor()
                        plddt_list[res_id] = b_factor  # pLDDT is the B-factor value
    
    # Compute mean pLDDT (ignoring None)
    valid_plddt = [v for v in plddt_list if v is not None]
    mean_plddt = np.mean(valid_plddt) if valid_plddt else None
    
    return plddt_list, model_name, mean_plddt
# --- Linear plot with highlight ---
# --- Linear plot with highlight ---
# --- Linear plot with highlight ---
def render_linear_plot(residue_vals, title, seq_len, vmin, vmax, protein_seq, plddt_list, model_name, mean_plddt,
                       cmap_name='viridis', not_mapped_color='#d3d3d3', highlight_residues=[]):
    # Generate hex colors based on z-scores with AlphaFold-like mapping
    hex_colors, _, _ = generate_colormap(residue_vals, cmap_name, not_mapped_color)
    mapped = [i for i, v in enumerate(residue_vals) if v is not None]
    
    # Adjust pixel per residue for wider layout (min 2px, max 12px, based on sequence length)
    pixel_per_res = max(2, min(12, 1200 / seq_len))
    total_width = pixel_per_res * seq_len
    bar_height = 35  # Increased for better visibility and AlphaFold-like style
    label_offset = bar_height + 15
    total_height = label_offset + 20  # Space for labels and padding
    
    # SVG content
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
        # AlphaFold-like bar with subtle border
        bars += f'<rect x="{x}" y="0" width="{width}" height="{bar_height}" fill="{color}" '
        bars += f'stroke="#666" stroke-width="0.5" data-pos="{i}" data-mapped="{mapped_attr}" title="{tooltip}" />'
    
    # Add position labels below bars, styled like AlphaFold
    label_step = max(1, int(50 / pixel_per_res))
    labels = ""
    for i in range(0, seq_len, label_step):
        x = i * pixel_per_res + (pixel_per_res / 2)
        labels += f'<text x="{x}" y="{label_offset}" font-size="12" text-anchor="middle" fill="#333">{i+1}</text>'
    
    # Title with model and mean pLDDT
    mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
    title_html = f'<div style="text-align:center; font-size:18px; margin-bottom:5px; font-weight:bold; color:#2a2a2a;">{title}<br><span style="font-size:12px; color:#666;">{model_name} | Mean pLDDT: {mean_plddt_display}</span></div>'

    # Assemble SVG
    svg = f'<svg width="{total_width + 20}" height="{total_height + 20}" style="overflow:visible; background:#fff; border:1px solid #ddd; border-radius:6px; padding:10px;">{bars}{labels}</svg>'
    container_html = f'<div style="overflow-x:auto; max-width:100%; margin:10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{svg}</div>'

    # JavaScript for interactivity with mapped_js interpolation
    js = """
    <script>
    const mapped = {mapped_js};
    const rects = document.querySelectorAll('rect[data-pos]');
    rects.forEach(el => {
      const pos = parseInt(el.getAttribute('data-pos'));
      const isMapped = mapped.includes(pos);
      if (isMapped) {
        el.style.cursor = 'pointer';
        el.addEventListener('click', e => {
          const pos = parseInt(e.target.getAttribute('data-pos'));
          window.parent.postMessage({ type: 'SELECT_RESIDUE', residue: pos }, '*');
        });
      }
      el.addEventListener('mouseover', e => {
        if (e.target.title) {
          e.target.style.opacity = '0.7';
          e.target.style.strokeWidth = '1';
        }
      });
      el.addEventListener('mouseout', e => {
        e.target.style.opacity = '1';
        e.target.style.strokeWidth = '0.5';
      });
    });
    </script>
    """

    html_output = title_html + container_html + js
    st.components.v1.html(html_output, height=150)  # Adjusted height to focus on bars and labels
    
    return None, None  # No pmin, pmax needed since pLDDT line is removed # Return pmin and pmax for colorbar use # Return pmin and pmax for colorbar use # Increased height for line plot
# --- JavaScript listener to update Streamlit state when residue is clicked ---
listener_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  let previous_selected = null;
  // Use MutationObserver to wait for 3D viewers to load
  const observer = new MutationObserver(() => {
    const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
    if (viewerElems.length >= 2) {
      observer.disconnect(); // Stop observing once loaded
      console.log("3D viewers detected.");
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });

  window.addEventListener("message", (event) => {
    if (event.data && event.data.type === "SELECT_RESIDUE") {
      const residue = event.data.residue; // 0-based
      console.log("Received SELECT_RESIDUE for pos:", residue);
      // Clear previous highlights
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
        // Remove shapes from 3D viewers
        try {
          const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
          if (viewerElems.length >= 2) {
            const viewer0 = viewerElems[0].querySelector('div > canvas').parentElement.viewer;
            const viewer1 = viewerElems[1].querySelector('div > canvas').parentElement.viewer;
            viewer0.removeAllShapes();
            viewer1.removeAllShapes();
            viewer0.render();
            viewer1.render();
          }
        } catch (e) {
          console.error("Error clearing 3D shapes:", e);
        }
      }
      // Apply new highlights
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
      // Add sphere and center in 3D viewers (1-based resi)
      try {
        const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
        if (viewerElems.length >= 2) {
          const viewer0 = viewerElems[0].querySelector('div > canvas').parentElement.viewer;
          const viewer1 = viewerElems[1].querySelector('div > canvas').parentElement.viewer;
          const resi_str = (residue + 1).toString();
          const spec = {center: {resi: resi_str, atom: 'CA'}, radius: 5.0, color: 'red', alpha: 0.6};
          viewer0.addSphere(spec);
          viewer1.addSphere(spec);
          viewer0.center({resi: resi_str, atom: 'CA'});
          viewer1.center({resi: resi_str, atom: 'CA'});
          viewer0.render();
          viewer1.render();
        }
      } catch (e) {
        console.error("Error adding 3D highlight:", e);
      }
      previous_selected = residue;
    }
  });
});
</script>
"""     
# --- Interactive 3D view with linked highlighting ---
def render_synced_viewers(pdb_str, residue_vals1, residue_vals2, bg_color, title1, title2, cmap_name='autumn', not_mapped_color='#d3d3d3'):
    hex_colors1, vmin1, vmax1 = generate_colormap(residue_vals1, cmap_name, not_mapped_color)
    hex_colors2, vmin2, vmax2 = generate_colormap(residue_vals2, cmap_name, not_mapped_color)
    # Build JS arrays for linear plot highlights
    residues_js1 = str([i for i, v in enumerate(residue_vals1) if v is not None])
    residues_js2 = str([i for i, v in enumerate(residue_vals2) if v is not None])
    # Render py3Dmol viewers
    view = py3Dmol.view(width='95vw', height='400px', viewergrid=(1,2), linked=True)
    view.addModel(pdb_str, 'pdb', viewer=(0,0))
    view.addModel(pdb_str, 'pdb', viewer=(0,1))
    # Map bg_color to hex values, defaulting to white
    bg_color_map = {'white': '#FFFFFF', 'black': '#000000', 'darkgrey': '#4A4A4A'}
    bg_color_hex = bg_color_map.get(bg_color.lower(), '#FFFFFF')  # Default to white if invalid
    view.setBackgroundColor(bg_color_hex, viewer=(0,0))  # First viewer uses selected bg_color
    view.setBackgroundColor(bg_color_hex, viewer=(0,1))  # Second viewer uses selected bg_color
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}}, viewer=(0,0))
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}}, viewer=(0,1))
    # Apply residue colors
    for i, c in enumerate(hex_colors1):
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': c}}, viewer=(0,0))
    for i, c in enumerate(hex_colors2):
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': c}}, viewer=(0,1))
    view.zoomTo(viewer=(0,0))
    view.zoomTo(viewer=(0,1))
    # Updated hover_js with corrected syntax
    hover_js = """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        try {
            const viewerElems = document.getElementsByClassName("viewer_3Dmoljs");
            if (viewerElems.length < 2) {
                console.error("Not enough viewer elements found");
                return;
            }
            const viewer0 = viewerElems[0].querySelector('div > canvas').parentElement.viewer;
            const viewer1 = viewerElems[1].querySelector('div > canvas').parentElement.viewer;
            const residues1 = {residues_js1};
            const residues2 = {residues_js2};

            // Add vertical bar between viewers
            const container = viewerElems[0].parentElement; // Use the common parent
            const divider = document.createElement('div');
            divider.id = 'viewerDivider'; // Add ID for debugging
            divider.style.position = 'absolute';
            divider.style.height = '400px'; // Match viewer height
            divider.style.width = '20px'; // Increased width as requested
            divider.style.backgroundColor = '#666'; // Gray color
            divider.style.left = '50%'; // Start at center, adjust below
            divider.style.top = '0';
            divider.style.zIndex = '100'; // Higher zIndex to ensure visibility
            divider.style.transform = 'translateX(-10px)'; // Center the 20px width
            container.appendChild(divider);

            // Debug: Log and adjust position after render
            setTimeout(() => {
                const rect0 = viewerElems[0].getBoundingClientRect();
                const rect1 = viewerElems[1].getBoundingClientRect();
                const midX = (rect0.right + rect1.left) / 2;
                divider.style.left = midX + 'px';
                divider.style.transform = 'translateX(-10px)'; // Half of 20px to center
                console.log("Divider position set to:", midX, "Container width:", container.offsetWidth);
            }, 100); // Delay to ensure viewers are rendered

            function handlePick(viewer, residues) {
                return function(atom, event) {
                    if (!atom) return;
                    const resi = parseInt(atom.resi, 10) - 1; // Convert to 0-based
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
        }
    });
    </script>
    """
    html = view._make_html()
    st.markdown(f"#### {title1} (Left) | {title2} (Right)")
    st.components.v1.html(html + hover_js, height=420)
    st.components.v1.html(listener_js, height=0)
def create_download_zip(protein_of_interest, pdb_str, peptide_data, residue_data, conditions, min_max_logs, seq_len, cmap_name='autumn', not_mapped_color='#d3d3d3'):
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
    zip_buffer.seek(0)
    return zip_buffer

# --- NEW: Sequence display helper (AlphaFold-like) ---
def format_sequence_for_display(seq, residue_data, condition1_name, condition2_name, line_len=80, group=20):
    """Interactive AlphaFold-style sequence with clickable residues"""
    # Compute union of mapped positions for both conditions
    mapped_positions = set(i for i, v in enumerate(residue_data[condition1_name]) if v is not None)
    mapped_positions.update(i for i, v in enumerate(residue_data[condition2_name]) if v is not None)
    mapped_js = str(list(mapped_positions))

    lines = []
    seq_len = len(seq)
    for start in range(0, seq_len, line_len):
        end = min(start + line_len, seq_len)
        segment = seq[start:end]
        
        # Create numbering line with vertical alignment markers, including end number at start
        num_line = "<div style='display: flex; align-items: flex-start; font-family: monospace; font-size: 10px; color: #888;'>"
        seq_line = "<div style='display: flex; align-items: flex-start; font-family: monospace; font-size: 12px; line-height: 1.5;'>"
        
        # Add end number at the start of the line
        if start > 0:
            num_line += f"<span style='position: relative;'><span style='margin-right: {group - 1}ch;'>{start}</span><span style='position: absolute; left: 0; right: 0; top: 100%; height: 10px; border-left: 1px dashed #888;'></span></span>"
        
        # Add numbers every 'group' residues with dashed lines
        for i in range(0, len(segment), group):
            pos = start + i + 1
            num_span = f"<span style='margin-right: {group - 1}ch;'>{pos}</span>" if i + group <= len(segment) else f"<span>{pos}</span>"
            num_line += f"<span style='position: relative;'>{num_span}<span style='position: absolute; left: 0; right: 0; top: 100%; height: 10px; border-left: 1px dashed #888;'></span></span>"
            
            # Add sequence segment with clickable residues
            seq_segment = segment[i:i + group]
            segment_html = ""
            for j, aa in enumerate(seq_segment):
                abs_pos = start + i + j
                style = "cursor:pointer;color:blue;" if abs_pos in mapped_positions else "color:gray;"
                segment_html += f"<span class='aa' data-pos='{abs_pos}' style='{style}'>{aa}</span>"
            seq_line += f"<span>{segment_html}</span>"
        
        num_line += "</div>"
        seq_line += f"<span style='margin-left: auto;'>{end}</span></div>"
        
        # Combine numbering and sequence
        lines.append(num_line + seq_line)

    seq_html = "<div id='seq-panel' style='padding:10px; background:#fafafa; border-radius:6px; border:1px solid #ddd;'>" + "".join(lines) + "</div>"

    js = """
    <script>
    const mapped = {mapped_js};
    document.querySelectorAll('.aa').forEach(el => {
        const pos = parseInt(el.getAttribute('data-pos'));
        if (mapped.includes(pos)) {
            el.style.cursor = 'pointer';
            el.addEventListener('click', e => {
                window.parent.postMessage({ type: 'SELECT_RESIDUE', residue: pos }, '*');
                console.log("Sequence click sent for pos:", pos);
            });
        }
    });
    </script>
    """
    return seq_html + js
def sequence_copy_component(seq):
    """Return HTML that shows a Copy button for sequence and a hidden pre with sequence text to copy using JS."""
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

# Streamlit App UI top banner (unchanged)
html_content = """
<div style="position: relative; width: 100%; overflow: hidden; background-color: #1a1a2e; padding: 20px 0;">
    <h1 id="animated-title" style="font-family: 'Arial', sans-serif; font-size: 48px; color: #e94560; margin: 0; text-align: relative; 
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

# --- Session state defaults ---
if 'conditions_confirmed' not in st.session_state:
    st.session_state.conditions_confirmed = False
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'selected_residue' not in st.session_state:
    st.session_state.selected_residue = None

# File upload widgets
csv_file = st.file_uploader("Upload Peptide CSV", type=["csv"], help="CSV with Protein.Group, Stripped.Sequence, and intensity columns")
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

    # detect intensity columns (expecting 2)
    intensity_cols = [c for c in df.columns if 'intensity' in c.lower()]
    if len(intensity_cols) != 2:
        st.error(f"Expected exactly 2 intensity columns, found: {intensity_cols}")
        st.stop()
    col1, col2 = st.columns([1, 1])
    with col1:
        condition1_name = st.text_input("Name for Condition 1", value="Condition 1")
        condition1_col = st.selectbox("Map Condition 1 to Column", intensity_cols, index=0)
    with col2:
        condition2_name = st.text_input("Name for Condition 2", value="Condition 2")
        condition2_col = st.selectbox("Map Condition 2 to Column", intensity_cols, index=1)
    if condition1_col == condition2_col:
        st.error("Intensity columns must be different.")
        st.stop()
    if st.button("Confirm Conditions", use_container_width=True):
        st.session_state.conditions_confirmed = True
        st.session_state.processed = False
        st.rerun()

    # Once conditions are confirmed, allow protein selection and processing
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

                # --- Try direct ID match in FASTA headers first (UniProt style like sp|P00533|EGFR)
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

                # --- NEW: If no direct ID match, attempt best FASTA record by peptide overlap
                if protein_seq is None:
                    st.info(f"No direct FASTA header match for {base_id}. Attempting peptide-based matching...")
                    # collect unique peptides from the selected protein group (before isoform filtering)
                    peptides_unique = df[df['Protein.Group'] == selected_protein]['Stripped.Sequence'].dropna().unique().tolist()
                    if len(peptides_unique) == 0:
                        # fallback: use all peptides for this protein group (possible isoforms)
                        peptides_unique = df[df['Protein.Group'].str.contains(base_id)]['Stripped.Sequence'].dropna().unique().tolist()

                    best_count = -1
                    best_rec = None
                    for rec in seq_records:
                        rec_seq = str(rec.seq)
                        count = 0
                        # count how many peptides are found inside this rec_seq
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
                        # if still not found, choose the first FASTA entry if only one exists
                        if len(seq_records) == 1:
                            protein_seq = str(seq_records[0].seq)
                            matched_header = seq_records[0].id
                            st.info(f"No peptide matches found; using the single FASTA entry {matched_header}.")
                        else:
                            st.error("Protein sequence could not be unambiguously detected from FASTA (no header match and no peptide overlap).")
                            st.stop()

                seq_len = len(protein_seq)
                # isoform selection and mapping
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
                min_max_logs = {}
                for condition, intensity_col in conditions.items():
                    residues = map_peptides_to_residues(selected_df, protein_seq, intensity_col, overlap_strategy)
                    residue_data[condition] = residues
                    covered = [v for v in residues if v is not None]
                    if not covered:
                        st.error(f"No peptides mapped for {condition}.")
                        st.stop()
                    min_max_logs[condition] = (min(covered), max(covered))
                    peptides = selected_df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
                    peptide_data[condition] = peptides
                # show detected sequence panel (AlphaFold-like)
                st.subheader("Detected Sequence")
                st.markdown(f"**FASTA header:** {matched_header}")
                seq_html = format_sequence_for_display(protein_seq, residue_data, condition1_name, condition2_name, line_len=150,group=20)
                copy_html = sequence_copy_component(protein_seq)
                st.components.v1.html(copy_html + seq_html, height=320)

                # Fetch AlphaFold structure
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
                        st.error(f"Failed to fetch PDB for {base_id}: {str(e)}. Please try again later.")
                        st.stop()
                st.success(f"Loaded AlphaFold v6 structure for {base_id} ({len(pdb_str)} bytes)")
                plddt_list, model_name, mean_plddt = extract_plddt_and_model(pdb_str, protein_seq)
                mean_plddt_display = f"{mean_plddt:.1f}" if mean_plddt is not None else "N/A"
                st.info(f"**Mean pLDDT:** {mean_plddt_display} (Overall Confidence)")
                # Compute mapped positions for pLDDT (optional: only show for intensity-mapped residues)
                mapped_positions = set(i for cond in residue_data.values() for i, v in enumerate(cond) if v is not None)
                # viewer and plotting options
                bg_color = st.selectbox("Background Color", ["white", "black", "darkgrey"], index=0)
                cmap_options = ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                selected_cmap = st.selectbox("Select Color Gradient", cmap_options, index=0)
                selected_not_mapped_color = st.color_picker("Select Not Mapped Color", "#d3d3d3")
                with st.container():
                    st.subheader("3D Structure Visualizations")
                    render_synced_viewers(pdb_str, residue_data[condition1_name], residue_data[condition2_name], bg_color, condition1_name, condition2_name, selected_cmap, selected_not_mapped_color)
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.subheader("Linear Sequence Visualizations")
                    st.markdown(f"#### {condition1_name}")
                    pmin,pmax=render_linear_plot(residue_data[condition1_name], condition1_name, seq_len,
                                       min_max_logs[condition1_name][0], min_max_logs[condition1_name][1],protein_seq, plddt_list,model_name,mean_plddt,cmap_name=selected_cmap, not_mapped_color=selected_not_mapped_color)
                    st.markdown(f"#### {condition2_name}")
                    pmin,pmax=render_linear_plot(residue_data[condition2_name], condition2_name, seq_len,
                                       min_max_logs[condition2_name][0], min_max_logs[condition2_name][1], protein_seq,plddt_list,model_name,mean_plddt,cmap_name=selected_cmap, not_mapped_color=selected_not_mapped_color)
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
                    # After existing colorbar
                    if any(plddt_list):
                        fig, ax = plt.subplots(figsize=(4, 0.2))
                        p_norm = Normalize(vmin=pmin, vmax=pmax)
                        sm_p = ScalarMappable(cmap=colormaps['viridis'], norm=p_norm)
                        cbar_p = plt.colorbar(sm_p, cax=ax, orientation='horizontal')
                        cbar_p.set_label('pLDDT Confidence', fontsize=10)
                        # ... save and display as before
                    st.components.v1.html(html_content, height=100)
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Download Files (ZIP)", use_container_width=True):
                        zip_buffer = create_download_zip(selected_protein, pdb_str, peptide_data, residue_data, conditions.keys(), min_max_logs, seq_len, selected_cmap, selected_not_mapped_color)
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
                        st.rerun()
