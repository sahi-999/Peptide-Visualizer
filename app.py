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
from matplotlib.cm import ScalarMappable
import zipfile
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import base64

# Set wide layout at the top
st.set_page_config(layout="wide", page_title="Peptide3D Mapper")

def z_score(intensities):
    log_int = np.log10(intensities + 1)
    mean_log = np.mean(log_int)
    std_log = np.std(log_int)
    if std_log == 0:
        return np.zeros_like(log_int)
    else:
        return (log_int - mean_log) / std_log

def map_peptides_to_residues(df, protein_seq, intensity_col, overlap_strategy='merge'):
    if 'Stripped.Sequence' not in df.columns or 'Protein.Group' not in df.columns:
        raise ValueError("CSV must have 'Stripped.Sequence' and 'Protein.Group' columns.")
    
    seq_len = len(protein_seq)
    residue_vals = [None] * seq_len

    peptides = df.groupby('Stripped.Sequence')[intensity_col].mean().reset_index()
    z_scores = z_score(peptides[intensity_col])

    for idx, row in peptides.iterrows():
        pep = row['Stripped.Sequence']
        start = protein_seq.find(pep)
        if start == -1:
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

def render_synced_viewers(pdb_str, residue_vals1, residue_vals2, bg_color, title1, title2, cmap_name='autumn', not_mapped_color='#d3d3d3'):
    hex_colors1, vmin1, vmax1 = generate_colormap(residue_vals1, cmap_name, not_mapped_color)
    hex_colors2, vmin2, vmax2 = generate_colormap(residue_vals2, cmap_name, not_mapped_color)
    
    view = py3Dmol.view(width='95vw', height='400px', viewergrid=(1,2), linked=True)
    
    view.addModel(pdb_str, 'pdb', viewer=(0,0))
    view.addModel(pdb_str, 'pdb', viewer=(0,1))
    
    view.setBackgroundColor(bg_color, viewer=(0,0))
    view.setBackgroundColor(bg_color, viewer=(0,1))
    
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}}, viewer=(0,0))
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}}, viewer=(0,1))
    
    for i, c in enumerate(hex_colors1):
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': c}}, viewer=(0,0))
    
    for i, c in enumerate(hex_colors2):
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': c}}, viewer=(0,1))
    
    view.zoomTo(viewer=(0,0))
    view.zoomTo(viewer=(0,1))
    
    # Customize the HTML to add space between viewers
    html = view._make_html()
    html = html.replace('<div class="viewer_3Dmoljs" style="',
                        '<div class="viewer_3Dmoljs" style="width: 50%; float: left; margin-right: 6%;')
    html = html.replace('</div><div class="viewer_3Dmoljs" style="',
                        '</div><div style="clear: both;"></div><div class="viewer_3Dmoljs" style="width: 48%; float: left; margin-right: 0;')
    
    st.markdown(f"#### {title1} (Left) | {title2} (Right)")
    st.components.v1.html(html, height=400)

def render_linear_plot(residue_vals, title, seq_len, vmin, vmax, cmap_name='autumn', not_mapped_color='#d3d3d3'):
    fig_width = min(50, max(20, seq_len * 0.15))
    fig_height = 3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    cmap = colormaps[cmap_name]
    ax.add_patch(patches.Rectangle((0, 0), seq_len, 1, facecolor=not_mapped_color, edgecolor='none'))
    for i in range(seq_len):
        if residue_vals[i] is not None:
            norm = (residue_vals[i] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            ax.add_patch(patches.Rectangle((i, 0), 1, 1, facecolor=cmap(norm)[:3], edgecolor='none'))
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Amino Acid Position ({title})", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plt.close()
    html_content = f"""
    <div id="plot-container-{title}" style="width: 100%; height: auto;">
        <img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="width: 100%; height: auto;">
    </div>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            var elem = document.getElementById("plot-container-{title}");
            if (elem.requestFullscreen) {{
                elem.requestFullscreen();
            }} else if (elem.mozRequestFullScreen) {{
                elem.mozRequestFullScreen();
            }} else if (elem.webkitRequestFullscreen) {{
                elem.webkitRequestFullscreen();
            }} else if (elem.msRequestFullscreen) {{
                elem.msRequestFullscreen();
            }}
        }});
    </script>
    """
    st.components.v1.html(html_content, height=300)

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
            ax.set_xlabel(f'Amino Acid Position ({condition})', fontsize=14)
            ax.tick_params(axis='x', labelsize=12)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='jpeg', dpi=600, bbox_inches='tight')
            img_buffer.seek(0)
            zipf.writestr(f"{protein_of_interest}_{condition}_linear.jpeg", img_buffer.read())
            plt.close(fig)
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit App
st.markdown(
    """
    <p style='text-align: justify; font-size: 16px; color: #4a4a4a;'>
   The **Peptide3D Mapper** is a web-based tool that visualizes peptide intensity data from proteomics experiments on AlphaFold 3D protein structures.
   Upload peptide CSV and FASTA files to compare conditions (e.g., control vs. disease) using z-score intensity scales.
   Explore residue-level differences in interactive 3D and linear sequence views, with customizable colors and exportable outputs.
    </p>
    """,
    unsafe_allow_html=True
)

if 'conditions_confirmed' not in st.session_state:
    st.session_state.conditions_confirmed = False
if 'processed' not in st.session_state:
    st.session_state.processed = False

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
                st.info("🔄 Processing... (This may take a moment for PDB fetch.)")
                base_id = selected_protein.split('-')[0]
                protein_seq = None
                for rec in seq_records:
                    parts = rec.id.split('|')
                    if len(parts) > 1 and parts[1] == base_id:
                        protein_seq = str(rec.seq)
                        break
                if protein_seq is None:
                    st.error("Protein sequence not found in FASTA. Ensure IDs match (e.g., UniProt prefix).")
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
                pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v4.pdb"
                with st.spinner("Fetching AlphaFold structure..."):
                    r = requests.get(pdb_url, timeout=10)
                    if r.status_code == 200:
                        pdb_str = r.text
                    else:
                        st.error(f"PDB fetch failed: {r.status_code}")
                        st.stop()
                bg_color = st.selectbox("Background Color", ["white", "black", "darkgrey"], index=1)
                # Add colormap and not-mapped color options
                cmap_options = ['autumn', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                selected_cmap = st.selectbox("Select Color Gradient", cmap_options, index=0)
                selected_not_mapped_color = st.color_picker("Select Not Mapped Color", "#d3d3d3")
                with st.container():
                    st.subheader("3D Structure Visualizations")
                    render_synced_viewers(pdb_str, residue_data[condition1_name], residue_data[condition2_name], bg_color, condition1_name, condition2_name, selected_cmap, selected_not_mapped_color)
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.subheader("Linear Sequence Visualizations")
                    st.markdown(f"#### {condition1_name}")
                    render_linear_plot(residue_data[condition1_name], condition1_name, seq_len,
                                       min_max_logs[condition1_name][0], min_max_logs[condition1_name][1], selected_cmap, selected_not_mapped_color)
                    st.markdown(f"#### {condition2_name}")
                    render_linear_plot(residue_data[condition2_name], condition2_name, seq_len,
                                       min_max_logs[condition2_name][0], min_max_logs[condition2_name][1], selected_cmap, selected_not_mapped_color)
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.subheader("Colorbar")
                    overall_vmin = min(min_max_logs[condition1_name][0], min_max_logs[condition2_name][0])
                    overall_vmax = max(min_max_logs[condition1_name][1], min_max_logs[condition2_name][1])
                    fig, ax = plt.subplots(figsize=(4, 0.2))
                    norm = Normalize(vmin=overall_vmin, vmax=overall_vmax)
                    sm = ScalarMappable(cmap=colormaps[selected_cmap], norm=norm)
                    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                    cbar.set_label('Z-Score Intensity', fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
                    buf.seek(0)
                    plt.close(fig)
                    html_content = f"""
                    <div id="colorbar-container" style="width: 100%; height: auto; max-width: 400px;">
                        <img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="width: 100%; height: auto;">
                    </div>
                    <script>
                        document.addEventListener("DOMContentLoaded", function() {{
                            var elem = document.getElementById("colorbar-container");
                            if (elem.requestFullscreen) {{
                                elem.requestFullscreen();
                            }} else if (elem.mozRequestFullScreen) {{
                                elem.mozRequestFullScreen();
                            }} else if (elem.webkitRequestFullscreen) {{
                                elem.webkitRequestFullscreen();
                            }} else if (elem.msRequestFullscreen) {{
                                elem.msRequestFullscreen();
                            }}
                        }});
                    </script>
                    """
                    st.components.v1.html(html_content, height=80)
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
                        st.rerun()
