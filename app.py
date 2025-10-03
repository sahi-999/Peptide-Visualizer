import streamlit as st
import pandas as pd
import numpy as np
from Bio import SeqIO
import py3Dmol
import io
import requests
from matplotlib import cm, colors
import matplotlib.pyplot as plt

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
            continue
        end = start + len(pep)
        for i in range(start, end):
            if residue_vals[i] is None:
                residue_vals[i] = [z_scores[idx]]
            else:
                residue_vals[i].append(z_scores[idx])

    for i in range(seq_len):
        if residue_vals[i]:
            residue_vals[i] = np.mean(residue_vals[i])
        else:
            residue_vals[i] = None
    return residue_vals

def generate_colormap(residue_vals, cmap_name='autumn'):
    cmap = cm.get_cmap(cmap_name)
    vals = [v for v in residue_vals if v is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0,1)
    hex_colors = []
    for val in residue_vals:
        if val is None:
            hex_colors.append('#d3d3d3')
        else:
            norm = (val - vmin)/(vmax - vmin) if vmax>vmin else 0.5
            hex_colors.append(colors.rgb2hex(cmap(norm)[:3]))
    return hex_colors, vmin, vmax

def render_viewer(pdb_str, residue_vals, bg_color, title, vmin, vmax):
    hex_colors, _, _ = generate_colormap(residue_vals)
    view = py3Dmol.view(width=400, height=400)
    view.addModel(pdb_str, 'pdb')
    view.setBackgroundColor(bg_color)
    view.setStyle({}, {'cartoon': {'color': 'lightgray'}})
    for i, c in enumerate(hex_colors):
        view.setStyle({'resi': str(i+1)}, {'cartoon': {'color': c}})
    view.zoomTo()

    st.markdown(f"#### {title}")
    st.components.v1.html(view._make_html(), height=420)

    # Show colorbar below each viewer
    fig, ax = plt.subplots(figsize=(4,0.5))
    sm = plt.cm.ScalarMappable(cmap=cm.autumn, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label(f'{title} Z-score')
    st.pyplot(fig)

# ------------------------
# Streamlit App
# ------------------------

st.title("Peptide Z-Score Visualization App")

csv_file = st.file_uploader("Upload Peptide CSV", type=["csv"])
fasta_file = st.file_uploader("Upload FASTA", type=["fasta"])

if csv_file and fasta_file:
    df = pd.read_csv(csv_file)
    fasta_str = fasta_file.getvalue().decode("utf-8")
    fasta_handle = io.StringIO(fasta_str)
    fasta_records = list(SeqIO.parse(fasta_handle, "fasta"))

    intensity_cols = [c for c in df.columns if 'intensity' in c.lower()]
    if len(intensity_cols) != 2:
        st.warning(f"Expected 2 intensity columns, found {intensity_cols}")
    else:
        control_col = st.selectbox("Select Control Column", intensity_cols)
        disease_col = st.selectbox("Select Disease Column", [c for c in intensity_cols if c != control_col])

        protein_options = sorted(df['Protein.Group'].unique())
        selected_protein = st.selectbox("Select Protein", protein_options)

        bg_color = st.selectbox("Background Color", ["white","black","darkgrey"], index=1)

        protein_seq = None
        for rec in fasta_records:
            if selected_protein in rec.id:
                protein_seq = str(rec.seq)
                break

        if protein_seq is None:
            st.error("Protein sequence not found in FASTA")
        else:
            df_protein = df[df['Protein.Group'].str.contains(selected_protein)]
            residues_control = map_peptides_to_residues(df_protein, protein_seq, control_col)
            residues_disease = map_peptides_to_residues(df_protein, protein_seq, disease_col)

            base_id = selected_protein.split('-')[0]
            pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{base_id}-F1-model_v4.pdb"
            r = requests.get(pdb_url)
            if r.status_code==200:
               pdb_str = r.text

    # safely compute min/max ignoring None
    control_vals = [v for v in residues_control if v is not None]
    disease_vals = [v for v in residues_disease if v is not None]

    if control_vals and disease_vals:  # make sure both are non-empty
        col1, col2 = st.columns(2)
        with col1:
            render_viewer(pdb_str, residues_control, bg_color, "Control",
                          min(control_vals), max(control_vals))
        with col2:
            render_viewer(pdb_str, residues_disease, bg_color, "Disease",
                          min(disease_vals), max(disease_vals))
    else:
        st.error("No valid Z-score values found for this protein.")
