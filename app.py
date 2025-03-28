import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import distance
from scipy.stats import pearsonr
import os
from datetime import datetime

# Streamlit Configuration
st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergent Cross Mapping (CCM) Analysis Tool")

# Helper Functions
def shadow_manifold(time_series_Y, L, E, tau):
    shadow_M = {}
    for t in range((E - 1) * tau, L):
        lag = [time_series_Y[t - t2 * tau] for t2 in range(0, E)]
        shadow_M[t] = lag
    return shadow_M

def vec_dist_matrx(shadow_M):
    steps = np.array(list(shadow_M.keys()))
    vecs = np.array(list(shadow_M.values()))
    return distance.cdist(vecs, vecs, metric="euclidean"), steps

def nearest_dist_and_step(timepoint_oi, steps, dist_matr, E):
    index = np.where(steps == timepoint_oi)[0][0]
    dists = dist_matr[index]
    nearest_indices = np.argsort(dists)[1:E+2]
    return steps[nearest_indices], dists[nearest_indices]

def find_causality(time_series_X, time_series_Y, L, E, tau):
    My = shadow_manifold(time_series_Y, L, E, tau)
    X_true, X_hat = [], []
    
    for t in My.keys():
        dist_matrix, steps = vec_dist_matrx(My)
        t_steps, t_dists = nearest_dist_and_step(t, steps, dist_matrix, E)
        weights = np.exp(-t_dists / np.max([0.000001, t_dists[0]]))
        weights /= weights.sum()
        X_true.append(time_series_X[t])
        X_hat.append((weights * np.array(time_series_X)[t_steps]).sum())
    
    return pearsonr(X_true, X_hat)

def show_current_pair():
    """Displays the current convergence plot and decision buttons"""
    source, target = st.session_state.all_pairs[st.session_state.current_pair_index]
    pair_score = st.session_state.ccm_matrix.loc[source, target]
    
    # Show progress
    total_pairs = len(st.session_state.all_pairs)
    current_pair_num = st.session_state.current_pair_index + 1
    st.write(f"Evaluating pair {current_pair_num} of {total_pairs}")
    
    # Create smaller convergence plot
    fig, ax = plt.subplots(figsize=(6, 3))  # Compact figure size
    
    source_to_target = []
    target_to_source = []
    for L_val in L_range:
        st_r, _ = find_causality(ccm_data[source].values, ccm_data[target].values, L_val, E, tau)
        ts_r, _ = find_causality(ccm_data[target].values, ccm_data[source].values, L_val, E, tau)
        source_to_target.append(max(0, st_r))
        target_to_source.append(max(0, ts_r))
    
    # Plot with consistent colors and legend in bottom right
    ax.plot(L_range, source_to_target, 'b-o', label=f"{source} → {target}", linewidth=1, markersize=3)
    ax.plot(L_range, target_to_source, 'r-s', label=f"{target} → {source}", linewidth=1, markersize=3)
    ax.set_xlabel('Library Size (L)')
    ax.set_ylabel('Cross-Map Skill (ρ)')
    ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=False)
    
    # Decision buttons
    cols = st.columns([1,1,1,1.3])  # Wider column for "No convergence"
    with cols[0]:
        if st.button("Both directions", key=f"both_{st.session_state.current_pair_index}"):
            save_pair_result(source, target, source_to_target[-1], target_to_source[-1])
    with cols[1]:
        if st.button(f"Only {source}→", key=f"xy_{st.session_state.current_pair_index}"):
            save_pair_result(source, target, source_to_target[-1], 0)
    with cols[2]:
        if st.button(f"Only {target}→", key=f"yx_{st.session_state.current_pair_index}"):
            save_pair_result(source, target, 0, target_to_source[-1])
    with cols[3]:
        if st.button("⨯ No convergence", key=f"none_{st.session_state.current_pair_index}"):
            st.session_state.processed_pairs.add(f"{source}_{target}")
            advance_to_next_pair()
    
    # Navigation buttons
    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.session_state.current_pair_index > 0 and st.button("◀ Previous Pair"):
            st.session_state.current_pair_index -= 1
            st.rerun()
    with nav_cols[1]:
        if st.session_state.current_pair_index < len(st.session_state.all_pairs) - 1 and st.button("Next Pair ▶"):
            st.session_state.current_pair_index += 1
            st.rerun()
        elif st.session_state.current_pair_index == len(st.session_state.all_pairs) - 1 and st.button("Finish Evaluation"):
            st.session_state.current_pair_index += 1
            st.rerun()

def save_pair_result(source, target, st_score, ts_score):
    """Saves the pair result and advances to next pair"""
    st.session_state.convergence_df.loc[len(st.session_state.convergence_df)] = [
        source, target, st_score, ts_score]
    st.session_state.processed_pairs.add(f"{source}_{target}")
    advance_to_next_pair()

def advance_to_next_pair():
    """Advances to next pair with proper bounds checking"""
    st.session_state.current_pair_index += 1
    if st.session_state.current_pair_index >= len(st.session_state.all_pairs):
        st.session_state.current_pair_index = len(st.session_state.all_pairs) - 1
    st.rerun()

def show_final_results():
    """Displays the final results after all pairs are processed"""
    if not st.session_state.convergence_df.empty:
        st.subheader("Final Convergent Pairs")
        
        # Create final species list
        all_species = sorted(set(st.session_state.convergence_df['Source']).union(
                       set(st.session_state.convergence_df['Target'])))
        species_df = pd.DataFrame(all_species, columns=["Species"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(st.session_state.convergence_df.style.format({
                'Source_to_Target_ρ': '{:.3f}',
                'Target_to_Source_ρ': '{:.3f}'
            }), height=400)
        with col2:
            st.dataframe(species_df, height=400)
        
        # Create final matrix for heatmap
        final_matrix = pd.DataFrame(np.zeros((len(all_species), len(all_species))),
                                  index=all_species, columns=all_species)
        
        for _, row in st.session_state.convergence_df.iterrows():
            if row['Source_to_Target_ρ'] > THRESHOLD:
                final_matrix.loc[row['Source'], row['Target']] = row['Source_to_Target_ρ']
            if row['Target_to_Source_ρ'] > THRESHOLD:
                final_matrix.loc[row['Target'], row['Source']] = row['Target_to_Source_ρ']
        
        # Download buttons
        st.download_button(
            label="Download Convergent Pairs (CSV)",
            data=st.session_state.convergence_df.to_csv(index=False),
            file_name="ccm_convergent_pairs.csv",
            mime='text/csv'
        )
        
        st.download_button(
            label="Download Species List (CSV)",
            data=species_df.to_csv(index=False),
            file_name="ccm_species_list.csv",
            mime='text/csv'
        )
        
        st.download_button(
            label="Download Heatmap Data (CSV)",
            data=final_matrix.to_csv(),
            file_name="ccm_heatmap_matrix.csv",
            mime='text/csv'
        )
        
        # Compact Heatmap with improved layout
        st.subheader("Final CCM Network")
        
        # Calculate dynamic figure size based on number of species
        num_species = len(all_species)
        base_size = 6  # Base size for the figure
        max_size = 10  # Maximum figure size
        figsize = min(base_size + num_species * 0.2, max_size)
        
        fig, ax = plt.subplots(figsize=(figsize, figsize*0.8))  # Wider than tall
        
        # Improved colormap
        cmap = plt.cm.Reds
        cmap.set_under('white')
        norm = colors.Normalize(vmin=THRESHOLD+0.01, vmax=1)
        
        # Create heatmap with adjusted parameters
        im = ax.imshow(final_matrix, cmap=cmap, norm=norm, aspect='auto')
        
        # Adjust font size based on number of species
        fontsize = 8 if num_species < 15 else 6
        
        # Add text annotations only for significant values
        for i in range(num_species):
            for j in range(num_species):
                if i != j and final_matrix.iloc[i, j] > THRESHOLD:
                    ax.text(j, i, f"{final_matrix.iloc[i, j]:.2f}",
                           ha='center', va='center',
                           color='black' if final_matrix.iloc[i, j] < 0.5 else 'white',
                           fontsize=fontsize, weight='bold')
        
        # Configure axes
        ax.set_xticks(range(num_species))
        ax.set_yticks(range(num_species))
        ax.set_xticklabels(all_species, rotation=45, ha='right', fontsize=fontsize)
        ax.set_yticklabels(all_species, fontsize=fontsize)
        ax.set_title(f"Significant CCM Relationships (ρ > {THRESHOLD})", pad=15, fontsize=fontsize+2)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, num_species, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_species, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2)
        
        # Compact colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Cross-Map Score', rotation=270, labelpad=15, fontsize=fontsize)
        
        # Adjust layout to prevent cutting off labels
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No convergent pairs were found above the threshold")

# Initialize session state
if 'ccm_matrix' not in st.session_state:
    st.session_state.ccm_matrix = None
if 'convergence_df' not in st.session_state:
    st.session_state.convergence_df = pd.DataFrame(columns=['Source', 'Target', 'Source_to_Target_ρ', 'Target_to_Source_ρ'])
if 'processed_pairs' not in st.session_state:
    st.session_state.processed_pairs = set()
if 'current_pair_index' not in st.session_state:
    st.session_state.current_pair_index = 0
if 'all_pairs' not in st.session_state:
    st.session_state.all_pairs = []

# Sidebar for Parameters
with st.sidebar:
    st.header("Analysis Parameters")
    THRESHOLD = st.slider("Convergence Threshold", 0.1, 1.0, 0.8)
    E = st.number_input("Embedding Dimension (E)", 1, 5, 2)
    tau = st.number_input("Time Delay (τ)", 1, 10, 1)

# File Upload
uploaded_file = st.file_uploader("Upload Time Series Data (TXT/CSV)", type=["txt", "csv"])

if uploaded_file:
    try:
        # Load data with automatic delimiter detection
        if uploaded_file.name.endswith('.txt'):
            sep = '\t' if '\t' in uploaded_file.getvalue().decode()[:1000] else ','
            ccm_data = pd.read_csv(uploaded_file, sep=sep, engine='python')
        else:
            ccm_data = pd.read_csv(uploaded_file)
        
        # Set default L_MAX to length of time series
        max_possible_L = len(ccm_data)
        
        # Add L selection to sidebar after file is loaded
        with st.sidebar:
            L_MAX = st.number_input("Maximum Library Size (L)", 
                                  min_value=10, 
                                  max_value=max_possible_L, 
                                  value=max_possible_L,
                                  help=f"Maximum possible value based on your data: {max_possible_L}")
            L_range = range(5, L_MAX, 5)
        
        st.success(f"Data loaded successfully! Shape: {ccm_data.shape}")
        st.info(f"Time series length: {max_possible_L} | Current L range: {L_range[0]} to {L_range[-1]} (step 5)")
        
        # Raw Data Preview
        with st.expander("🔍 View Raw Data"):
            st.dataframe(ccm_data)
            st.write(f"Columns detected: {list(ccm_data.columns)}")
        
        # CCM Matrix Calculation
        if st.button("Calculate CCM Matrix"):
            with st.spinner("Calculating initial CCM matrix..."):
                empty_df = pd.DataFrame(index=ccm_data.columns, columns=ccm_data.columns)
                
                for i, col1 in enumerate(ccm_data.columns):
                    for j, col2 in enumerate(ccm_data.columns):
                        if i != j:
                            r, _ = find_causality(ccm_data[col1].values, 
                                                ccm_data[col2].values, 
                                                L_MAX, E, tau)
                            empty_df.loc[col1, col2] = max(0, round(r, 4))
                
                st.session_state.ccm_matrix = empty_df.fillna(0)
                st.session_state.processed_pairs = set()
                st.session_state.current_pair_index = 0
                
                # Prepare all pairs above threshold
                st.session_state.all_pairs = []
                for i, source in enumerate(ccm_data.columns):
                    for j, target in enumerate(ccm_data.columns):
                        if i != j and st.session_state.ccm_matrix.loc[source, target] > THRESHOLD:
                            st.session_state.all_pairs.append((source, target))
            
            # Initial Matrix Display
            st.subheader("Initial CCM Matrix")
            st.dataframe(st.session_state.ccm_matrix.style.format("{:.2f}").background_gradient(
                cmap='Reds', vmin=THRESHOLD, vmax=1.0))
        
        # Only show convergence analysis if we have pairs to process
        if st.session_state.ccm_matrix is not None and len(st.session_state.all_pairs) > 0:
            st.subheader("Convergence Evaluation")
            
            # Check if we've processed all pairs
            if st.session_state.current_pair_index >= len(st.session_state.all_pairs):
                st.success("All pairs processed!")
                show_final_results()
            else:
                # Show current pair
                show_current_pair()

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a data file to begin analysis")

# Instructions
with st.expander("📘 How to use this tool"):
    st.markdown("""
    ### Step-by-Step Guide:
    1. **Upload your data** (TXT or CSV format)
    2. **Adjust parameters** in the sidebar:
       - Threshold, E, τ will appear immediately
       - L selection appears after data upload
    3. Click **Calculate CCM Matrix**
    4. **Evaluate convergence plots** one by one
    5. **Select relationships** using the buttons
    6. **View/download** final results when all pairs are processed
    
    ### Key Features:
    - L can be adjusted but defaults to your data length
    - All results (pairs, species, heatmap matrix) are downloadable
    """)