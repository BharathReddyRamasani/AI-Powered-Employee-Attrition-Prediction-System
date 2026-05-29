"""
Clustering module — K-Means + PCA for employee segmentation.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

TEXT    = '#f9fafb'
ACCENT1 = '#6366f1'
ACCENT2 = '#f59e0b'
DANGER  = '#ef4444'
SUCCESS = '#10b981'
MUTED   = '#6b7280'

# Extended clusters configuration to support dynamic cluster counts (3 to 6)
CLUSTER_COLORS = ['#6366f1', '#f59e0b', '#ef4444', '#10b981', '#a78bfa', '#fb923c']
CLUSTER_NAMES  = ['🏆 High Performers', '⚠️ At Risk', '🌱 New Employees', '🎖️ Veterans', '💡 Rising Stars', '🔄 Transitional']
CLUSTER_DESC   = [
    'Experienced, satisfied, high-income employees with low attrition risk.',
    'Mid-tenure employees showing warning signs: lower satisfaction or income.',
    'Early-career employees adapting; moderate to high attrition potential.',
    'Long-tenured senior leaders.',
    'High-potential employees on the rise.',
    'Employees in career transition phase.',
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.6)',
    font=dict(color=TEXT, family='Inter, sans-serif'),
    margin=dict(l=20, r=20, t=50, b=20),
)


@st.cache_resource(show_spinner=False)
def run_pca_kmeans(X_scaled: np.ndarray, n_clusters: int = 3):
    """
    Run PCA (2D) + K-Means clustering.
    Returns: (pca_coords, cluster_labels, pca_model, kmeans_model, explained_variance)
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    return X_pca, labels, pca, kmeans, pca.explained_variance_ratio_


@st.cache_resource(show_spinner=False)
def run_pca_3d(X_scaled: np.ndarray, n_clusters: int = 3):
    """Run PCA with 3 components for 3D visualization."""
    pca3 = PCA(n_components=3, random_state=42)
    X_pca3 = pca3.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca3)
    return X_pca3, labels, pca3.explained_variance_ratio_


def pca_scatter_2d(X_pca: np.ndarray, labels: np.ndarray,
                   df_orig: pd.DataFrame, explained_var: np.ndarray) -> go.Figure:
    """2D PCA scatter colored by K-Means cluster."""
    fig = go.Figure()
    unique_clusters = sorted(np.unique(labels))

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        name = CLUSTER_NAMES[cluster_id % len(CLUSTER_NAMES)]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1],
            mode='markers',
            name=name,
            marker=dict(
                color=color,
                size=6, opacity=0.75,
                line=dict(color='rgba(0,0,0,0.3)', width=0.5),
            ),
            hovertemplate=(
                f'<b>{name}</b><br>'
                'PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f'PCA Employee Segmentation (PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%)',
            font=dict(size=16, color=TEXT), x=0.5
        ),
        xaxis_title=f'Principal Component 1 ({explained_var[0]*100:.1f}% variance)',
        yaxis_title=f'Principal Component 2 ({explained_var[1]*100:.1f}% variance)',
        **PLOTLY_LAYOUT,
        height=480,
        legend=dict(bgcolor='rgba(17,24,39,0.8)',
                    bordercolor='rgba(99,102,241,0.3)', borderwidth=1),
    )
    return fig


def pca_scatter_3d(X_pca3: np.ndarray, labels: np.ndarray,
                   explained_var: np.ndarray) -> go.Figure:
    """3D PCA scatter."""
    fig = go.Figure()
    unique_clusters = sorted(np.unique(labels))

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        name = CLUSTER_NAMES[cluster_id % len(CLUSTER_NAMES)]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=X_pca3[mask, 0], y=X_pca3[mask, 1], z=X_pca3[mask, 2],
            mode='markers',
            name=name,
            marker=dict(
                color=color,
                size=3, opacity=0.7,
            ),
        ))
    fig.update_layout(
        title=dict(text='3D PCA Segmentation', font=dict(size=16, color=TEXT), x=0.5),
        scene=dict(
            bgcolor='rgba(17,24,39,0.6)',
            xaxis=dict(title=f'PC1 ({explained_var[0]*100:.1f}%)', color=TEXT, gridcolor=MUTED),
            yaxis=dict(title=f'PC2 ({explained_var[1]*100:.1f}%)', color=TEXT, gridcolor=MUTED),
            zaxis=dict(title=f'PC3 ({explained_var[2]*100:.1f}%)', color=TEXT, gridcolor=MUTED),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, family='Inter, sans-serif'),
        height=520,
    )
    return fig


def cluster_profile_bars(X_pca: np.ndarray, labels: np.ndarray,
                          df_orig: pd.DataFrame) -> go.Figure:
    """Bar chart showing mean feature values per cluster."""
    feature_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany',
                    'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']
    available = [c for c in feature_cols if c in df_orig.columns]

    # Assign cluster labels
    df_c = df_orig[available].copy()
    df_c = df_c.iloc[:len(labels)].copy()
    df_c['Cluster'] = labels

    means = df_c.groupby('Cluster')[available].mean().reset_index()

    # Normalise for comparison
    for col in available:
        mn, mx = means[col].min(), means[col].max()
        means[col] = (means[col] - mn) / (mx - mn + 1e-9) * 100

    fig = go.Figure()
    unique_clusters = sorted(means['Cluster'].unique())

    for cluster_id in unique_clusters:
        matching = means[means['Cluster'] == cluster_id]
        if len(matching) == 0:
            continue
        row = matching.iloc[0]
        name = CLUSTER_NAMES[cluster_id % len(CLUSTER_NAMES)]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        fig.add_trace(go.Bar(
            name=name,
            x=available,
            y=[row[c] for c in available],
            marker_color=color,
            opacity=0.85,
        ))
    fig.update_layout(
        barmode='group',
        title=dict(text='Cluster Profile (Normalised Features)',
                   font=dict(size=16, color=TEXT), x=0.5),
        xaxis_title='Feature', yaxis_title='Normalised Value (0–100)',
        **PLOTLY_LAYOUT,
        height=380,
        legend=dict(bgcolor='rgba(17,24,39,0.8)',
                    bordercolor='rgba(99,102,241,0.3)', borderwidth=1),
    )
    return fig


def cluster_attrition_pie(labels: np.ndarray, df_orig: pd.DataFrame) -> go.Figure:
    """Donut charts showing attrition composition per cluster."""
    unique_clusters = sorted(np.unique(labels))
    n_cl = len(unique_clusters)
    colors_yes = [c + 'cc' for c in CLUSTER_COLORS]
    colors_no  = ['#10b981' + 'aa'] * n_cl

    df_c = df_orig.copy().iloc[:len(labels)]
    df_c['Cluster'] = labels

    fig = go.Figure()
    attrition_col = df_orig.columns[df_orig.columns.str.lower() == 'attrition'].tolist()
    if not attrition_col:
        return fig
    acol = attrition_col[0]

    for cid in unique_clusters:
        subset = df_c[df_c['Cluster'] == cid]
        yes = (subset[acol] == 'Yes').sum()
        no  = (subset[acol] == 'No').sum()
        rate = yes / (yes + no) * 100 if (yes + no) > 0 else 0
        fig.add_annotation(
            text=f"Cluster {cid+1}<br>{rate:.1f}% Attrition",
            x=cid / n_cl + 1/(2*n_cl) if n_cl > 0 else 0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color=TEXT),
            xref='paper', yref='paper',
        )

    return cluster_profile_bars(X_pca=None, labels=labels, df_orig=df_orig)


def get_cluster_summary(labels: np.ndarray, df_orig: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per cluster."""
    df_c = df_orig.copy().iloc[:len(labels)]
    df_c['Cluster'] = labels

    num_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany',
                'JobSatisfaction', 'TotalWorkingYears']
    available = [c for c in num_cols if c in df_c.columns]

    summary = df_c.groupby('Cluster')[available].mean().round(1)
    summary.index = [CLUSTER_NAMES[i % len(CLUSTER_NAMES)] for i in summary.index]

    if 'Attrition' in df_c.columns:
        att_rate = df_c.groupby('Cluster')['Attrition'].apply(
            lambda x: round((x == 'Yes').mean() * 100, 1)
        )
        att_rate.index = [CLUSTER_NAMES[i % len(CLUSTER_NAMES)] for i in att_rate.index]
        summary['Attrition Rate (%)'] = att_rate

    summary['Size'] = df_c.groupby('Cluster').size().values

    return summary
