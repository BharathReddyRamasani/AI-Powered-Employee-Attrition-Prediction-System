"""
EDA visualizations module — all Plotly charts for Phase 1 analysis.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Shared plot theme ─────────────────────────────────────────────────────────
DARK_BG = '#0a0e1a'
CARD_BG = '#111827'
ACCENT1 = '#6366f1'   # indigo
ACCENT2 = '#f59e0b'   # amber
DANGER  = '#ef4444'   # red
SUCCESS = '#10b981'   # emerald
MUTED   = '#6b7280'
TEXT    = '#f9fafb'

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.6)',
        font=dict(color=TEXT, family='Inter, sans-serif'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)'),
        margin=dict(l=20, r=20, t=40, b=20),
    )
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    return fig


# ── 1. Attrition Donut ───────────────────────────────────────────────────────
def attrition_donut(df: pd.DataFrame) -> go.Figure:
    counts = df['Attrition'].value_counts()
    fig = go.Figure(go.Pie(
        labels=['Retained', 'Attrited'],
        values=[counts.get('No', 0), counts.get('Yes', 0)],
        hole=0.65,
        marker=dict(colors=[SUCCESS, DANGER],
                    line=dict(color=DARK_BG, width=3)),
        textinfo='percent+label',
        textfont=dict(size=13, color=TEXT),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Overall Attrition Distribution', font=dict(size=16, color=TEXT), x=0.5),
        showlegend=True,
        **PLOTLY_TEMPLATE['layout'],
        annotations=[dict(
            text=f"{counts.get('Yes', 0) / len(df) * 100:.1f}%<br>Attrition",
            x=0.5, y=0.5, font_size=18, showarrow=False, font_color=DANGER
        )]
    )
    return fig


# ── 2. Department-wise Attrition Bar ─────────────────────────────────────────
def dept_attrition_bar(df: pd.DataFrame) -> go.Figure:
    dept = (
        df.groupby('Department')['Attrition']
        .apply(lambda x: (x == 'Yes').mean() * 100)
        .reset_index()
        .rename(columns={'Attrition': 'Rate'})
        .sort_values('Rate', ascending=True)
    )
    colors = [DANGER if r > 18 else ACCENT1 for r in dept['Rate']]
    fig = go.Figure(go.Bar(
        y=dept['Department'], x=dept['Rate'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)')),
        text=[f"{r:.1f}%" for r in dept['Rate']],
        textposition='outside',
        textfont=dict(color=TEXT),
        hovertemplate='<b>%{y}</b><br>Attrition Rate: %{x:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Attrition Rate by Department', font=dict(size=16, color=TEXT), x=0.5),
        xaxis_title='Attrition Rate (%)',
        **PLOTLY_TEMPLATE['layout'],
        height=320,
    )
    return fig


# ── 3. Gender Attrition Grouped Bar ──────────────────────────────────────────
def gender_attrition_bar(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby(['Gender', 'Attrition'])
        .size().reset_index(name='Count')
    )
    fig = px.bar(
        grp, x='Gender', y='Count', color='Attrition',
        barmode='group',
        color_discrete_map={'Yes': DANGER, 'No': SUCCESS},
        text='Count',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title=dict(text='Attrition by Gender', font=dict(size=16, color=TEXT), x=0.5),
        **PLOTLY_TEMPLATE['layout'],
        height=350,
        legend_title='Attrition',
    )
    return _apply_theme(fig)


# ── 4. Monthly Income Box Plot ────────────────────────────────────────────────
def income_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in [('No', SUCCESS), ('Yes', DANGER)]:
        subset = df[df['Attrition'] == label]['MonthlyIncome']
        fig.add_trace(go.Box(
            y=subset, name=f"{'Retained' if label == 'No' else 'Attrited'}",
            marker_color=color,
            line_color=color,
            fillcolor=color.replace(')', ',0.2)').replace('rgb', 'rgba') if 'rgb' in color else color,
            boxmean='sd',
            hovertemplate='<b>%{fullData.name}</b><br>%{y:$,.0f}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='Monthly Income vs Attrition', font=dict(size=16, color=TEXT), x=0.5),
        yaxis_title='Monthly Income ($)',
        **PLOTLY_TEMPLATE['layout'],
        height=380,
    )
    return fig


# ── 5. Experience (Years at Company) Histogram ───────────────────────────────
def experience_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in [('No', SUCCESS), ('Yes', DANGER)]:
        subset = df[df['Attrition'] == label]['YearsAtCompany']
        fig.add_trace(go.Histogram(
            x=subset, name=f"{'Retained' if label == 'No' else 'Attrited'}",
            marker_color=color, opacity=0.75,
            nbinsx=20,
            hovertemplate='<b>%{fullData.name}</b><br>Years: %{x}<br>Count: %{y}<extra></extra>',
        ))
    fig.update_layout(
        barmode='overlay',
        title=dict(text='Years at Company vs Attrition', font=dict(size=16, color=TEXT), x=0.5),
        xaxis_title='Years at Company',
        yaxis_title='Employee Count',
        **PLOTLY_TEMPLATE['layout'],
        height=350,
    )
    return fig


# ── 6. Job Role Attrition Heatmap ─────────────────────────────────────────────
def jobrole_attrition_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(['JobRole', 'Department'])['Attrition']
        .apply(lambda x: round((x == 'Yes').mean() * 100, 1))
        .reset_index(name='Rate')
        .pivot(index='JobRole', columns='Department', values='Rate')
        .fillna(0)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate='%{text}',
        hovertemplate='<b>%{y}</b> | %{x}<br>Attrition: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='Rate (%)', tickfont=dict(color=TEXT), titlefont=dict(color=TEXT)),
    ))
    fig.update_layout(
        title=dict(text='Attrition Rate by Job Role & Department (%)', font=dict(size=16, color=TEXT), x=0.5),
        **PLOTLY_TEMPLATE['layout'],
        height=420,
        xaxis_tickangle=-20,
    )
    return fig


# ── 7. Overtime Impact ────────────────────────────────────────────────────────
def overtime_attrition(df: pd.DataFrame) -> go.Figure:
    ot = (
        df.groupby('OverTime')['Attrition']
        .apply(lambda x: (x == 'Yes').mean() * 100)
        .reset_index()
        .rename(columns={'Attrition': 'Rate'})
    )
    colors = [DANGER if v > 20 else SUCCESS for v in ot['Rate']]
    fig = go.Figure(go.Bar(
        x=ot['OverTime'], y=ot['Rate'],
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in ot['Rate']],
        textposition='outside',
        textfont=dict(color=TEXT),
        hovertemplate='<b>Overtime: %{x}</b><br>Attrition: %{y:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Attrition Rate by Overtime Status', font=dict(size=16, color=TEXT), x=0.5),
        xaxis_title='Overtime', yaxis_title='Attrition Rate (%)',
        **PLOTLY_TEMPLATE['layout'],
        height=320,
    )
    return fig


# ── 8. Satisfaction Radar ─────────────────────────────────────────────────────
def satisfaction_radar(df: pd.DataFrame) -> go.Figure:
    sat_cols = ['JobSatisfaction', 'EnvironmentSatisfaction',
                'RelationshipSatisfaction', 'WorkLifeBalance', 'JobInvolvement']
    labels = ['Job Satisfaction', 'Environment', 'Relationships', 'Work-Life Balance', 'Involvement']

    yes_means = df[df['Attrition'] == 'Yes'][sat_cols].mean().tolist()
    no_means  = df[df['Attrition'] == 'No'][sat_cols].mean().tolist()

    fig = go.Figure()
    for vals, name, color in [(no_means, 'Retained', SUCCESS), (yes_means, 'Attrited', DANGER)]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]],
            fill='toself', name=name,
            line_color=color,
            fillcolor=color + '33',
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(17,24,39,0.6)',
            radialaxis=dict(visible=True, range=[1, 4], color=MUTED),
            angularaxis=dict(color=TEXT),
        ),
        title=dict(text='Satisfaction Profiles: Attrited vs Retained', font=dict(size=16, color=TEXT), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, family='Inter, sans-serif'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=40, r=40, t=60, b=20),
        height=420,
    )
    return fig


# ── 9. Age Distribution ───────────────────────────────────────────────────────
def age_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in [('No', SUCCESS), ('Yes', DANGER)]:
        subset = df[df['Attrition'] == label]['Age']
        fig.add_trace(go.Violin(
            y=subset, name=f"{'Retained' if label == 'No' else 'Attrited'}",
            box_visible=True, meanline_visible=True,
            fillcolor=color + '44', line_color=color,
            hovertemplate='<b>%{fullData.name}</b><br>Age: %{y}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='Age Distribution by Attrition', font=dict(size=16, color=TEXT), x=0.5),
        yaxis_title='Age',
        **PLOTLY_TEMPLATE['layout'],
        height=360,
    )
    return fig


# ── 10. Correlation Heatmap ───────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
                'DistanceFromHome', 'TotalWorkingYears', 'YearsSinceLastPromotion',
                'WorkLifeBalance', 'EnvironmentSatisfaction']
    df2 = df.copy()
    df2['Attrition_Num'] = (df2['Attrition'] == 'Yes').astype(int)
    corr = df2[num_cols + ['Attrition_Num']].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale='RdBu',
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate='%{text}',
        hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
        colorbar=dict(title='r', tickfont=dict(color=TEXT), titlefont=dict(color=TEXT)),
    ))
    fig.update_layout(
        title=dict(text='Feature Correlation Matrix', font=dict(size=16, color=TEXT), x=0.5),
        **PLOTLY_TEMPLATE['layout'],
        height=460,
        xaxis_tickangle=-30,
    )
    return fig
