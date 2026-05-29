"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AI-Powered Employee Attrition Prediction System                           ║
║   IBM HR Analytics — Streamlit Application                                  ║
║   Phase 1: EDA  |  Phase 2: ML Models  |  Segmentation & Prediction        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AttritionAI — Employee Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Source path ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_raw_data, preprocess_data, get_eda_stats
from src import eda as EDA
from src import models as ML
from src import clustering as CLU

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="nav-logo">
        <span style="font-size:2rem;">🧠</span>
        <div>
            <div class="nav-logo-text">AttritionAI</div>
            <div class="nav-logo-sub">IBM HR Analytics System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Overview Dashboard",
            "📊  EDA — Phase 1",
            "🤖  ML Models — Phase 2",
            "🌳  Decision Tree Rules",
            "🔮  Predict Employee",
            "🧩  Employee Segmentation",
            "📋  Dataset Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#6b7280; line-height:1.8;">
        <b style="color:#a78bfa;">Dataset:</b> IBM HR Analytics<br>
        <b style="color:#a78bfa;">Records:</b> 1,470 employees<br>
        <b style="color:#a78bfa;">Features:</b> 35 attributes<br>
        <b style="color:#a78bfa;">Models:</b> LR · DT · RF · SVM · KNN · NB
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached globally)
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Loading IBM HR dataset…"):
    df_raw = load_raw_data()

with st.spinner("⚙️ Preprocessing features…"):
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc, feature_names,
     scaler, df_enc) = preprocess_data(df_raw)

stats = get_eda_stats(df_raw)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi_card(label: str, value: str, delta: str = "", delta_type: str = "neutral", icon: str = ""):
    delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card animate-in">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, badge: str = "", icon: str = ""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <h2 class="section-title">{icon} {title}</h2>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview Dashboard":
    # Hero
    st.markdown("""
    <div class="hero-banner animate-in">
        <h1 class="hero-title">AI-Powered Employee Attrition Prediction</h1>
        <p class="hero-sub">
            IBM HR Analytics · 1,470 Employees · 6 ML Models · Real-time Risk Scoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Employees", f"{stats['total']:,}", icon="👥")
    with c2:
        kpi_card("Attrition Count", f"{stats['attrition_yes']:,}",
                 delta=f"▲ {stats['attrition_rate']}% of workforce",
                 delta_type="up", icon="🚨")
    with c3:
        kpi_card("Retained Employees", f"{stats['attrition_no']:,}",
                 delta=f"✓ {100-stats['attrition_rate']:.1f}% retention rate",
                 delta_type="down", icon="✅")
    with c4:
        kpi_card("Avg Monthly Income", f"${stats['income_no']:,.0f}",
                 delta=f"Attrited avg: ${stats['income_yes']:,.0f}",
                 delta_type="neutral", icon="💰")

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Row 2
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Average Age", f"{stats['avg_age']} yrs", icon="📅")
    with c2:
        kpi_card("Avg Tenure", f"{stats['avg_tenure']} yrs", icon="🏢")
    with c3:
        top_dept = stats['dept_attrition'].sort_values('Attrition_Rate', ascending=False).iloc[0]
        kpi_card("Highest Risk Dept", top_dept['Department'],
                 delta=f"▲ {top_dept['Attrition_Rate']:.1f}% attrition",
                 delta_type="up", icon="⚠️")
    with c4:
        income_diff = stats['income_no'] - stats['income_yes']
        kpi_card("Income Gap", f"${income_diff:,.0f}",
                 delta="Retained vs Attrited avg",
                 delta_type="neutral", icon="📊")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns([1, 1.6])
    with col_left:
        st.plotly_chart(EDA.attrition_donut(df_raw), use_container_width=True)
    with col_right:
        st.plotly_chart(EDA.dept_attrition_bar(df_raw), use_container_width=True)

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.plotly_chart(EDA.overtime_attrition(df_raw), use_container_width=True)
    with col_right2:
        st.plotly_chart(EDA.gender_attrition_bar(df_raw), use_container_width=True)

    # Key Insights
    section_header("Key Business Insights", badge="AI-Derived", icon="💡")
    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.8rem;margin-bottom:0.5rem">📉</div>
            <div style="font-weight:700;margin-bottom:0.4rem;color:#f9fafb">Income Drives Attrition</div>
            <div style="font-size:0.88rem;color:#6b7280">
                Employees who leave earn on average <b style="color:#ef4444">
                ${:,.0f}</b> less per month than those who stay.
                Salary benchmarking is critical.
            </div>
        </div>
        """.format(income_diff), unsafe_allow_html=True)
    with ins2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.8rem;margin-bottom:0.5rem">🔥</div>
            <div style="font-weight:700;margin-bottom:0.4rem;color:#f9fafb">Overtime = Risk</div>
            <div style="font-size:0.88rem;color:#6b7280">
                Employees working overtime show significantly higher attrition rates.
                Work-life balance initiatives can reduce this risk substantially.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ins3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.8rem;margin-bottom:0.5rem">🎯</div>
            <div style="font-weight:700;margin-bottom:0.4rem;color:#f9fafb">Sales Dept at Risk</div>
            <div style="font-size:0.88rem;color:#6b7280">
                Sales department has the highest attrition rate at
                <b style="color:#f59e0b">20.6%</b> — targeted retention
                programs are urgently needed.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA — PHASE 1
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA — Phase 1":
    st.markdown("""
    <div class="hero-banner animate-in">
        <h1 class="hero-title">Exploratory Data Analysis</h1>
        <p class="hero-sub">Phase 1 — Uncovering patterns, trends & risk factors in the IBM HR dataset</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "📈 Attrition Overview", "🏢 By Department", "👥 By Gender",
        "💰 Salary Impact", "⏱️ Experience", "🌡️ Satisfaction",
        "🗺️ Correlation", "📋 Statistics"
    ])

    with tabs[0]:
        section_header("Overall Attrition Distribution", icon="📈")
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.plotly_chart(EDA.attrition_donut(df_raw), use_container_width=True)
        with c2:
            st.plotly_chart(EDA.age_distribution(df_raw), use_container_width=True)
        st.markdown("""
        <div class="glass-card">
            <b>📌 Finding:</b> The overall attrition rate is <b style="color:#ef4444">16.1%</b>.
            Younger employees (25–35) show higher attrition tendency, suggesting
            career growth and development programs are needed for early-career talent.
        </div>
        """, unsafe_allow_html=True)

    with tabs[1]:
        section_header("Department-wise Attrition Analysis", icon="🏢")
        st.plotly_chart(EDA.dept_attrition_bar(df_raw), use_container_width=True)
        st.plotly_chart(EDA.jobrole_attrition_heatmap(df_raw), use_container_width=True)
        st.markdown("""
        <div class="glass-card">
            <b>📌 Finding:</b> <b style="color:#ef4444">Sales (20.6%)</b> and
            <b style="color:#f59e0b">Human Resources (19.0%)</b> have the highest attrition.
            Research & Development is relatively stable at 13.8%.
        </div>
        """, unsafe_allow_html=True)

    with tabs[2]:
        section_header("Gender-wise Attrition", icon="👥")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(EDA.gender_attrition_bar(df_raw), use_container_width=True)
        with c2:
            # Pie breakdown
            gender_counts = df_raw.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
            fig = go.Figure()
            for gender, color in [('Male', '#6366f1'), ('Female', '#f59e0b')]:
                sub = gender_counts[gender_counts['Gender'] == gender]
                yes_val = sub[sub['Attrition']=='Yes']['Count'].sum() if len(sub[sub['Attrition']=='Yes']) else 0
                no_val  = sub[sub['Attrition']=='No']['Count'].sum()  if len(sub[sub['Attrition']=='No'])  else 0
                total   = yes_val + no_val
                rate    = yes_val / total * 100 if total > 0 else 0
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=round(rate, 1),
                    number=dict(suffix="%", font=dict(color='#f9fafb', size=28)),
                    title=dict(text=f"{gender} Attrition Rate", font=dict(color='#6b7280', size=13)),
                    gauge=dict(
                        axis=dict(range=[0, 30], tickcolor='#6b7280'),
                        bar=dict(color=color),
                        bgcolor='rgba(17,24,39,0.6)',
                        bordercolor='rgba(255,255,255,0.1)',
                        steps=[
                            dict(range=[0, 10], color='rgba(16,185,129,0.15)'),
                            dict(range=[10, 20], color='rgba(245,158,11,0.15)'),
                            dict(range=[20, 30], color='rgba(239,68,68,0.15)'),
                        ],
                    ),
                    domain=dict(
                        x=[0, 0.45] if gender=='Male' else [0.55, 1],
                        y=[0, 1]
                    ),
                ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb', family='Inter'),
                height=280,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        section_header("Monthly Income Impact on Attrition", icon="💰")
        st.plotly_chart(EDA.income_box(df_raw), use_container_width=True)
        # Income bins
        df_temp = df_raw.copy()
        df_temp['Income Band'] = pd.cut(df_temp['MonthlyIncome'],
                                         bins=[0, 2500, 5000, 10000, 20000],
                                         labels=['< $2.5K', '$2.5K–5K', '$5K–10K', '> $10K'])
        income_att = df_temp.groupby('Income Band')['Attrition'].apply(
            lambda x: (x=='Yes').mean()*100
        ).reset_index(name='Attrition Rate (%)')
        fig2 = go.Figure(go.Bar(
            x=income_att['Income Band'],
            y=income_att['Attrition Rate (%)'],
            marker=dict(color=['#ef4444','#f59e0b','#6366f1','#10b981']),
            text=[f"{v:.1f}%" for v in income_att['Attrition Rate (%)']],
            textposition='outside',
            textfont=dict(color='#f9fafb'),
        ))
        fig2.update_layout(
            title=dict(text='Attrition Rate by Income Band', font=dict(size=15, color='#f9fafb'), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.6)',
            font=dict(color='#f9fafb', family='Inter'), height=320,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[4]:
        section_header("Experience & Tenure Analysis", icon="⏱️")
        st.plotly_chart(EDA.experience_histogram(df_raw), use_container_width=True)
        # Years since promotion
        df_temp2 = df_raw.copy()
        fig3 = go.Figure()
        for label, color in [('No','#10b981'),('Yes','#ef4444')]:
            sub = df_temp2[df_temp2['Attrition']==label]['YearsSinceLastPromotion']
            fig3.add_trace(go.Box(
                y=sub, name=f"{'Retained' if label=='No' else 'Attrited'}",
                marker_color=color, line_color=color, boxmean='sd',
            ))
        fig3.update_layout(
            title=dict(text='Years Since Last Promotion vs Attrition',
                       font=dict(size=15, color='#f9fafb'), x=0.5),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.6)',
            font=dict(color='#f9fafb', family='Inter'), height=340,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(EDA.overtime_attrition(df_raw), use_container_width=True)

    with tabs[5]:
        section_header("Satisfaction & Engagement Analysis", icon="🌡️")
        st.plotly_chart(EDA.satisfaction_radar(df_raw), use_container_width=True)
        # Bar breakdown of satisfaction scores
        sat_cols = ['JobSatisfaction', 'EnvironmentSatisfaction',
                    'RelationshipSatisfaction', 'WorkLifeBalance']
        c1, c2 = st.columns(2)
        for i, col in enumerate(sat_cols):
            target = c1 if i % 2 == 0 else c2
            with target:
                grp = df_raw.groupby([col, 'Attrition']).size().reset_index(name='Count')
                fig_s = go.Figure()
                for label, color in [('No','#10b981'),('Yes','#ef4444')]:
                    sub = grp[grp['Attrition']==label]
                    fig_s.add_trace(go.Bar(
                        x=sub[col], y=sub['Count'], name=label,
                        marker_color=color, opacity=0.85,
                    ))
                fig_s.update_layout(
                    barmode='group',
                    title=dict(text=col.replace('Satisfaction',' Satisfaction').replace('Balance',' Balance'),
                               font=dict(size=13, color='#f9fafb'), x=0.5),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.6)',
                    font=dict(color='#f9fafb', family='Inter'),
                    height=260, margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_s, use_container_width=True)

    with tabs[6]:
        section_header("Feature Correlation Matrix", icon="🗺️")
        st.plotly_chart(EDA.correlation_heatmap(df_raw), use_container_width=True)
        st.markdown("""
        <div class="glass-card">
            <b>📌 Key Correlations with Attrition:</b> Distance from home, 
            overtime, and lower job satisfaction are positively correlated 
            with attrition. Higher monthly income, tenure, and years at company 
            are negatively correlated (protective factors).
        </div>
        """, unsafe_allow_html=True)

    with tabs[7]:
        section_header("Descriptive Statistics", icon="📋")
        numeric_df = df_raw.select_dtypes(include='number')
        st.dataframe(
            numeric_df.describe().round(2).T
            .style.background_gradient(cmap='RdYlGn', axis=1),
            use_container_width=True,
            height=500,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS — PHASE 2
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  ML Models — Phase 2":
    st.markdown("""
    <div class="hero-banner animate-in">
        <h1 class="hero-title">Machine Learning Models</h1>
        <p class="hero-sub">Phase 2 — Six classifiers trained, evaluated & compared on IBM HR data</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Training all models… (first load may take ~30s)"):
        results = ML.train_all_models(
            X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, feature_names
        )

    # ── Top metrics row ─────────────────────────────────────────────────────
    best_acc  = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1   = max(results.items(), key=lambda x: x[1]['f1'])
    best_auc  = max(results.items(), key=lambda x: x[1]['auc'] or 0)
    best_rec  = max(results.items(), key=lambda x: x[1]['recall'])

    section_header("Best Performing Models", badge="Auto-Selected", icon="🏆")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Best Accuracy", f"{best_acc[1]['accuracy']:.1f}%",
                       delta=best_acc[0], delta_type="down", icon="🎯")
    with c2: kpi_card("Best F1 Score", f"{best_f1[1]['f1']:.1f}%",
                       delta=best_f1[0], delta_type="down", icon="⚖️")
    with c3: kpi_card("Best AUC-ROC",  f"{best_auc[1]['auc']:.1f}%",
                       delta=best_auc[0], delta_type="down", icon="📈")
    with c4: kpi_card("Best Recall",   f"{best_rec[1]['recall']:.1f}%",
                       delta=best_rec[0], delta_type="down", icon="🔍")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs for each model ─────────────────────────────────────────────────
    tabs = st.tabs(["📊 Comparison", "📈 ROC Curves", "🎯 Random Forest",
                    "🔬 Logistic Reg", "🌳 Decision Tree",
                    "🔷 SVM", "🔵 KNN", "📐 Naive Bayes"])

    with tabs[0]:
        section_header("Model Performance Comparison", icon="📊")
        df_metrics = ML.metrics_comparison_df(results)

        # Style the table
        def color_val(val):
            if isinstance(val, float):
                if val >= 85: return 'color: #10b981; font-weight: 700'
                if val >= 70: return 'color: #f59e0b; font-weight: 600'
                return 'color: #ef4444'
            return ''

        st.dataframe(
            df_metrics.style.map(color_val).format("{:.1f}", subset=df_metrics.select_dtypes('float').columns),
            use_container_width=True,
            height=260,
        )
        st.plotly_chart(ML.radar_comparison_chart(results), use_container_width=True)

    with tabs[1]:
        section_header("ROC Curves — All Models", icon="📈")
        st.plotly_chart(ML.roc_curves_fig(results, y_test), use_container_width=True)
        st.info("💡 Higher AUC-ROC = better model discrimination between attrited and retained employees. "
                "An AUC of 1.0 = perfect; 0.5 = random guessing.")

    def model_detail_tab(name):
        r = results[name]
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Accuracy",  f"{r['accuracy']}%", icon="🎯")
        with c2: kpi_card("Precision", f"{r['precision']}%", icon="📌")
        with c3: kpi_card("Recall",    f"{r['recall']}%", icon="🔍")
        with c4: kpi_card("F1 Score",  f"{r['f1']}%", icon="⚖️")
        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.plotly_chart(ML.confusion_matrix_fig(r['cm'], name), use_container_width=True)
        with col_right:
            st.markdown("**Classification Report**")
            st.code(r['report'], language='text')

    with tabs[2]:
        section_header("Random Forest — Production Model", icon="🎯")
        model_detail_tab('Random Forest')
        st.plotly_chart(
            ML.feature_importance_fig(results['Random Forest']['model'], feature_names),
            use_container_width=True
        )

    with tabs[3]:
        section_header("Logistic Regression", icon="🔬")
        model_detail_tab('Logistic Regression')

    with tabs[4]:
        section_header("Decision Tree", icon="🌳")
        model_detail_tab('Decision Tree')

    with tabs[5]:
        section_header("Support Vector Machine (SVM)", icon="🔷")
        model_detail_tab('SVM')

    with tabs[6]:
        section_header("K-Nearest Neighbors", icon="🔵")
        model_detail_tab('KNN')

    with tabs[7]:
        section_header("Naive Bayes", icon="📐")
        model_detail_tab('Naive Bayes')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DECISION TREE RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌳  Decision Tree Rules":
    st.markdown("""
    <div class="hero-banner animate-in">
        <h1 class="hero-title">HR Decision Tree Rules</h1>
        <p class="hero-sub">Interpretable IF-THEN rules automatically extracted from the Decision Tree model</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training Decision Tree…"):
        results = ML.train_all_models(
            X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, feature_names
        )

    dt_model = results['Decision Tree']['model']
    rules_text = ML.get_decision_rules(dt_model, feature_names)

    section_header("Top IF-THEN Decision Rules", badge="Auto-Generated", icon="📜")

    # Highlight example rules
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(239,68,68,0.4);background:rgba(239,68,68,0.05)">
            <div style="color:#ef4444;font-weight:700;font-size:1rem;margin-bottom:0.75rem">🚨 HIGH RISK Rule</div>
            <code style="color:#fca5a5;font-size:0.88rem;line-height:1.9">
            IF JobSatisfaction ≤ 1<br>
            AND OverTime = Yes<br>
            AND MonthlyIncome ≤ 4000<br>
            ──────────────────<br>
            THEN Attrition = <b>HIGH</b>
            </code>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(245,158,11,0.4);background:rgba(245,158,11,0.05)">
            <div style="color:#f59e0b;font-weight:700;font-size:1rem;margin-bottom:0.75rem">⚠️ MEDIUM RISK Rule</div>
            <code style="color:#fcd34d;font-size:0.88rem;line-height:1.9">
            IF YearsAtCompany ≤ 2<br>
            AND JobLevel ≤ 1<br>
            AND DistanceFromHome > 15<br>
            ──────────────────<br>
            THEN Attrition = <b>MEDIUM</b>
            </code>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card" style="border-color:rgba(16,185,129,0.4);background:rgba(16,185,129,0.05)">
            <div style="color:#10b981;font-weight:700;font-size:1rem;margin-bottom:0.75rem">✅ LOW RISK Rule</div>
            <code style="color:#6ee7b7;font-size:0.88rem;line-height:1.9">
            IF TotalWorkingYears > 10<br>
            AND JobSatisfaction ≥ 3<br>
            AND StockOptionLevel ≥ 1<br>
            ──────────────────<br>
            THEN Attrition = <b>LOW</b>
            </code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Visual Decision Tree (max_depth=4)", icon="🌳")

    # ── Matplotlib visual tree ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.tree import plot_tree

    fig_tree, ax = plt.subplots(figsize=(28, 12))
    fig_tree.patch.set_facecolor('#0d1321')
    ax.set_facecolor('#0d1321')

    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=['Retained', 'Attrited'],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        impurity=False,
        proportion=False,
        precision=2,
    )

    # Re-colour the boxes: leaf nodes green (retained) or red (attrited), splits = indigo
    for artist in ax.get_children():
        if hasattr(artist, 'get_facecolor'):
            fc = artist.get_facecolor()
            # plot_tree colours nodes by class; darken/theme them
            if hasattr(artist, 'set_facecolor'):
                r, g, b, a = fc
                # Reddish → attrition node
                if r > 0.6 and g < 0.5:
                    artist.set_facecolor('#5c1a1a')
                    artist.set_edgecolor('#ef4444')
                # Greenish → retained node
                elif g > 0.5 and r < 0.6:
                    artist.set_facecolor('#0e3d2d')
                    artist.set_edgecolor('#10b981')
                else:
                    artist.set_facecolor('#1e2a4a')
                    artist.set_edgecolor('#6366f1')

    # Style all text white
    for text in ax.texts:
        text.set_color('#f9fafb')
        text.set_fontfamily('monospace')

    ax.set_title('HR Attrition Decision Tree  (max_depth = 4)',
                 color='#f9fafb', fontsize=14, fontweight='bold', pad=12)

    # Legend patches
    patches = [
        mpatches.Patch(facecolor='#0e3d2d', edgecolor='#10b981', label='Retained (class 0)'),
        mpatches.Patch(facecolor='#5c1a1a', edgecolor='#ef4444', label='Attrited  (class 1)'),
        mpatches.Patch(facecolor='#1e2a4a', edgecolor='#6366f1', label='Split node'),
    ]
    ax.legend(handles=patches, loc='lower right', framealpha=0.3,
              facecolor='#111827', edgecolor='#6366f1',
              labelcolor='#f9fafb', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_tree, use_container_width=True)
    plt.close(fig_tree)

    # ── Text rules as collapsible fallback ─────────────────────────────────
    with st.expander("📄 View raw text rules", expanded=False):
        st.code(rules_text, language='text')

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Model Interpretability", icon="🔎")
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    fig = go.Figure(go.Bar(
        y=feat_imp['Feature'][::-1], x=feat_imp['Importance'][::-1],
        orientation='h',
        marker=dict(color=['#ef4444' if i < 3 else '#6366f1'
                           for i in range(len(feat_imp)-1, -1, -1)]),
        text=[f"{v:.4f}" for v in feat_imp['Importance'][::-1]],
        textposition='outside',
        textfont=dict(color='#f9fafb'),
    ))
    fig.update_layout(
        title=dict(text='Decision Tree Feature Importance', font=dict(size=15, color='#f9fafb'), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.6)',
        font=dict(color='#f9fafb', family='Inter'),
        height=380, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT EMPLOYEE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict Employee":
    st.markdown("""
    <div class="hero-banner animate-in">
        <h1 class="hero-title">Employee Attrition Risk Predictor</h1>
        <p class="hero-sub">
            Enter employee details below to get real-time attrition risk scoring 
            from 6 ML models combined into an ensemble prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        results = ML.train_all_models(
            X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, feature_names
        )

    section_header("Employee Profile", icon="👤")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age             = st.slider("Age", 18, 65, 35)
            monthly_income  = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
            daily_rate      = st.slider("Daily Rate", 100, 1500, 800)
            distance_home   = st.slider("Distance from Home (km)", 1, 30, 10)
            education       = st.selectbox("Education Level", [1,2,3,4,5],
                                           format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
        with c2:
            job_satisfaction    = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
            env_satisfaction    = st.slider("Environment Satisfaction", 1, 4, 3)
            work_life_balance   = st.slider("Work-Life Balance", 1, 4, 3)
            job_involvement     = st.slider("Job Involvement", 1, 4, 3)
            rel_satisfaction    = st.slider("Relationship Satisfaction", 1, 4, 3)
        with c3:
            years_at_company    = st.slider("Years at Company", 0, 40, 5)
            total_working_yrs   = st.slider("Total Working Years", 0, 40, 8)
            years_curr_role     = st.slider("Years in Current Role", 0, 20, 3)
            yrs_since_promo     = st.slider("Years Since Last Promotion", 0, 15, 2)
            yrs_with_mgr        = st.slider("Years with Current Manager", 0, 18, 4)

        c4, c5, c6 = st.columns(3)
        with c4:
            overtime        = st.selectbox("Overtime", ["No", "Yes"])
            business_travel = st.selectbox("Business Travel",
                                           ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
            department      = st.selectbox("Department",
                                           ["Sales", "Research & Development", "Human Resources"])
        with c5:
            job_role        = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])
            gender          = st.selectbox("Gender", ["Male", "Female"])
            marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with c6:
            education_field = st.selectbox("Education Field", [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"
            ])
            job_level           = st.selectbox("Job Level", [1,2,3,4,5])
            stock_option        = st.selectbox("Stock Option Level", [0,1,2,3])
            num_companies       = st.slider("Num Companies Worked", 0, 9, 2)
            training_times      = st.slider("Training Times Last Year", 0, 6, 2)

        perf_rating     = st.selectbox("Performance Rating", [3, 4],
                                       format_func=lambda x: "Excellent (3)" if x==3 else "Outstanding (4)")
        percent_hike    = st.slider("Percent Salary Hike (%)", 11, 25, 14)
        hourly_rate     = st.slider("Hourly Rate", 30, 100, 65)
        monthly_rate    = st.number_input("Monthly Rate", 2000, 27000, 14000, 500)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        # Build feature vector matching preprocessing order
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        # Map categoricals (same encoding as data_loader)
        enc_map = {
            'BusinessTravel': {'Non-Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2},
            'Department':     {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2},
            'EducationField': {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2,
                               'Medical': 3, 'Other': 4, 'Technical Degree': 5},
            'Gender':         {'Female': 0, 'Male': 1},
            'JobRole':        {'Healthcare Representative': 0, 'Human Resources': 1,
                               'Laboratory Technician': 2, 'Manager': 3,
                               'Manufacturing Director': 4, 'Research Director': 5,
                               'Research Scientist': 6, 'Sales Executive': 7,
                               'Sales Representative': 8},
            'MaritalStatus':  {'Divorced': 0, 'Married': 1, 'Single': 2},
            'OverTime':       {'No': 0, 'Yes': 1},
        }

        feat_vals = {
            'Age': age, 'BusinessTravel': enc_map['BusinessTravel'].get(business_travel, 0),
            'DailyRate': daily_rate, 'Department': enc_map['Department'].get(department, 1),
            'DistanceFromHome': distance_home, 'Education': education,
            'EducationField': enc_map['EducationField'].get(education_field, 1),
            'EnvironmentSatisfaction': env_satisfaction, 'Gender': enc_map['Gender'].get(gender, 1),
            'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': enc_map['JobRole'].get(job_role, 6),
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': enc_map['MaritalStatus'].get(marital_status, 1),
            'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies,
            'OverTime': enc_map['OverTime'].get(overtime, 0),
            'PercentSalaryHike': percent_hike, 'PerformanceRating': perf_rating,
            'RelationshipSatisfaction': rel_satisfaction, 'StockOptionLevel': stock_option,
            'TotalWorkingYears': total_working_yrs,
            'TrainingTimesLastYear': training_times, 'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_curr_role,
            'YearsSinceLastPromotion': yrs_since_promo, 'YearsWithCurrManager': yrs_with_mgr,
        }

        # Build ordered array matching feature_names
        emp_arr = np.array([feat_vals.get(f, 0) for f in feature_names], dtype=float)

        pred_result = ML.predict_employee(results, scaler, feature_names, emp_arr)
        ens_prob   = pred_result['ensemble_prob']
        risk_level = pred_result['risk_level']

        risk_class = {'HIGH':'high','MEDIUM':'medium','LOW':'low'}[risk_level]
        risk_icon  = {'HIGH':'🚨','MEDIUM':'⚠️','LOW':'✅'}[risk_level]
        risk_color = {'HIGH':'#ef4444','MEDIUM':'#f59e0b','LOW':'#10b981'}[risk_level]

        section_header("Prediction Results", icon="🎯")

        st.markdown(f"""
        <div class="pred-result animate-in">
            <div class="pred-prob {risk_class}">{ens_prob:.1f}%</div>
            <div style="font-size:1.1rem;color:#6b7280;margin-bottom:1rem">Ensemble Attrition Probability</div>
            <div class="risk-badge risk-{risk_class} pulse-badge">
                {risk_icon} {risk_level} ATTRITION RISK
            </div>
            <div style="font-size:0.85rem;color:#6b7280;margin-top:1rem">
                Based on 6 ML models: Logistic Regression · Decision Tree · 
                Random Forest · SVM · KNN · Naive Bayes
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ens_prob,
            number=dict(suffix="%", font=dict(color='#f9fafb', size=32)),
            delta=dict(reference=16.1, valueformat=".1f",
                       increasing=dict(color='#ef4444'),
                       decreasing=dict(color='#10b981')),
            title=dict(text="vs Dataset Baseline (16.1%)", font=dict(color='#6b7280', size=13)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='rgba(255,255,255,0.3)',
                          tickwidth=1),
                bar=dict(color=risk_color),
                bgcolor='rgba(17,24,39,0.6)',
                bordercolor='rgba(255,255,255,0.1)',
                steps=[
                    dict(range=[0, 35],  color='rgba(16,185,129,0.15)'),
                    dict(range=[35, 60], color='rgba(245,158,11,0.15)'),
                    dict(range=[60, 100],color='rgba(239,68,68,0.15)'),
                ],
                threshold=dict(line=dict(color='white',width=2), value=ens_prob)
            ),
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb', family='Inter'),
            height=300, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Individual model results
        section_header("Individual Model Predictions", icon="🤖")
        cols = st.columns(3)
        for i, (name, pred) in enumerate(pred_result['individual'].items()):
            with cols[i % 3]:
                prob_val = pred['probability'] or 0
                pred_color = '#ef4444' if pred['prediction'] == 'Yes' else '#10b981'
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;padding:1rem">
                    <div style="font-size:0.8rem;color:#6b7280;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.06em;margin-bottom:0.4rem">{name}</div>
                    <div style="font-size:1.8rem;font-weight:800;color:{pred_color}">{prob_val:.0f}%</div>
                    <div style="font-size:0.85rem;color:{pred_color};font-weight:600">
                        {"🚨 Will Leave" if pred['prediction']=='Yes' else "✅ Will Stay"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Recommendation
        st.markdown("<br>", unsafe_allow_html=True)
        if risk_level == 'HIGH':
            st.error(f"""
            🚨 **Immediate Action Required!** This employee has a {ens_prob:.1f}% probability of leaving.
            
            **Recommended HR Actions:**
            - 🤝 Schedule a 1-on-1 retention conversation immediately
            - 💰 Review compensation and benchmark against market rates
            - 📈 Discuss career growth path and promotion timeline
            - 🏖️ Evaluate workload and overtime hours
            - 🎁 Consider special retention incentives or benefits
            """)
        elif risk_level == 'MEDIUM':
            st.warning(f"""
            ⚠️ **Moderate Attrition Risk** — {ens_prob:.1f}% probability. Monitor closely.
            
            **Recommended HR Actions:**
            - 📅 Schedule quarterly check-ins to gauge satisfaction
            - 🎯 Assign meaningful projects to increase engagement
            - 💼 Explore internal mobility or cross-functional opportunities
            - 🧑‍🏫 Offer skills development and training programs
            """)
        else:
            st.success(f"""
            ✅ **Low Attrition Risk** — {ens_prob:.1f}% probability. Employee is likely to stay.
            
            **Recommended HR Actions:**
            - 🌟 Recognize and reward performance to sustain engagement
            - 📊 Continue regular career conversations
            - 🔑 Consider for leadership development pipeline
            """)

        # Similar employees
        section_header("Similar Employee Profiles (KNN)", icon="👥")
        knn_model = results['KNN']['model']
        similar = ML.find_similar_employees(knn_model, X_test_sc, df_raw, emp_arr, n=5)
        st.dataframe(
            similar.style.background_gradient(subset=['Similarity (%)'], cmap='Blues'),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EMPLOYEE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧩  Employee Segmentation":
    st.markdown("""
    <div class="hero-banner animate-in">
        <h1 class="hero-title">Employee Segmentation</h1>
        <p class="hero-sub">K-Means Clustering + PCA Dimensionality Reduction — Identify employee cohorts</p>
    </div>
    """, unsafe_allow_html=True)

    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3)

    with st.spinner("Running PCA + K-Means clustering…"):
        X_all_sc = np.vstack([X_train_sc, X_test_sc])
        X_pca, labels_2d, pca_model, kmeans_model, explained_var_2d = CLU.run_pca_kmeans(X_all_sc, n_clusters)
        X_pca3, labels_3d, explained_var_3d = CLU.run_pca_3d(X_all_sc, n_clusters)

    # Cluster cards
    section_header("Employee Clusters", badge="K-Means", icon="🧩")
    cluster_names_local  = ['🏆 High Performers', '⚠️ At Risk', '🌱 New Employees',
                             '🎖️ Veterans', '💡 Rising Stars', '🔄 Transitional']
    cluster_colors_local = ['#6366f1', '#f59e0b', '#ef4444', '#10b981', '#a78bfa', '#fb923c']
    cluster_desc_local   = [
        'Experienced, satisfied, high-income employees.',
        'Mid-tenure employees with warning signs.',
        'Early-career employees adapting to role.',
        'Long-tenured senior leaders.',
        'High-potential employees on the rise.',
        'Employees in career transition phase.',
    ]

    cols_c = st.columns(min(n_clusters, 3))
    cluster_sizes = [int((labels_2d == i).sum()) for i in range(n_clusters)]
    for i in range(n_clusters):
        with cols_c[i % 3]:
            color = cluster_colors_local[i % len(cluster_colors_local)]
            st.markdown(f"""
            <div class="cluster-card c{i % 3}">
                <div class="cluster-title" style="color:{color}">
                    {cluster_names_local[i % len(cluster_names_local)]}
                </div>
                <div style="font-size:1.6rem;font-weight:800;color:#f9fafb;margin:0.3rem 0">
                    {cluster_sizes[i]:,}
                </div>
                <div style="font-size:0.75rem;color:#6b7280;font-weight:600;text-transform:uppercase;margin-bottom:0.4rem">
                    employees
                </div>
                <div class="cluster-desc">{cluster_desc_local[i % len(cluster_desc_local)]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["🗺️ 2D PCA Scatter", "🌐 3D PCA Scatter", "📊 Cluster Profiles"])

    with tab1:
        # Build figure with dynamic cluster count
        fig_2d = go.Figure()
        for cid in range(n_clusters):
            mask = labels_2d == cid
            fig_2d.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1], mode='markers',
                name=cluster_names_local[cid % len(cluster_names_local)],
                marker=dict(color=cluster_colors_local[cid % len(cluster_colors_local)],
                            size=5, opacity=0.7),
            ))
        fig_2d.update_layout(
            title=dict(
                text=f'PCA 2D Segmentation (PC1={explained_var_2d[0]*100:.1f}%, PC2={explained_var_2d[1]*100:.1f}%)',
                font=dict(size=16, color='#f9fafb'), x=0.5
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.6)',
            font=dict(color='#f9fafb', family='Inter'), height=480,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(bgcolor='rgba(17,24,39,0.8)', bordercolor='rgba(99,102,241,0.3)', borderwidth=1),
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    with tab2:
        fig_3d = go.Figure()
        for cid in range(n_clusters):
            mask = labels_3d == cid
            fig_3d.add_trace(go.Scatter3d(
                x=X_pca3[mask, 0], y=X_pca3[mask, 1], z=X_pca3[mask, 2],
                mode='markers', name=cluster_names_local[cid % len(cluster_names_local)],
                marker=dict(color=cluster_colors_local[cid % len(cluster_colors_local)],
                            size=3, opacity=0.7),
            ))
        fig_3d.update_layout(
            title=dict(text='3D PCA Segmentation', font=dict(size=16, color='#f9fafb'), x=0.5),
            scene=dict(
                bgcolor='rgba(17,24,39,0.6)',
                xaxis=dict(title=f'PC1 ({explained_var_3d[0]*100:.1f}%)', color='#f9fafb'),
                yaxis=dict(title=f'PC2 ({explained_var_3d[1]*100:.1f}%)', color='#f9fafb'),
                zaxis=dict(title=f'PC3 ({explained_var_3d[2]*100:.1f}%)', color='#f9fafb'),
            ),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f9fafb', family='Inter'), height=540,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        section_header("Cluster Profiles", icon="📊")
        summary = CLU.get_cluster_summary(labels_2d, df_raw)
        st.dataframe(
            summary.style.background_gradient(cmap='RdYlGn_r', subset=['Attrition Rate (%)'] if 'Attrition Rate (%)' in summary.columns else []),
            use_container_width=True,
        )
        st.plotly_chart(CLU.cluster_profile_bars(X_pca, labels_2d, df_raw), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Dataset Explorer":
    st.markdown("""
    <div class="hero-banner animate-in">
        <h1 class="hero-title">Dataset Explorer</h1>
        <p class="hero-sub">Browse, filter and analyze the IBM HR Analytics Employee Attrition dataset</p>
    </div>
    """, unsafe_allow_html=True)

    section_header("Filters", icon="🔍")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dept_filter = st.multiselect("Department",
                                      df_raw['Department'].unique().tolist(),
                                      default=df_raw['Department'].unique().tolist())
    with col2:
        att_filter = st.multiselect("Attrition", ["Yes", "No"], default=["Yes", "No"])
    with col3:
        gender_filter = st.multiselect("Gender",
                                        df_raw['Gender'].unique().tolist(),
                                        default=df_raw['Gender'].unique().tolist())
    with col4:
        income_range = st.slider("Monthly Income Range ($)",
                                  int(df_raw['MonthlyIncome'].min()),
                                  int(df_raw['MonthlyIncome'].max()),
                                  (int(df_raw['MonthlyIncome'].min()),
                                   int(df_raw['MonthlyIncome'].max())))

    df_filtered = df_raw[
        df_raw['Department'].isin(dept_filter) &
        df_raw['Attrition'].isin(att_filter) &
        df_raw['Gender'].isin(gender_filter) &
        df_raw['MonthlyIncome'].between(*income_range)
    ]

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Filtered Records", f"{len(df_filtered):,}", icon="📋")
    with c2: kpi_card("Attrition Rate",
                       f"{(df_filtered['Attrition']=='Yes').mean()*100:.1f}%", icon="📈")
    with c3: kpi_card("Avg Monthly Income",
                       f"${df_filtered['MonthlyIncome'].mean():,.0f}", icon="💰")

    st.markdown("<br>", unsafe_allow_html=True)

    col_select = st.multiselect(
        "Select columns to display",
        df_raw.columns.tolist(),
        default=['Age', 'Department', 'JobRole', 'Gender', 'Attrition',
                 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany',
                 'OverTime', 'MaritalStatus']
    )

    def highlight_attrition(row):
        if row.get('Attrition') == 'Yes':
            return ['background-color: rgba(239,68,68,0.08)'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df_filtered[col_select].style.apply(highlight_attrition, axis=1),
        use_container_width=True,
        height=500,
    )

    col_dl, _ = st.columns([1, 4])
    with col_dl:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Download Filtered CSV",
            csv, "filtered_employees.csv", "text/csv",
            use_container_width=True,
        )
