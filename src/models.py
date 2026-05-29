"""
Machine Learning models module.
Trains and evaluates all 6 classifiers from the assignment.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import plotly.graph_objects as go
import plotly.figure_factory as ff

TEXT    = '#f9fafb'
ACCENT1 = '#6366f1'
ACCENT2 = '#f59e0b'
DANGER  = '#ef4444'
SUCCESS = '#10b981'
MUTED   = '#6b7280'
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.6)',
    font=dict(color=TEXT, family='Inter, sans-serif'),
    margin=dict(l=20, r=20, t=40, b=20),
)


@st.cache_resource(show_spinner=False)
def train_all_models(X_train, X_test, y_train, y_test,
                     X_train_scaled, X_test_scaled, feature_names):
    """Train all 6 classifiers and return results dict."""
    results = {}

    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            X_train_scaled, X_test_scaled
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),
            X_train, X_test
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight='balanced', n_jobs=-1),
            X_train, X_test
        ),
        'SVM': (
            SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            X_train_scaled, X_test_scaled
        ),
        'KNN': (
            KNeighborsClassifier(n_neighbors=7),
            X_train_scaled, X_test_scaled
        ),
        'Naive Bayes': (
            GaussianNB(),
            X_train_scaled, X_test_scaled
        ),
    }

    for name, (clf, Xtr, Xte) in models.items():
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1] if hasattr(clf, 'predict_proba') else None

        results[name] = {
            'model': clf,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'X_train': Xtr,
            'X_test': Xte,
            'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            'recall':    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            'f1':        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            'auc':       round(roc_auc_score(y_test, y_prob) * 100, 2) if y_prob is not None else None,
            'cm':        confusion_matrix(y_test, y_pred),
            'report':    classification_report(y_test, y_pred, target_names=['Retained', 'Attrited']),
        }

    return results


def metrics_comparison_df(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'Accuracy (%)': r['accuracy'],
            'Precision (%)': r['precision'],
            'Recall (%)': r['recall'],
            'F1 Score (%)': r['f1'],
            'AUC (%)': r['auc'] if r['auc'] else '—',
        })
    return pd.DataFrame(rows).set_index('Model')


def radar_comparison_chart(results: dict) -> go.Figure:
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    colors = [ACCENT1, ACCENT2, DANGER, SUCCESS, '#a78bfa', '#fb923c']

    fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        vals = [
            r['accuracy'], r['precision'], r['recall'],
            r['f1'], r['auc'] if r['auc'] else 0
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=name,
            fill='toself',
            opacity=0.65,
            line_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(17,24,39,0.6)',
            radialaxis=dict(visible=True, range=[50, 100], color=MUTED,
                            tickfont=dict(color=MUTED)),
            angularaxis=dict(color=TEXT),
        ),
        title=dict(text='Model Performance Comparison', font=dict(size=16, color=TEXT), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, family='Inter, sans-serif'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=40, r=40, t=60, b=20),
        height=460,
    )
    return fig


def confusion_matrix_fig(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ['Retained', 'Attrited']
    text = [[str(cm[i][j]) for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm[::-1],
        x=labels,
        y=labels[::-1],
        colorscale=[[0, '#111827'], [1, ACCENT1]],
        showscale=False,
        text=text[::-1],
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=22, color=TEXT),
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=f'Confusion Matrix — {model_name}',
                   font=dict(size=14, color=TEXT), x=0.5),
        xaxis_title='Predicted', yaxis_title='Actual',
        **PLOTLY_LAYOUT,
        height=320,
    )
    return fig


def roc_curves_fig(results: dict, y_test: np.ndarray) -> go.Figure:
    from sklearn.metrics import roc_curve
    colors = [ACCENT1, ACCENT2, DANGER, SUCCESS, '#a78bfa', '#fb923c']
    fig = go.Figure()
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=MUTED, dash='dot', width=1))
    for i, (name, r) in enumerate(results.items()):
        if r['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} ({r['auc']:.1f}%)",
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
            ))
    fig.update_layout(
        title=dict(text='ROC Curves — All Models', font=dict(size=16, color=TEXT), x=0.5),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        **PLOTLY_LAYOUT,
        height=420,
        legend=dict(bgcolor='rgba(17,24,39,0.8)', bordercolor='rgba(99,102,241,0.3)',
                    borderwidth=1),
    )
    return fig


def feature_importance_fig(rf_model, feature_names: list) -> go.Figure:
    importances = rf_model.feature_importances_
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df = df.sort_values('Importance', ascending=True).tail(15)

    colors = [DANGER if v > df['Importance'].quantile(0.75) else ACCENT1
              for v in df['Importance']]
    fig = go.Figure(go.Bar(
        y=df['Feature'], x=df['Importance'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.3f}" for v in df['Importance']],
        textposition='outside',
        textfont=dict(color=TEXT),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Top Feature Importances (Random Forest)',
                   font=dict(size=16, color=TEXT), x=0.5),
        xaxis_title='Importance Score',
        **PLOTLY_LAYOUT,
        height=460,
    )
    return fig


def get_decision_rules(dt_model, feature_names: list) -> str:
    """Extract human-readable decision rules from Decision Tree."""
    return export_text(dt_model, feature_names=feature_names, max_depth=4)


def predict_employee(models_dict: dict, scaler, feature_names: list,
                     employee_features: np.ndarray) -> dict:
    """
    Run all models on a single employee's features.
    Returns dict with predictions and probabilities.
    """
    emp_scaled = scaler.transform(employee_features.reshape(1, -1))

    predictions = {}
    for name, r in models_dict.items():
        model = r['model']
        # Determine which input to use (scaled vs raw)
        if name in ['Decision Tree', 'Random Forest']:
            inp = employee_features.reshape(1, -1)
        else:
            inp = emp_scaled

        pred = model.predict(inp)[0]
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(inp)[0][1]

        predictions[name] = {
            'prediction': 'Yes' if pred == 1 else 'No',
            'probability': round(prob * 100, 1) if prob is not None else None,
        }

    # Ensemble: average probability
    probs = [v['probability'] for v in predictions.values() if v['probability'] is not None]
    ensemble_prob = round(np.mean(probs), 1) if probs else 0

    risk = 'HIGH' if ensemble_prob >= 60 else 'MEDIUM' if ensemble_prob >= 35 else 'LOW'

    return {
        'individual': predictions,
        'ensemble_prob': ensemble_prob,
        'risk_level': risk,
    }


def find_similar_employees(knn_model, X_test_scaled: np.ndarray,
                            df_original: pd.DataFrame,
                            employee_scaled: np.ndarray, n: int = 5):
    """Find N most similar employees using KNN."""
    distances, indices = knn_model.kneighbors(
        employee_scaled.reshape(1, -1), n_neighbors=min(n, len(X_test_scaled))
    )
    # Map back to original df approximate rows
    similar = df_original.iloc[indices[0] % len(df_original)].copy()
    similar['Similarity (%)'] = [round((1 - d / (d + 1)) * 100, 1) for d in distances[0]]
    return similar[['Age', 'Department', 'JobRole', 'MonthlyIncome',
                     'YearsAtCompany', 'Attrition', 'Similarity (%)']]
