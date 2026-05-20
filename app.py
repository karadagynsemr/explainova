import os
import re
import html as html_lib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data, suggest_ordinal_columns
from src.model_training import (
    detect_problem_type,
    get_available_models,
    train_and_evaluate_models
)
from src.visualization import (
    plot_confusion_matrix_figure,
    plot_metric_grid,
    plot_model_leaderboard_figure,
    plot_roc_curve_figure,
    build_outlier_dataframe,
    build_target_correlation_table,
    plot_correlation_heatmap_figure,
    plot_correlation_profile_figure,
    plot_feature_behavior_summary_figure,
    get_confusion_matrix_interpretation,
    get_roc_interpretation,
    get_metric_commentary,
    get_model_recommendation_text,
    get_correlation_profile_interpretation
)
from src.explainability import (
    compute_shap_outputs,
    plot_shap_importance_bar,
    plot_shap_summary_figure,
    plot_shap_feature_effect_figure,
    compute_kfold_stability,
    plot_kfold_stability_figure,
    get_kfold_interpretation,
    compute_pdp_ice_data,
    plot_pdp_ice_figure,
    get_pdp_ice_interpretation,
    get_feature_effect_interpretation,
    get_shap_selection_guidance,
    get_shap_intro_text
)
from src.utils import generate_word_report

st.set_page_config(
    page_title="Explainova",
    layout="wide"
)

dark_mode = st.sidebar.toggle(
    "Dark mode",
    value=st.session_state.get("dark_mode", False),
    help="Switch the workspace to a darker visual theme."
)
st.session_state["dark_mode"] = dark_mode

dark_theme_css = """
    :root {
        --bg-main: #111827;
        --bg-soft: #172033;
        --card-bg: #162033;
        --card-border: #2D3B52;
        --text-main: #E5E7EB;
        --text-soft: #CBD5E1;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(139,92,246,0.18), transparent 25%),
            radial-gradient(circle at top left, rgba(16,185,129,0.12), transparent 24%),
            linear-gradient(180deg, #08111F 0%, #111827 100%) !important;
    }

    .hero-card {
        background: linear-gradient(135deg, #4C1D95 0%, #1D4ED8 55%, #047857 100%);
        border-color: rgba(255,255,255,0.14);
        box-shadow: 0 24px 54px rgba(0, 0, 0, 0.34);
    }

    .status-strip,
    .stepper-wrap,
    .section-box,
    .summary-card,
    .chart-frame,
    .metric-comment,
    .chart-note,
    .order-box,
    .download-card,
    div[data-testid="stMetric"],
    div[data-testid="stDataFrame"],
    .stExpander {
        background: linear-gradient(180deg, #162033 0%, #111827 100%) !important;
        border-color: #2D3B52 !important;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.24) !important;
    }

    [data-testid="stDataFrame"] {
        background: #111827 !important;
        border-color: #334155 !important;
    }

    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] *,
    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] [role="row"],
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
        border-color: #263449 !important;
    }

    .dark-table-wrap {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 16px;
        overflow: auto;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.20);
        margin: 0.35rem 0 1rem 0;
    }

    .dark-table-wrap table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: #111827 !important;
        color: #E5E7EB !important;
        font-size: 0.9rem;
    }

    .dark-table-wrap thead th {
        position: sticky;
        top: 0;
        z-index: 1;
        background: #0F172A !important;
        color: #CBD5E1 !important;
        border-bottom: 1px solid #334155 !important;
        border-right: 1px solid #263449 !important;
        padding: 0.72rem 0.75rem;
        text-align: left;
        white-space: nowrap;
    }

    .dark-table-wrap tbody th,
    .dark-table-wrap tbody td {
        background: #162033 !important;
        color: #E5E7EB !important;
        border-bottom: 1px solid #263449 !important;
        border-right: 1px solid #263449 !important;
        padding: 0.66rem 0.75rem;
        white-space: nowrap;
    }

    .dark-table-wrap tbody tr:nth-child(even) th,
    .dark-table-wrap tbody tr:nth-child(even) td {
        background: #111827 !important;
    }

    .dark-table-wrap tbody tr:hover th,
    .dark-table-wrap tbody tr:hover td {
        background: #1E293B !important;
    }

    .dark-table-wrap thead th:first-child {
        border-top-left-radius: 15px;
    }

    .dark-table-wrap thead th:last-child {
        border-top-right-radius: 15px;
    }

    .dark-table-wrap tbody tr:last-child th:first-child {
        border-bottom-left-radius: 15px;
    }

    .dark-table-wrap tbody tr:last-child td:last-child {
        border-bottom-right-radius: 15px;
    }

    .compact-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.34);
        color: #D1FAE5;
        border-radius: 12px;
        padding: 9px 12px;
        margin: 8px 0 10px 0;
        font-weight: 750;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
    }

    .compact-status::before {
        content: "✓";
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 999px;
        background: #10B981;
        color: #FFFFFF;
        font-size: 0.82rem;
        font-weight: 900;
    }

    .insight-box {
        background: linear-gradient(180deg, #17233A 0%, #111C2F 100%) !important;
        border-color: #2D3B52 !important;
    }

    .story-panel {
        background: linear-gradient(135deg, #16213A 0%, #111827 100%) !important;
        border-color: #334155 !important;
    }

    h1, h2, h3, h4, p, li, label, span, div {
        color: var(--text-main);
    }

    .sidebar-shell {
        background: linear-gradient(180deg, #162033 0%, #111827 100%) !important;
        border-color: #2D3B52 !important;
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.28) !important;
    }

    .sidebar-stage-card {
        background: rgba(17, 24, 39, 0.82) !important;
        border-color: #2D3B52 !important;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.24) !important;
    }

    .sidebar-stage-card.active {
        background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(59,130,246,0.14)) !important;
    }

    .sidebar-stage-index {
        background: #1E293B !important;
        color: #C4B5FD !important;
    }

    .sidebar-stage-card.ready .sidebar-stage-index {
        background: rgba(16,185,129,0.18) !important;
        color: #6EE7B7 !important;
    }

    .sidebar-stage-card.active .sidebar-stage-index {
        background: var(--accent-shap) !important;
        color: white !important;
    }

    .download-title,
    .insight-title,
    .story-text,
    div[data-testid="stMetricValue"],
    div[data-testid="stRadio"] label,
    div[data-testid="stDataFrame"],
    .summary-value {
        color: #E5E7EB !important;
    }

    .download-subtitle,
    .insight-text,
    .metric-comment,
    .chart-note,
    .summary-note,
    .section-subtitle,
    .status-strip,
    .story-title {
        color: #CBD5E1 !important;
    }

    .summary-label,
    div[data-testid="stMetricLabel"] {
        color: #A5B4FC !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1220 0%, #111827 100%) !important;
    }

    section[data-testid="stSidebar"],
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0B1220 0%, #111827 100%) !important;
    }

    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] div {
        color: #E5E7EB !important;
    }

    header[data-testid="stHeader"] {
        background: linear-gradient(180deg, #111827 0%, rgba(17,24,39,0.92) 100%) !important;
        border-bottom: 1px solid #1F2937 !important;
    }

    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] button span {
        color: #CBD5E1 !important;
        -webkit-text-fill-color: #CBD5E1 !important;
        opacity: 1 !important;
    }

    header[data-testid="stHeader"] button:has(span[class*="material"]),
    header[data-testid="stHeader"] [data-testid="collapsedControl"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22) !important;
    }

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        background: transparent !important;
    }

    section.main,
    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #08111F 0%, #111827 100%) !important;
    }

    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] > div,
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileName"] {
        background: #111827 !important;
        border-color: #334155 !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    [data-testid="stFileUploader"] button {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #111827 !important;
        background-color: #111827 !important;
    }

    [data-testid="stFileUploaderDropzone"] div {
        background-color: #111827 !important;
    }

    [data-testid="stFileUploaderDropzone"] div:has([data-testid="stFileUploaderFileName"]),
    [data-testid="stFileUploaderDropzone"] div:has([title$=".csv" i]),
    [data-testid="stFileUploaderDropzone"] div:has([title$=".xlsx" i]),
    [data-testid="stFileUploaderDropzone"] div:has([title$=".xls" i]),
    [data-testid="stFileUploaderDropzone"] div:has([title$=".tsv" i]) {
        background: #1E293B !important;
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    [data-testid="stFileUploader"] svg,
    [data-testid="stFileUploader"] svg * {
        color: #A5B4FC !important;
        fill: #A5B4FC !important;
        stroke: #A5B4FC !important;
    }

    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileData"] *,
    [data-testid="stFileUploaderFileName"] * {
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] > div,
    [data-testid="stFileUploaderFileData"] {
        background: #1E293B !important;
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }

    [data-testid="stFileUploaderFile"] {
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18) !important;
        overflow: hidden !important;
    }

    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileData"] *,
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileName"] * {
        background: transparent !important;
        background-color: transparent !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] [data-testid*="FileSize"],
    [data-testid="stFileUploaderFile"] [class*="fileSize"] {
        color: #AAB7CF !important;
        -webkit-text-fill-color: #AAB7CF !important;
    }

    [data-testid="stFileUploaderFile"] button {
        background: #334155 !important;
        border: 1px solid #64748B !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] > div,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #CBD5E1 !important;
        -webkit-text-fill-color: #CBD5E1 !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"],
    [data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"],
    [data-testid="stWidgetLabel"] button[aria-label*="help" i],
    [data-testid="stWidgetLabel"] button[aria-label*="Help" i],
    [data-testid="stMetricLabel"] [data-testid="stTooltipIcon"],
    [data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"],
    [data-testid="stMetricLabel"] button[aria-label*="help" i],
    [data-testid="stMetricLabel"] button[aria-label*="Help" i] {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: auto !important;
        height: auto !important;
        min-width: 16px !important;
        min-height: 16px !important;
        margin-left: 6px !important;
        border-radius: 999px !important;
        background: transparent !important;
        border: 0 !important;
        color: #CBD5E1 !important;
        -webkit-text-fill-color: #CBD5E1 !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] svg,
    [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] svg *,
    [data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"] svg,
    [data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"] svg *,
    [data-testid="stWidgetLabel"] button[aria-label*="help" i] svg,
    [data-testid="stWidgetLabel"] button[aria-label*="help" i] svg *,
    [data-testid="stWidgetLabel"] button[aria-label*="Help" i] svg,
    [data-testid="stWidgetLabel"] button[aria-label*="Help" i] svg *,
    [data-testid="stMetricLabel"] [data-testid="stTooltipIcon"] svg,
    [data-testid="stMetricLabel"] [data-testid="stTooltipIcon"] svg *,
    [data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"] svg,
    [data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"] svg *,
    [data-testid="stMetricLabel"] button[aria-label*="help" i] svg,
    [data-testid="stMetricLabel"] button[aria-label*="help" i] svg *,
    [data-testid="stMetricLabel"] button[aria-label*="Help" i] svg,
    [data-testid="stMetricLabel"] button[aria-label*="Help" i] svg * {
        color: #E5E7EB !important;
        fill: #E5E7EB !important;
        stroke: #E5E7EB !important;
        opacity: 1 !important;
    }

    button[aria-label^="Help for"] {
        position: relative !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 20px !important;
        height: 20px !important;
        min-width: 20px !important;
        min-height: 20px !important;
        margin-left: 6px !important;
        padding: 0 !important;
        border-radius: 999px !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
        opacity: 1 !important;
        overflow: visible !important;
    }

    button[aria-label^="Help for"] * {
        display: none !important;
    }

    button[aria-label^="Help for"]::before {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: 999px;
        background: #F8FAFC !important;
        border: 1px solid rgba(15, 23, 42, 0.24) !important;
        box-shadow: 0 0 0 2px rgba(248, 250, 252, 0.18) !important;
    }

    button[aria-label^="Help for"]::after {
        content: "?";
        position: absolute;
        inset: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-size: 0.74rem;
        font-weight: 900;
        line-height: 1;
    }

    button[aria-label^="Help for"]:hover::before {
        background: #FFFFFF !important;
        border-color: rgba(15, 23, 42, 0.36) !important;
    }

    [role="tooltip"],
    [role="tooltip"] *,
    [data-baseweb="tooltip"],
    [data-baseweb="tooltip"] *,
    [data-testid="stTooltipContent"],
    [data-testid="stTooltipContent"] *,
    div[data-baseweb="popover"][role="tooltip"],
    div[data-baseweb="popover"][role="tooltip"] * {
        background: #0F172A !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
        border-color: #334155 !important;
        opacity: 1 !important;
    }

    [role="tooltip"],
    [data-baseweb="tooltip"],
    [data-testid="stTooltipContent"] {
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.36) !important;
    }

    [data-testid="stAlert"],
    [data-testid="stAlert"] * {
        background: #10231D !important;
        border-color: #10B981 !important;
        color: #D1FAE5 !important;
        -webkit-text-fill-color: #D1FAE5 !important;
    }

    [data-testid="stAlert"] {
        border: 1px solid #10B981 !important;
        border-radius: 14px !important;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.32) !important;
    }

    .step-connector {
        background: #334155 !important;
    }

    .step-circle.pending {
        background: #1E293B !important;
        border-color: #334155 !important;
        color: #CBD5E1 !important;
    }

    .step-label {
        color: #94A3B8 !important;
    }

    div[data-testid="stSidebar"] .sidebar-caption,
    div[data-testid="stSidebar"] .sidebar-stage-status {
        color: #AAB7CF !important;
    }

    div[data-testid="stSidebar"] .sidebar-stage-label,
    div[data-testid="stSidebar"] .sidebar-title {
        color: #E5E7EB !important;
    }

    div[data-testid="stSidebar"] .sidebar-stage-index {
        background: #1E293B !important;
        color: #C4B5FD !important;
    }

    div[data-testid="stSidebar"] .sidebar-stage-card.ready .sidebar-stage-index {
        background: rgba(16,185,129,0.18) !important;
        color: #6EE7B7 !important;
    }

    div[data-testid="stSidebar"] .sidebar-stage-card.active .sidebar-stage-index {
        background: var(--accent-shap) !important;
        color: #FFFFFF !important;
    }

    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[aria-label*="sidebar" i],
    button[title*="sidebar" i] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #CBD5E1 !important;
        -webkit-text-fill-color: #CBD5E1 !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.20) !important;
        opacity: 1 !important;
    }

    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] svg *,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg *,
    button[aria-label*="sidebar" i] svg,
    button[aria-label*="sidebar" i] svg *,
    button[title*="sidebar" i] svg,
    button[title*="sidebar" i] svg * {
        color: #CBD5E1 !important;
        fill: #CBD5E1 !important;
        stroke: #CBD5E1 !important;
        opacity: 1 !important;
    }

    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="popover"]:has([role="listbox"]),
    div[data-baseweb="popover"] > div:has([role="listbox"]),
    .stMultiSelect div[data-baseweb="tag"],
    .stTextInput input,
    textarea,
    div[role="listbox"],
    ul[role="listbox"] {
        background: #111827 !important;
        border-color: #334155 !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    div[data-baseweb="popover"]:has([role="listbox"]),
    div[data-baseweb="popover"]:has([role="listbox"]) > div,
    div[data-baseweb="popover"]:has([role="listbox"]) div:not([role="option"]),
    div[data-baseweb="popover"]:has([role="listbox"]) ul,
    div[data-baseweb="popover"]:has([role="listbox"]) li,
    div[role="listbox"],
    div[role="listbox"] > div,
    ul[role="listbox"],
    ul[role="listbox"] > li {
        background: #111827 !important;
        background-color: #111827 !important;
        border-color: #334155 !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    .stSelectbox div[data-baseweb="select"] input,
    .stSelectbox div[data-baseweb="select"] textarea,
    .stSelectbox div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] input,
    .stMultiSelect div[data-baseweb="select"] textarea,
    .stMultiSelect div[data-baseweb="select"] span,
    .stTextInput input,
    textarea {
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    .stSelectbox div[data-baseweb="select"] [class*="singleValue"],
    .stSelectbox div[data-baseweb="select"] [class*="valueContainer"],
    .stSelectbox div[data-baseweb="select"] [class*="placeholder"],
    .stMultiSelect div[data-baseweb="select"] [class*="valueContainer"],
    .stMultiSelect div[data-baseweb="select"] [class*="placeholder"] {
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
        opacity: 1 !important;
    }

    .stSelectbox div[data-baseweb="select"] svg,
    .stSelectbox div[data-baseweb="select"] svg *,
    .stMultiSelect div[data-baseweb="select"] svg,
    .stMultiSelect div[data-baseweb="select"] svg * {
        color: #E5E7EB !important;
        fill: #E5E7EB !important;
        stroke: #E5E7EB !important;
        opacity: 1 !important;
    }

    div[role="option"],
    li[role="option"] {
        background: #111827 !important;
        background-color: #111827 !important;
        border: 0 !important;
        border-bottom: 1px solid #243247 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
        min-height: 42px !important;
        height: auto !important;
        width: auto !important;
        padding: 0.56rem 0.75rem !important;
    }

    div[role="option"] *,
    li[role="option"] * {
        background: transparent !important;
        background-color: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        color: #E5E7EB !important;
        -webkit-text-fill-color: #E5E7EB !important;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background: #312E81 !important;
        border: 1px solid #4F46E5 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    .stMultiSelect [data-baseweb="tag"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    div[role="option"]:hover,
    li[role="option"]:hover,
    div[role="option"][data-highlighted],
    li[role="option"][data-highlighted],
    div[role="option"][aria-selected="true"],
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"] div,
    li[role="option"][aria-selected="true"] div,
    div[role="option"][aria-selected="true"] span,
    li[role="option"][aria-selected="true"] span,
    div[role="option"][aria-selected="true"] *,
    li[role="option"][aria-selected="true"] * {
        background: #312E81 !important;
        background-color: #312E81 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    div[data-baseweb="select"] input::placeholder,
    .stTextInput input::placeholder,
    textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }
""" if dark_mode else ""

workflow_anchor_css = """
    .stApp.sidebar-scroll-sync .sidebar-stage-card.active:not(.scroll-active) {
        background: var(--card-bg) !important;
        border-color: var(--card-border) !important;
        border-left-color: #CBD5E1 !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06) !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.active.available:not(.scroll-active) {
        border-left-color: var(--accent-preprocessing) !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.active:not(.scroll-active) .sidebar-stage-index {
        background: #EEF2FF !important;
        color: #4338CA !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.active.available:not(.scroll-active) .sidebar-stage-index {
        background: #D1FAE5 !important;
        color: #047857 !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.active:not(.scroll-active) .sidebar-stage-status {
        font-size: 0 !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.active:not(.scroll-active) .sidebar-stage-status::after {
        content: "Ready";
        font-size: 0.68rem;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.scroll-active {
        background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(59,130,246,0.14)) !important;
        border-color: var(--card-border) !important;
        border-left-color: var(--accent-shap) !important;
        opacity: 1 !important;
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.18) !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.scroll-active .sidebar-stage-index {
        background: var(--accent-shap) !important;
        color: #FFFFFF !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.scroll-active .sidebar-stage-status {
        font-size: 0 !important;
    }

    .stApp.sidebar-scroll-sync .sidebar-stage-card.scroll-active .sidebar-stage-status::after {
        content: "Active";
        font-size: 0.68rem;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card.active,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card.active,
    .stApp:has(#model-training:target) .sidebar-stage-card.active,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card.active,
    .stApp:has(#download-report:target) .sidebar-stage-card.active {
        background: var(--card-bg) !important;
        border-color: var(--card-border) !important;
        border-left-color: #CBD5E1 !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06) !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card.active.available,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card.active.available,
    .stApp:has(#model-training:target) .sidebar-stage-card.active.available,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card.active.available,
    .stApp:has(#download-report:target) .sidebar-stage-card.active.available {
        border-left-color: var(--accent-preprocessing) !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card.active .sidebar-stage-index,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card.active .sidebar-stage-index,
    .stApp:has(#model-training:target) .sidebar-stage-card.active .sidebar-stage-index,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card.active .sidebar-stage-index,
    .stApp:has(#download-report:target) .sidebar-stage-card.active .sidebar-stage-index {
        background: #EEF2FF !important;
        color: #4338CA !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card.active.available .sidebar-stage-index,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card.active.available .sidebar-stage-index,
    .stApp:has(#model-training:target) .sidebar-stage-card.active.available .sidebar-stage-index,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card.active.available .sidebar-stage-index,
    .stApp:has(#download-report:target) .sidebar-stage-card.active.available .sidebar-stage-index {
        background: #D1FAE5 !important;
        color: #047857 !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card.active.available .sidebar-stage-status,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card.active.available .sidebar-stage-status,
    .stApp:has(#model-training:target) .sidebar-stage-card.active.available .sidebar-stage-status,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card.active.available .sidebar-stage-status,
    .stApp:has(#download-report:target) .sidebar-stage-card.active.available .sidebar-stage-status {
        font-size: 0 !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card.active.available .sidebar-stage-status::after,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card.active.available .sidebar-stage-status::after,
    .stApp:has(#model-training:target) .sidebar-stage-card.active.available .sidebar-stage-status::after,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card.active.available .sidebar-stage-status::after,
    .stApp:has(#download-report:target) .sidebar-stage-card.active.available .sidebar-stage-status::after {
        content: "Ready";
        font-size: 0.68rem;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card[href="#dataset-preview"],
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card[href="#preprocessing-options"],
    .stApp:has(#model-training:target) .sidebar-stage-card[href="#model-training"],
    .stApp:has(#shap-explainability:target) .sidebar-stage-card[href="#shap-explainability"],
    .stApp:has(#download-report:target) .sidebar-stage-card[href="#download-report"] {
        background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(59,130,246,0.14)) !important;
        border-color: var(--card-border) !important;
        border-left-color: var(--accent-shap) !important;
        opacity: 1 !important;
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.18) !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card[href="#dataset-preview"] .sidebar-stage-index,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card[href="#preprocessing-options"] .sidebar-stage-index,
    .stApp:has(#model-training:target) .sidebar-stage-card[href="#model-training"] .sidebar-stage-index,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card[href="#shap-explainability"] .sidebar-stage-index,
    .stApp:has(#download-report:target) .sidebar-stage-card[href="#download-report"] .sidebar-stage-index {
        background: var(--accent-shap) !important;
        color: #FFFFFF !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card[href="#dataset-preview"] .sidebar-stage-status,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card[href="#preprocessing-options"] .sidebar-stage-status,
    .stApp:has(#model-training:target) .sidebar-stage-card[href="#model-training"] .sidebar-stage-status,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card[href="#shap-explainability"] .sidebar-stage-status,
    .stApp:has(#download-report:target) .sidebar-stage-card[href="#download-report"] .sidebar-stage-status {
        font-size: 0 !important;
    }

    .stApp:has(#dataset-preview:target) .sidebar-stage-card[href="#dataset-preview"] .sidebar-stage-status::after,
    .stApp:has(#preprocessing-options:target) .sidebar-stage-card[href="#preprocessing-options"] .sidebar-stage-status::after,
    .stApp:has(#model-training:target) .sidebar-stage-card[href="#model-training"] .sidebar-stage-status::after,
    .stApp:has(#shap-explainability:target) .sidebar-stage-card[href="#shap-explainability"] .sidebar-stage-status::after,
    .stApp:has(#download-report:target) .sidebar-stage-card[href="#download-report"] .sidebar-stage-status::after {
        content: "Active";
        font-size: 0.68rem;
    }
"""

st.markdown("""
<style>
    :root {
        --bg-main: #F7F8FC;
        --bg-soft: #F1F5FB;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --text-main: #0F172A;
        --text-soft: #475569;
        --primary: #4F46E5;
        --primary-dark: #4338CA;
        --accent: #0EA5E9;
        --accent-model: #3B82F6;
        --accent-shap: #8B5CF6;
        --accent-preprocessing: #10B981;
        --accent-warning: #F59E0B;
        --success: #10B981;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(99,102,241,0.10), transparent 24%),
            radial-gradient(circle at top left, rgba(14,165,233,0.08), transparent 22%),
            linear-gradient(180deg, #FAFBFE 0%, #F3F6FB 100%);
        color: var(--text-main);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
        border-right: 1px solid var(--card-border);
    }

    div[data-testid="stSidebar"] section {
        padding-top: 1.1rem;
    }

    .sidebar-shell {
        background: linear-gradient(180deg, var(--card-bg) 0%, var(--bg-soft) 100%);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 14px;
        margin: 12px 0 16px 0;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.10);
    }

    .sidebar-title {
        font-size: 0.9rem;
        font-weight: 850;
        color: var(--text-main);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .sidebar-caption {
        font-size: 0.82rem;
        color: var(--text-soft);
        line-height: 1.45;
        margin-bottom: 12px;
    }

    .sidebar-stage-card {
        display: block;
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--card-border);
        border-left: 4px solid #CBD5E1;
        border-radius: 14px;
        padding: 10px 10px 9px 10px;
        margin-bottom: 9px;
        text-decoration: none !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
    }

    .sidebar-stage-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.10);
    }

    .sidebar-stage-card.active {
        border-left-color: var(--accent-shap);
        background: linear-gradient(135deg, rgba(139,92,246,0.14), rgba(59,130,246,0.10));
    }

    .sidebar-stage-card.ready {
        border-left-color: var(--accent-preprocessing);
    }

    .sidebar-stage-card.locked {
        opacity: 0.66;
    }

    .sidebar-stage-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }

    .sidebar-stage-label {
        color: var(--text-main);
        font-size: 0.9rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .sidebar-stage-status {
        color: var(--text-soft);
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .sidebar-stage-index {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
        background: #EEF2FF;
        color: #4338CA;
        font-size: 0.74rem;
        font-weight: 900;
    }

    .sidebar-stage-card.ready .sidebar-stage-index {
        background: #D1FAE5;
        color: #047857;
    }

    .sidebar-stage-card.active .sidebar-stage-index {
        background: var(--accent-shap);
        color: white;
    }

    .sidebar-stage-card.locked .sidebar-stage-index {
        background: #E2E8F0;
        color: #64748B;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        padding-left: 1.6rem;
        padding-right: 1.6rem;
    }

    @media (max-width: 1200px) {
        .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    h1, h2, h3, h4 {
        color: var(--text-main) !important;
        letter-spacing: -0.02em;
    }

    p, li, label, span, div {
        color: var(--text-main);
    }

    .hero-wrap {
        text-align: center;
        margin-bottom: 1.15rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 58%, #0EA5E9 100%);
        border-radius: 30px;
        padding: 34px 28px 30px 28px;
        margin-bottom: 8px;
        box-shadow: 0 20px 42px rgba(79, 70, 229, 0.18);
        border: 1px solid rgba(255,255,255,0.18);
    }

    .hero-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin-bottom: 0.55rem;
        text-align: center;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.96);
        font-size: 1.04rem;
        line-height: 1.7;
        max-width: 860px;
        margin: 0 auto;
        text-align: center;
    }

    .hero-pills {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .hero-pill {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.26);
        color: white;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 700;
        backdrop-filter: blur(10px);
    }

    .status-strip {
        background: rgba(255,255,255,0.78);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        font-size: 0.95rem;
        color: #334155;
    }

    .stepper-wrap {
        background: rgba(255,255,255,0.92);
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 20px 24px 14px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.05);
        backdrop-filter: blur(8px);
    }

    .stepper {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 7px;
        flex: 1;
        max-width: 190px;
    }

    .step-connector {
        flex: 1;
        height: 4px;
        max-width: 96px;
        border-radius: 999px;
        background: #E2E8F0;
        margin-bottom: 24px;
        transition: background 0.3s ease;
    }

    .step-connector.done {
        background: linear-gradient(90deg, #4F46E5, #22C55E);
    }

    .step-circle {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.92rem;
        font-weight: 800;
        transition: all 0.3s ease;
    }

    .step-circle.pending {
        background: #F8FAFC;
        border: 2px solid #CBD5E1;
        color: #94A3B8;
    }

    .step-circle.active {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.32);
    }

    .step-circle.done {
        background: linear-gradient(135deg, #22C55E 0%, #10B981 100%);
        color: white;
        box-shadow: 0 8px 18px rgba(16, 185, 129, 0.22);
    }

    .step-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-align: center;
        line-height: 1.35;
    }

    .step-label.active {
        color: #4F46E5;
        font-weight: 800;
    }

    .step-label.done {
        color: #0F766E;
    }

    .section-divider {
        border: none;
        border-top: 2px solid #EEF2FF;
        margin: 24px 0 16px 0;
    }

    .section-box {
        background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFF 100%);
        border: 1px solid var(--card-border);
        border-top: 4px solid #6366F1;
        border-radius: 18px;
        padding: 16px 16px 12px 16px;
        margin-bottom: 16px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--text-main);
        margin-bottom: 0.22rem;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: var(--text-soft);
        margin-bottom: 0.25rem;
    }

    .insight-box {
        background: linear-gradient(180deg, #F8FAFF 0%, #F4F8FF 100%);
        border: 1px solid #DCE6F7;
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .insight-title {
        font-size: 1.03rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.45rem;
    }

    .insight-text {
        font-size: 1rem;
        line-height: 1.74;
        color: #334155;
    }

    .metric-comment, .chart-note {
        background: linear-gradient(180deg, #F8FAFF 0%, #FFFFFF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 1rem;
        line-height: 1.72;
        color: #334155;
    }

    .order-box {
        background: #F8FAFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .download-card {
        background: linear-gradient(135deg, #F8FAFF 0%, #EEF2FF 100%);
        border: 1px solid #C7D2FE;
        border-radius: 18px;
        padding: 20px 22px;
        margin-top: 16px;
        margin-bottom: 8px;
        box-shadow: 0 10px 24px rgba(79, 70, 229, 0.08);
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin: 10px 0 16px 0;
    }

    .summary-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid #DCE6F2;
        border-top: 4px solid var(--card-accent, var(--accent-model));
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
    }

    .summary-card-link {
        display: block;
        text-decoration: none !important;
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    }

    .summary-card-link:hover {
        transform: translateY(-2px);
        border-color: var(--card-accent, var(--accent-model));
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.09);
    }

    .summary-label {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    .summary-value {
        color: #0F172A;
        font-size: 1.28rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 4px;
    }

    .summary-note {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    @media (max-width: 760px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }

        .summary-grid {
            grid-template-columns: 1fr;
        }
    }

    .story-panel {
        background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 18px;
        padding: 16px 18px;
        margin: 12px 0 16px 0;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
    }

    .story-title {
        color: #1D4ED8;
        font-size: 0.92rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    .story-text {
        color: #1E293B;
        font-size: 1rem;
        line-height: 1.72;
    }

    .chart-frame {
        background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 12px 14px 6px 14px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        margin-bottom: 10px;
    }

    .download-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 4px;
    }

    .download-subtitle {
        font-size: 0.94rem;
        color: #475569;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FCFDFE 100%) !important;
        border: 1px solid #DCE6F2 !important;
        border-radius: 16px !important;
        padding: 10px 12px !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.62rem 1rem !important;
        box-shadow: 0 8px 18px rgba(79, 70, 229, 0.18);
        width: 100%;
        -webkit-text-fill-color: white !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        color: white !important;
    }

    .stDownloadButton > button p,
    .stDownloadButton > button span,
    .stDownloadButton > button div {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }

    div[data-baseweb="select"] > div,
    .stTextInput input {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }

    div[role="listbox"],
    ul[role="listbox"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12) !important;
    }

    div[role="option"],
    li[role="option"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
    }

    div[role="option"]:hover,
    li[role="option"]:hover {
        background: #EEF2FF !important;
        color: #0F172A !important;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background: #EEF2FF !important;
        color: #3730A3 !important;
        border-radius: 999px !important;
    }

    .stExpander {
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        background: #FCFDFE !important;
    }

    .stAlert {
        border-radius: 14px !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        background: white;
        box-shadow: 0 6px 16px rgba(15,23,42,0.04);
    }

    div[data-testid="stRadio"] label {
        color: #0F172A !important;
        font-weight: 500;
    }

""" + dark_theme_css + workflow_anchor_css + """
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-card">
        <div class="hero-title">Explainova</div>
        <div class="hero-subtitle">
            Turn raw data into a clearer decision workflow. Prepare datasets, compare models,
            explain chart behavior, and export polished reports that are easier for any audience to follow.
        </div>
        <div class="hero-pills">
            <div class="hero-pill">Guided Data Preparation</div>
            <div class="hero-pill">Model Comparison Dashboard</div>
            <div class="hero-pill">Report-Ready Explanations</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def reset_training_state():
    for key in [
        "problem_type",
        "results_df",
        "detailed_results",
        "selected_training_mode",
        "selected_model_name",
        "shap_outputs",
        "shap_model_name",
        "shap_bar_fig",
        "shap_summary_fig",
        "shap_effect_fig",
        "shap_effect_note",
        "feature_behavior_df",
        "feature_behavior_fig",
        "model_leaderboard_fig",
        "model_recommendation_text",
        "kfold_df",
        "kfold_fig",
        "kfold_note",
        "kfold_metric_name",
        "pdp_ice_fig",
        "pdp_ice_note",
        "feature_detail_reports",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def get_completed_steps() -> int:
    if "shap_outputs" in st.session_state:
        return 4
    if "results_df" in st.session_state:
        return 3
    if "X_processed" in st.session_state:
        return 2
    if st.session_state.get("data_uploaded", False):
        return 1
    return 0


def sync_uploaded_file_state(uploaded_file):
    if not uploaded_file:
        st.session_state["data_uploaded"] = False
        st.session_state["last_uploaded_filename"] = None
        st.session_state["last_uploaded_display_name"] = None
        return

    if isinstance(uploaded_file, list):
        current_name = " + ".join(
            f"{file.name}:{getattr(file, 'size', 0)}"
            for file in uploaded_file
        )
        display_name = " + ".join(file.name for file in uploaded_file)
    else:
        current_name = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
        display_name = uploaded_file.name

    if st.session_state.get("last_uploaded_filename") != current_name:
        st.session_state["last_uploaded_filename"] = current_name
        st.session_state["last_uploaded_display_name"] = display_name
        st.session_state["data_uploaded"] = True
        reset_training_state()
        for key in [
            "X_processed",
            "y_processed",
            "X_explain_reference",
            "preprocessing_report",
            "target_column",
            "large_dataset_flag",
            "corr_heatmap_fig",
            "corr_table_for_report",
            "corr_profile_fig",
            "corr_profile_note",
            "loaded_dataset_files",
            "combined_dataset_info",
        ]:
            if key in st.session_state:
                del st.session_state[key]


def show_workflow_status(completed_steps: int):
    if completed_steps == 0:
        text = "Workflow status: waiting for dataset upload."
    elif completed_steps == 1:
        text = "Workflow status: data upload completed. Next step: preprocessing."
    elif completed_steps == 2:
        text = "Workflow status: preprocessing completed. Next step: model training."
    elif completed_steps == 3:
        text = "Workflow status: model training completed. Next step: SHAP analysis."
    else:
        text = "Workflow status: all workflow steps completed."

    st.markdown(f'<div class="status-strip">{text}</div>', unsafe_allow_html=True)


def show_step_progress(completed_steps: int):
    steps = ["Data Upload", "Preprocessing", "Model Training", "SHAP Analysis"]

    html = '<div class="stepper-wrap"><div class="stepper">'

    for i, label in enumerate(steps, 1):
        if i <= completed_steps:
            circle_class, label_class, circle_content = "done", "done", "✓"
        elif i == completed_steps + 1 and completed_steps < len(steps):
            circle_class, label_class, circle_content = "active", "active", str(i)
        else:
            circle_class, label_class, circle_content = "pending", "", str(i)

        html += (
            f'<div class="step-item">'
            f'  <div class="step-circle {circle_class}">{circle_content}</div>'
            f'  <div class="step-label {label_class}">{label}</div>'
            f'</div>'
        )

        if i < len(steps):
            conn_class = "done" if i <= completed_steps else ""
            html += f'<div class="step-connector {conn_class}"></div>'

    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def show_section_header(title, subtitle=None, anchor=None):
    anchor_html = f'<span id="{anchor}"></span>' if anchor else ""
    st.markdown(
        f"""
        {anchor_html}
        <div class="section-box">
            <div class="section-title">{title}</div>
            {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_stage_sidebar(completed_steps: int):
    stages = [
        ("Data", "Dataset Preview", "dataset-preview", completed_steps >= 1),
        ("Prep", "Preprocessing", "preprocessing-options", completed_steps >= 2),
        ("Train", "Model Training", "model-training", completed_steps >= 3),
        ("Explain", "SHAP Explainability", "shap-explainability", completed_steps >= 4),
        ("Report", "Download Report", "download-report", completed_steps >= 4),
    ]

    html_parts = [
        '<div class="sidebar-shell">',
        '<div class="sidebar-title">Workflow</div>',
        '<div class="sidebar-caption">Jump between available analysis stages.</div>',
    ]

    for idx, (_, label, anchor, available) in enumerate(stages, 1):
        is_active = (completed_steps + 1 == idx and completed_steps < 4) or (completed_steps >= 4 and idx == 4)
        status = "Active" if is_active else ("Ready" if available else "Locked")
        state_class = "active" if is_active else ("ready" if available else "locked")
        if available:
            state_class += " available"
        tag = "a" if available or idx <= completed_steps + 1 else "div"
        href_attr = f' href="#{html_lib.escape(anchor)}"' if tag == "a" else ""
        stage_index = "✓" if available and not is_active else str(idx)
        html_parts.append(
            f'<{tag} class="sidebar-stage-card {state_class}"{href_attr}>'
            f'  <div class="sidebar-stage-row">'
            f'    <span class="sidebar-stage-index">{html_lib.escape(stage_index)}</span>'
            f'    <span class="sidebar-stage-label">{html_lib.escape(label)}</span>'
            f'  </div>'
            f'  <div class="sidebar-stage-status">{html_lib.escape(status)}</div>'
            f'</{tag}>'
        )

    html_parts.append("</div>")
    st.sidebar.markdown("".join(html_parts), unsafe_allow_html=True)


def install_sidebar_scroll_sync():
    components.html(
        """
        <script>
        (() => {
            const anchors = [
                "dataset-preview",
                "preprocessing-options",
                "model-training",
                "shap-explainability",
                "download-report"
            ];

            const getDoc = () => window.parent.document;
            const getWindow = () => window.parent;

            function visibleAnchor(doc) {
                let current = null;
                let bestTop = -Infinity;
                const activationLine = 150;

                anchors.forEach((id) => {
                    const element = doc.getElementById(id);
                    if (!element) {
                        return;
                    }
                    const top = element.getBoundingClientRect().top;
                    if (top <= activationLine && top > bestTop) {
                        current = id;
                        bestTop = top;
                    }
                });

                if (current) {
                    return current;
                }

                let nearest = null;
                let nearestTop = Infinity;
                anchors.forEach((id) => {
                    const element = doc.getElementById(id);
                    if (!element) {
                        return;
                    }
                    const top = element.getBoundingClientRect().top;
                    if (top >= 0 && top < nearestTop) {
                        nearest = id;
                        nearestTop = top;
                    }
                });

                if (nearest) {
                    return nearest;
                }

                const hash = getWindow().location.hash.replace("#", "");
                if (anchors.includes(hash) && doc.getElementById(hash)) {
                    return hash;
                }
                return null;
            }

            function syncSidebar() {
                const doc = getDoc();
                const app = doc.querySelector(".stApp");
                if (!app) {
                    return;
                }

                const activeAnchor = visibleAnchor(doc);
                const cards = doc.querySelectorAll(".sidebar-stage-card[href]");
                cards.forEach((card) => {
                    card.classList.toggle(
                        "scroll-active",
                        Boolean(activeAnchor) && card.getAttribute("href") === `#${activeAnchor}`
                    );
                });
                app.classList.toggle("sidebar-scroll-sync", Boolean(activeAnchor));
            }

            let ticking = false;
            function requestSync() {
                if (ticking) {
                    return;
                }
                ticking = true;
                getWindow().requestAnimationFrame(() => {
                    ticking = false;
                    syncSidebar();
                });
            }

            getWindow().addEventListener("scroll", requestSync, { passive: true });
            getWindow().addEventListener("hashchange", requestSync);
            getWindow().addEventListener("resize", requestSync);
            setInterval(syncSidebar, 800);
            syncSidebar();
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def show_section_divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def show_info_box(title, text):
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_metric_comment(text):
    st.markdown(f'<div class="metric-comment">{text}</div>', unsafe_allow_html=True)


def show_chart_note(text):
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


def show_story_panel(title, text):
    st.markdown(
        f"""
        <div class="story-panel">
            <div class="story-title">{title}</div>
            <div class="story-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_compact_success(message):
    if st.session_state.get("dark_mode"):
        st.markdown(
            f'<div class="compact-status">{html_lib.escape(str(message))}</div>',
            unsafe_allow_html=True
        )
    else:
        st.success(message)


def show_summary_cards(cards):
    html_parts = ['<div class="summary-grid">']
    for card in cards:
        label = html_lib.escape(str(card.get("label", "")))
        value = html_lib.escape(str(card.get("value", "")))
        note = html_lib.escape(str(card.get("note", "")))
        href = str(card.get("href", "")).strip()
        accent = html_lib.escape(str(card.get("accent", "#3B82F6")))
        tag = "a" if href else "div"
        href_attr = f' href="{html_lib.escape(href)}"' if href else ""
        link_class = " summary-card-link" if href else ""
        html_parts.append(
            f'<{tag} class="summary-card{link_class}"{href_attr} style="--card-accent: {accent};">'
            f'<div class="summary-label">{label}</div>'
            f'<div class="summary-value">{value}</div>'
            f'<div class="summary-note">{note}</div>'
            f'</{tag}>'
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def show_chart_frame(fig, use_container_width=True):
    if st.session_state.get("dark_mode"):
        apply_dark_figure_theme(fig)
    st.pyplot(fig, use_container_width=use_container_width)


def apply_dark_figure_theme(fig):
    if fig is None:
        return None

    fig.patch.set_facecolor("#111827")
    for ax in fig.axes:
        ax.set_facecolor("#162033")
        ax.title.set_color("#E5E7EB")
        ax.xaxis.label.set_color("#CBD5E1")
        ax.yaxis.label.set_color("#CBD5E1")
        ax.tick_params(axis="x", colors="#CBD5E1")
        ax.tick_params(axis="y", colors="#CBD5E1")

        for spine in ax.spines.values():
            spine.set_color("#334155")

        for text in ax.texts:
            if text.get_gid() == "confusion-cell-annotation":
                continue
            if text.get_color() not in ["white", "#FFFFFF", "#ffffff"]:
                text.set_color("#E5E7EB")

        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_color("#CBD5E1")

        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#111827")
            legend.get_frame().set_edgecolor("#334155")
            for text in legend.get_texts():
                text.set_color("#E5E7EB")

    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def apply_dark_dataframe_style(data):
    if not st.session_state.get("dark_mode"):
        return data

    table_styles = [
        {
            "selector": "thead th",
            "props": [
                ("background-color", "#111827"),
                ("color", "#E5E7EB"),
                ("border-color", "#334155"),
                ("font-weight", "700"),
            ],
        },
        {
            "selector": "tbody th",
            "props": [
                ("background-color", "#111827"),
                ("color", "#CBD5E1"),
                ("border-color", "#263449"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("background-color", "#162033"),
                ("color", "#E5E7EB"),
                ("border-color", "#263449"),
            ],
        },
        {
            "selector": "tbody tr:nth-child(even) td",
            "props": [("background-color", "#111827")],
        },
        {
            "selector": "tbody tr:hover td",
            "props": [("background-color", "#1E293B")],
        },
    ]

    try:
        if isinstance(data, pd.DataFrame):
            return (
                data.style
                .set_table_styles(table_styles)
                .set_properties(**{
                    "background-color": "#162033",
                    "color": "#E5E7EB",
                    "border-color": "#263449",
                })
            )
        if hasattr(data, "set_table_styles"):
            return data.set_table_styles(table_styles, overwrite=False)
    except Exception:
        return data

    return data


def show_dataframe(data, **kwargs):
    if st.session_state.get("dark_mode"):
        styled_data = apply_dark_dataframe_style(data)
        height = kwargs.get("height")
        max_height = f"max-height: {int(height)}px;" if height else "max-height: 420px;"

        try:
            table_html = styled_data.to_html()
        except Exception:
            try:
                table_html = pd.DataFrame(data).to_html(escape=True)
            except Exception:
                st.dataframe(data, **kwargs)
                return

        st.markdown(
            f'<div class="dark-table-wrap" style="{max_height}">{table_html}</div>',
            unsafe_allow_html=True
        )
        return

    st.dataframe(data, **kwargs)


def responsive_pair_columns(weights=None):
    if st.session_state.get("is_mobile"):
        return st.container(), st.container()
    return st.columns(weights or [1, 1])


def queue_toast(message, icon="✅"):
    st.session_state["_pending_toast"] = {"message": message, "icon": icon}


def flush_pending_toast():
    pending = st.session_state.pop("_pending_toast", None)
    if pending and hasattr(st, "toast"):
        if st.session_state.get("dark_mode"):
            return
        st.toast(pending["message"], icon=pending.get("icon", "✅"))


def show_list(title, items):
    if not items:
        return False

    st.markdown(f"**{title}**")
    for item in items:
        st.write(f"- {item}")
    return True


def show_list_group(title, list_entries):
    non_empty_entries = [
        (entry_title, items)
        for entry_title, items in list_entries
        if items
    ]

    if not non_empty_entries:
        return False

    st.subheader(title)
    for entry_title, items in non_empty_entries:
        show_list(entry_title, items)

    return True


def build_zero_value_summary(df, target_column, selected_feature_columns=None, numeric_parse_threshold=0.80):
    if selected_feature_columns is None:
        candidate_columns = [col for col in df.columns if col != target_column]
    else:
        candidate_columns = [col for col in selected_feature_columns if col in df.columns and col != target_column]

    rows = []

    for col in candidate_columns:
        series = df[col]
        non_null = series.dropna()

        if non_null.empty or pd.api.types.is_bool_dtype(series):
            continue

        numeric_values = pd.to_numeric(series, errors="coerce")
        numeric_count = int(numeric_values.notna().sum())

        if numeric_count == 0 or numeric_count / len(non_null) < numeric_parse_threshold:
            continue

        zero_count = int((numeric_values.eq(0) & series.notna()).sum())

        if zero_count == 0:
            continue

        rows.append({
            "Column": col,
            "Zero Values": zero_count,
            "Zero Share": zero_count / len(series),
            "Non-null Values": int(non_null.shape[0])
        })

    if not rows:
        return pd.DataFrame(columns=["Column", "Zero Values", "Zero Share", "Non-null Values"])

    return (
        pd.DataFrame(rows)
        .sort_values(by=["Zero Values", "Column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def count_total_dropped_columns(report):
    return (
            len(report.get("dropped_empty_columns", []))
            + len(report.get("dropped_high_missing_columns", []))
            + len(report.get("dropped_single_value_columns", []))
            + len(report.get("dropped_id_columns", []))
            + len(report.get("dropped_high_cardinality_columns", []))
    )


def build_help_text(items, default_text="No items recorded."):
    if not items:
        return default_text
    joined = ", ".join(map(str, items))
    if len(joined) > 350:
        joined = joined[:347] + "..."
    return joined


def explain_preprocessing_step(title, explanation):
    st.markdown(f"**{title}**")
    st.write(explanation)


def show_preprocessing_explanations(report):
    st.subheader("Detailed Preprocessing Explanations")

    shown_any = False

    if report.get("removed_duplicates", 0) > 0:
        shown_any = True
        explain_preprocessing_step(
            "Duplicate rows removed",
            "Rows that were exactly the same were removed. This helps prevent the model from giving extra weight to repeated observations."
        )

    if (
        report.get("dropped_empty_columns")
        or report.get("dropped_high_missing_columns")
        or report.get("dropped_single_value_columns")
        or report.get("dropped_id_columns")
        or report.get("dropped_high_cardinality_columns")
    ):
        shown_any = True
        explain_preprocessing_step(
            "Problematic columns removed",
            "Columns were removed only when they were unlikely to help prediction, such as fully empty columns, one-value columns, ID-like columns, high-missing columns, or columns with too many unique categories."
        )

    if report.get("zero_as_missing_columns"):
        shown_any = True
        explain_preprocessing_step(
            "Zero placeholders treated as missing",
            "The selected columns had 0 values converted to missing values before imputation, because the user marked those zeros as placeholders rather than real measurements."
        )

    if report.get("filled_missing_numerical") or report.get("filled_missing_categorical"):
        shown_any = True
        explain_preprocessing_step(
            "Missing values handled",
            "Blank cells were filled in so the model can train without failing. Numeric blanks use the median value; categorical blanks use the most common category."
        )

    if report.get("parsed_datetime_columns"):
        shown_any = True
        explain_preprocessing_step(
            "Datetime columns transformed",
            "Date and time columns were converted into useful parts such as year, month, day, and weekday, because models usually learn better from these pieces than from raw date text."
        )

    if report.get("capped_outlier_columns"):
        shown_any = True
        explain_preprocessing_step(
            "Extreme outliers capped",
            "Very unusual numeric values were capped so the most extreme values do not dominate training."
        )

    if (
        report.get("ordinal_encoded_columns")
        or report.get("one_hot_encoded_columns")
        or report.get("target_encoded")
    ):
        shown_any = True
        explain_preprocessing_step(
            "Categorical columns encoded",
            "Text categories were converted into numbers. Ordered categories keep their order, while unordered categories are expanded into separate yes/no columns."
        )

    if report.get("feature_reduction_applied"):
        shown_any = True
        explain_preprocessing_step(
            "Feature reduction",
            "The dataset was large or wide, so optional feature reduction simplified the input columns. This keeps the model easier to train and explain while preserving meaningful feature names."
        )

    if not shown_any:
        st.info("No major preprocessing changes were needed before model training.")


def show_metric_explanations(problem_type, has_roc_auc=False):
    st.subheader("Metric Explanations")

    if problem_type == "classification":
        st.markdown("**Accuracy** - The share of predictions that were correct overall.")
        st.markdown("**Use it when:** the classes are reasonably balanced and all mistake types have similar cost.")
        st.markdown("**Precision** - Of the rows predicted as positive, how many were truly positive.")
        st.markdown("**Use it when:** false alarms are expensive or annoying.")
        st.markdown("**Recall** - Of the truly positive rows, how many the model successfully found.")
        st.markdown("**Use it when:** missing a real positive case is costly.")
        st.markdown("**F1 Score** - A single balance score for precision and recall.")
        st.markdown("**Use it when:** both false alarms and missed positives matter.")
        if has_roc_auc:
            st.markdown("**ROC AUC** - How well the model separates two classes across possible thresholds.")
            st.markdown("**Use it when:** you care about ranking positive cases ahead of negative cases, not just one fixed cutoff.")
    else:
        st.markdown("**R2 Score** - How much of the target's variation the model can explain. Higher is better.")
        st.markdown("**Use it when:** you want a quick sense of overall fit.")
        st.markdown("**MAE** - The average absolute prediction error in the target's original units. Lower is better.")
        st.markdown("**Use it when:** you want an easy-to-read typical error size.")
        st.markdown("**RMSE** - Similar to MAE, but larger mistakes count more. Lower is better.")
        st.markdown("**Use it when:** big errors are especially harmful.")


def get_best_model_info(results_df, problem_type):
    if results_df.empty:
        return None, None, None

    if problem_type == "classification":
        metric = "Accuracy"
        ascending = False
    else:
        metric = "R2 Score"
        ascending = False

    best_row = results_df.sort_values(by=metric, ascending=ascending).iloc[0]
    return best_row["Model"], metric, best_row[metric]


def style_best_model_row(results_df, metric_name):
    if results_df is None or results_df.empty or metric_name not in results_df.columns:
        return results_df

    best_value = results_df[metric_name].max()

    def highlight_best(row):
        if row.get(metric_name) == best_value:
            if st.session_state.get("dark_mode"):
                return ["background-color: #312E81; color: #FFFFFF; font-weight: 700" for _ in row]
            return ["background-color: #EDE9FE; font-weight: 700" for _ in row]
        return ["" for _ in row]

    return results_df.style.apply(highlight_best, axis=1)


def build_kfold_column_config(kfold_df, metric_name):
    if kfold_df is None or kfold_df.empty:
        return {}

    mean_col = f"{metric_name} Mean"
    std_col = f"{metric_name} Std"
    config = {}

    if mean_col in kfold_df.columns:
        values = pd.to_numeric(kfold_df[mean_col], errors="coerce").dropna()
        if metric_name == "Accuracy" or (not values.empty and values.between(0, 1).all()):
            config[mean_col] = st.column_config.ProgressColumn(
                mean_col,
                min_value=0,
                max_value=1,
                format="%.4f",
                width="medium"
            )
        else:
            config[mean_col] = st.column_config.NumberColumn(
                mean_col,
                format="%.4f",
                width="medium"
            )

    if std_col in kfold_df.columns:
        config[std_col] = st.column_config.NumberColumn(
            std_col,
            format="%.4f",
            width="small"
        )

    if "Model" in kfold_df.columns:
        config["Model"] = st.column_config.TextColumn("Model", width="medium")

    return config


def get_positive_class_label(problem_type, report, y=None):
    if problem_type != "classification":
        return None

    mapping = report.get("target_label_mapping") or {}
    if 1 in mapping:
        return mapping[1]
    if "1" in mapping:
        return mapping["1"]

    if y is not None:
        values = pd.Series(y).dropna().unique().tolist()
        if len(values) == 2:
            try:
                return str(sorted(values)[-1])
            except Exception:
                return str(values[-1])

    return "positive class"


def format_positive_class_text(positive_class_label):
    if positive_class_label is None or str(positive_class_label).strip() == "":
        return "class 1 / the positive class"
    label_text = str(positive_class_label).strip()
    if label_text in ["1", "1.0"]:
        return "class 1"
    return f"the positive class ({label_text})"


def format_metric_value(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def get_metric_focus_label(problem_type):
    if problem_type == "classification":
        return "Correct class detection"
    return "Close-to-reality numeric prediction"


def is_large_dataset(df, row_threshold=50000, column_threshold=100, cell_threshold=2_000_000):
    rows, cols = df.shape
    return rows >= row_threshold or cols >= column_threshold or rows * cols >= cell_threshold


def should_offer_feature_reduction(df, row_threshold=10000, column_threshold=40, cell_threshold=300000):
    rows, cols = df.shape
    return rows >= row_threshold or cols >= column_threshold or rows * cols >= cell_threshold


def get_kfold_cost_label(X, model_count):
    rows, cols = X.shape
    total_cells = rows * cols
    if rows >= 20000 or cols >= 80 or total_cells >= 700000 or model_count >= 4:
        return "High", "This may take noticeably longer on the current dataset."
    if rows >= 5000 or cols >= 40 or total_cells >= 200000 or model_count >= 2:
        return "Medium", "This can add some extra processing time."
    return "Low", "This should usually complete quickly."


def get_kfold_metric_options(kfold_df, problem_type):
    if kfold_df is None or kfold_df.empty:
        return []

    if problem_type == "classification":
        candidates = ["ROC AUC", "F1 Score", "Accuracy"]
    else:
        candidates = ["R2 Score", "MAE", "RMSE"]

    return [metric for metric in candidates if f"{metric} Mean" in kfold_df.columns]


def build_kfold_metric_note(metric_name):
    notes = {
        "ROC AUC": "ROC AUC checks whether positives tend to be ranked ahead of negatives across different thresholds. It can favor a model even when raw accuracy is not the highest.",
        "F1 Score": "F1 balances precision and recall, so it is useful when false alarms and missed positives both matter.",
        "Accuracy": "Accuracy is the easiest overall correctness score to read, but it can look too optimistic when one class is much more common than the other.",
        "R2 Score": "R2 shows how much target variation the model explains. Higher values usually mean a stronger regression fit.",
        "MAE": "MAE shows the typical absolute prediction error. Lower values mean smaller everyday mistakes.",
        "RMSE": "RMSE gives extra weight to larger errors. Lower values mean fewer or smaller large misses."
    }
    return notes.get(metric_name, "Select the metric that best matches the decision goal.")


def get_primary_stability_metric(problem_type):
    return "Accuracy" if problem_type == "classification" else "R2 Score"


def show_metric_plots(results_df, problem_type):
    if problem_type == "classification":
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        if "ROC AUC" in results_df.columns:
            metrics.append("ROC AUC")
    else:
        metrics = ["R2 Score", "MAE", "RMSE"]

    metric_figures = plot_metric_grid(results_df, metrics)
    available_metrics = [m for m in metrics if m in metric_figures]

    if not available_metrics:
        return

    st.markdown('<span id="metric-comparisons"></span>', unsafe_allow_html=True)
    st.subheader("Metric Comparisons")

    for i in range(0, len(available_metrics), 2):
        cols = st.columns(2)
        pair = available_metrics[i:i + 2]

        for col, metric in zip(cols, pair):
            with col:
                show_chart_frame(metric_figures[metric], use_container_width=False)
                show_metric_comment(get_metric_commentary(results_df, metric, problem_type))


def parse_order_input(order_text):
    if not order_text:
        return []
    return [item.strip() for item in order_text.split(",") if item.strip()]


def validate_ordinal_order_input(column_name, available_values, parsed_order):
    if not parsed_order:
        return []

    available_lookup = {
        str(value).strip().lower(): str(value).strip()
        for value in available_values
        if str(value).strip()
    }
    entered_normalized = [str(value).strip().lower() for value in parsed_order if str(value).strip()]
    entered_set = set(entered_normalized)
    errors = []

    duplicated_values = sorted({
        parsed_order[idx]
        for idx, normalized_value in enumerate(entered_normalized)
        if entered_normalized.count(normalized_value) > 1
    })
    if duplicated_values:
        errors.append(
            f"{column_name}: remove duplicate values: {', '.join(map(str, duplicated_values))}."
        )

    unknown_values = [
        parsed_order[idx]
        for idx, normalized_value in enumerate(entered_normalized)
        if normalized_value not in available_lookup
    ]
    if unknown_values:
        errors.append(
            f"{column_name}: these values are not in the column: {', '.join(map(str, unknown_values))}."
        )

    missing_values = [
        original_value
        for normalized_value, original_value in available_lookup.items()
        if normalized_value not in entered_set
    ]
    if missing_values:
        errors.append(
            f"{column_name}: include every available value: {', '.join(map(str, missing_values))}."
        )

    return errors


def build_report_filename(dataset_filename):
    if not dataset_filename:
        return "Explainova - Analysis.docx"

    stem = os.path.splitext(dataset_filename)[0].strip()
    stem = re.sub(r'[\\/:*?"<>|]+', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip()

    if not stem:
        return "Explainova - Analysis.docx"

    return f"{stem} - Explainova - Analysis.docx"


def load_uploaded_datasets(uploaded_files):
    loaded = []

    for uploaded in uploaded_files:
        df_loaded = load_dataset(uploaded)
        loaded.append({
            "name": uploaded.name,
            "dataframe": df_loaded,
        })

    return loaded


def get_common_columns(loaded_datasets):
    if not loaded_datasets:
        return []

    common = set(loaded_datasets[0]["dataframe"].columns)
    for item in loaded_datasets[1:]:
        common &= set(item["dataframe"].columns)

    return sorted(common)


def build_loaded_files_summary(loaded_datasets):
    return pd.DataFrame([
        {
            "File": item["name"],
            "Rows": item["dataframe"].shape[0],
            "Columns": item["dataframe"].shape[1],
        }
        for item in loaded_datasets
    ])


def combine_datasets_by_rows(loaded_datasets):
    all_columns = sorted({
        col
        for item in loaded_datasets
        for col in item["dataframe"].columns
    })
    combined = pd.concat(
        [
            item["dataframe"].reindex(columns=all_columns).assign(_source_file=item["name"])
            for item in loaded_datasets
        ],
        ignore_index=True
    )

    return combined, {
        "method": "Stack rows",
        "source_files": [item["name"] for item in loaded_datasets],
    }


def make_unique_column_name(base_name, existing_columns):
    candidate = base_name
    counter = 2

    while candidate in existing_columns:
        candidate = f"{base_name}_{counter}"
        counter += 1

    return candidate


def merge_datasets_by_key(loaded_datasets, key_columns, merge_how):
    merged = loaded_datasets[0]["dataframe"].copy()
    source_files = [loaded_datasets[0]["name"]]

    for idx, item in enumerate(loaded_datasets[1:], start=2):
        right_df = item["dataframe"].copy()
        suffix = re.sub(r"[^A-Za-z0-9_]+", "_", Path(item["name"]).stem).strip("_") or f"file_{idx}"
        rename_map = {}

        for col in right_df.columns:
            if col in key_columns or col not in merged.columns:
                continue

            rename_map[col] = make_unique_column_name(f"{col}__{suffix}", set(merged.columns) | set(right_df.columns))

        if rename_map:
            right_df = right_df.rename(columns=rename_map)

        merged = merged.merge(
            right_df,
            on=key_columns,
            how=merge_how,
        )
        source_files.append(item["name"])

    return merged, {
        "method": f"Merge by key ({merge_how})",
        "key_columns": key_columns,
        "source_files": source_files,
    }


def show_multi_file_combine_controls(loaded_datasets):
    st.subheader("Combine Uploaded Files")
    show_dataframe(build_loaded_files_summary(loaded_datasets), use_container_width=True)

    combine_mode = st.radio(
        "How should these files be combined?",
        options=["Merge by shared ID columns", "Stack rows with matching columns"],
        help="Use merge when files contain different columns for the same entities. Use stack when files contain more rows of the same table."
    )

    if combine_mode == "Stack rows with matching columns":
        combined_df, combine_info = combine_datasets_by_rows(loaded_datasets)
        show_compact_success(f"Combined {len(loaded_datasets)} files by stacking rows.")
        return combined_df, combine_info

    common_columns = get_common_columns(loaded_datasets)

    if not common_columns:
        st.error("These files do not share any column names, so they cannot be merged by ID.")
        return None, None

    default_keys = [
        col for col in common_columns
        if col.strip().lower() in {"id", "patient_id", "sample_id", "record_id", "user_id", "customer_id"}
        or col.strip().lower().endswith("_id")
    ]

    key_columns = st.multiselect(
        "Select the ID/key column(s) to merge on",
        options=common_columns,
        default=default_keys[:1],
        help="Choose one or more columns that identify the same row/entity across all uploaded files."
    )

    merge_how = st.selectbox(
        "Merge type",
        options=["inner", "left", "outer"],
        index=0,
        help="Inner keeps only matching IDs. Left keeps all rows from the first file. Outer keeps all IDs from every file."
    )

    if not key_columns:
        st.info("Select at least one shared ID/key column to preview and continue.")
        return None, None

    duplicate_messages = []
    for item in loaded_datasets:
        duplicate_count = int(item["dataframe"].duplicated(subset=key_columns, keep=False).sum())
        if duplicate_count > 0:
            duplicate_messages.append(f"{item['name']}: {duplicate_count} duplicate key row(s)")

    if duplicate_messages:
        st.warning(
            "Some files have repeated key values. Merge can create extra rows when duplicate keys match. "
            + " | ".join(duplicate_messages)
        )

    combined_df, combine_info = merge_datasets_by_key(loaded_datasets, key_columns, merge_how)
    show_compact_success(
        f"Merged {len(loaded_datasets)} files using {', '.join(key_columns)}."
    )

    return combined_df, combine_info


def build_local_contribution_table(shap_outputs, sample_index, top_n=6):
    X_explain = shap_outputs["X_explain"]
    shap_values = np.array(shap_outputs["shap_values"], dtype=float)

    row = X_explain.iloc[sample_index]
    row_shap = shap_values[sample_index]

    df_local = pd.DataFrame({
        "Feature": X_explain.columns,
        "Value": row.values,
        "SHAP Contribution": row_shap,
        "Direction": np.where(row_shap >= 0, "Pushes up", "Pulls down"),
        "Absolute Effect": np.abs(row_shap),
    }).sort_values("Absolute Effect", ascending=False).head(top_n).reset_index(drop=True)

    return df_local[["Feature", "Value", "SHAP Contribution", "Direction"]]


def build_feature_behavior_summary(shap_outputs, top_n=8, problem_type=None):
    X_explain = shap_outputs["X_explain"]
    shap_values = np.array(shap_outputs["shap_values"], dtype=float)
    importance_df = shap_outputs["feature_importance_df"].head(top_n)

    rows = []
    for feature_name in importance_df["Feature"].tolist():
        feature_idx = list(X_explain.columns).index(feature_name)
        feature_vals = X_explain[feature_name].values
        feature_shap = shap_values[:, feature_idx]

        if np.std(feature_vals) == 0 or np.std(feature_shap) == 0:
            pattern = "No clear pattern"
        else:
            corr = np.corrcoef(feature_vals, feature_shap)[0, 1]
            if problem_type == "classification":
                if corr >= 0.35:
                    pattern = "Higher values move toward class 1"
                elif corr <= -0.35:
                    pattern = "Higher values move away from class 1"
                else:
                    pattern = "Mixed / non-linear effect"
            else:
                if corr >= 0.35:
                    pattern = "Higher values usually raise prediction"
                elif corr <= -0.35:
                    pattern = "Higher values usually lower prediction"
                else:
                    pattern = "Mixed / non-linear effect"

        rows.append({
            "Feature": feature_name,
            "Average Strength": float(np.mean(np.abs(feature_shap))),
            "Typical Pattern": pattern
        })

    return pd.DataFrame(rows)


def show_download_section(target_column, report, results_df, problem_type,
                          shap_outputs, shap_model_name, shap_bar_fig, shap_summary_fig,
                          corr_table_for_report=None, corr_heatmap_fig=None,
                          corr_profile_fig=None,
                          shap_effect_fig=None,
                          effect_note=None,
                          model_leaderboard_fig=None, model_recommendation_text=None,
                          corr_profile_note=None,
                          feature_behavior_df=None, feature_behavior_fig=None,
                          kfold_df=None, kfold_fig=None, kfold_note=None,
                          pdp_ice_fig=None, pdp_ice_note=None,
                          feature_detail_reports=None,
                          ):
    st.markdown(
        """
        <span id="download-report"></span>
        <div class="download-card">
            <div class="download-title">Download Report</div>
            <div class="download-subtitle">
                Export preprocessing steps, model results, summary visuals, and SHAP explanations into one polished Word report. Use the Download Report item in the workflow sidebar to return here quickly.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    dataset_filename = st.session_state.get("last_uploaded_display_name") or st.session_state.get("last_uploaded_filename")
    report_filename = build_report_filename(dataset_filename)

    try:
        word_buf = generate_word_report(
            target_column=target_column,
            report=report,
            results_df=results_df,
            problem_type=problem_type,
            shap_outputs=shap_outputs,
            shap_model_name=shap_model_name,
            shap_bar_fig=shap_bar_fig,
            shap_summary_fig=shap_summary_fig,
            corr_table=corr_table_for_report,
            corr_heatmap_fig=corr_heatmap_fig,
            corr_profile_fig=corr_profile_fig,
            shap_effect_fig=shap_effect_fig,
            effect_note=effect_note,
            model_leaderboard_fig=model_leaderboard_fig,
            model_recommendation_text=model_recommendation_text,
            corr_profile_note=corr_profile_note,
            feature_behavior_df=feature_behavior_df,
            feature_behavior_fig=feature_behavior_fig,
            kfold_df=kfold_df,
            kfold_fig=kfold_fig,
            kfold_note=kfold_note,
            pdp_ice_fig=pdp_ice_fig,
            pdp_ice_note=pdp_ice_note,
            feature_detail_reports=feature_detail_reports,
        )
        report_bytes = word_buf.getvalue()
        st.session_state["report_ready"] = True
        st.session_state["report_bytes"] = report_bytes
        st.session_state["report_filename"] = report_filename
        st.download_button(
            label="Download Word Report",
            data=report_bytes,
            file_name=report_filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="main_word_report_download",
        )
        show_compact_success("Your Word report is ready.")
    except ImportError:
        st.session_state["report_ready"] = False
        st.warning("Word export requires python-docx. Install it with: pip install python-docx")
    except Exception as e:
        st.session_state["report_ready"] = False
        st.error(f"Word report could not be generated: {e}")


uploaded_files = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "tsv"],
    accept_multiple_files=True,
    help="Supported file types: CSV, XLSX, XLS, TSV. Upload multiple files to merge them by a shared ID column."
)

sync_uploaded_file_state(uploaded_files)
flush_pending_toast()
completed_steps = get_completed_steps()
show_stage_sidebar(completed_steps)
install_sidebar_scroll_sync()
show_step_progress(completed_steps)

if uploaded_files:
    try:
        show_section_header("Dataset Preview", "Check the raw rows, column count, and missing values before choosing what the model should predict.", anchor="dataset-preview")
        loaded_datasets = load_uploaded_datasets(uploaded_files)

        if len(loaded_datasets) == 1:
            df = loaded_datasets[0]["dataframe"]
            st.session_state["combined_dataset_info"] = {
                "method": "Single file",
                "source_files": [loaded_datasets[0]["name"]],
            }
        else:
            df, combine_info = show_multi_file_combine_controls(loaded_datasets)

            if df is None:
                st.stop()

            st.session_state["combined_dataset_info"] = combine_info

        preview_rows = st.selectbox(
            "How many rows would you like to preview?",
            options=[5, 10, 15, 20],
            index=0
        )

        show_dataframe(df.head(preview_rows), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        show_section_divider()
        show_section_header("Feature and Target Selection", "Choose the target column, which is what the model should predict, and decide which input columns should help it learn.")

        feature_mode = st.radio(
            "How would you like to use features?",
            options=["Use all available features", "Select features manually"],
            index=0,
            help="Features are the input columns used to make predictions. You can use every available input or choose a smaller set yourself."
        )

        target_column = st.selectbox(
            "Select the target column",
            df.columns,
            help="The target is the answer column. For example: churn, diagnosis, price, score, or any value you want the model to predict."
        )

        selected_feature_columns = None
        available_feature_candidates = [col for col in df.columns if col != target_column]

        if feature_mode == "Select features manually":
            selected_feature_columns = st.multiselect(
                "Select the feature columns to include",
                options=available_feature_candidates,
                default=available_feature_candidates[: min(8, len(available_feature_candidates))],
                help="Only these input columns will be cleaned and used for model training. Leave out columns that leak the answer or should not influence the prediction."
            )

        ordinal_source_df = (
            df if selected_feature_columns is None
            else df[selected_feature_columns + [target_column]]
        )
        ordinal_info = suggest_ordinal_columns(ordinal_source_df, target_column)
        all_categorical_columns = ordinal_info["categorical_columns"]
        auto_detected_ordinal_columns = ordinal_info["auto_detected_ordinal_columns"]

        show_section_divider()
        show_section_header("Preprocessing Options", "Prepare the data so models can read it: handle blanks, encode text categories, and protect useful signal.", anchor="preprocessing-options")

        user_selected_ordinal_columns = []
        user_defined_ordinal_mappings = {}
        ordinal_order_errors = []

        if all_categorical_columns:
            with st.expander("Ordinal data information"):
                st.write(
                    "Ordinal data contains categories with a meaningful order. "
                    "Examples include low < medium < high or mild < moderate < severe. "
                    "Only mark a column as ordinal when the order is real, not just alphabetical."
                )

                if auto_detected_ordinal_columns:
                    show_compact_success("Automatically detected ordinal columns: " + ", ".join(auto_detected_ordinal_columns))
                else:
                    st.info("No ordinal columns were automatically detected from the known patterns.")

            selectable_manual_ordinal_columns = [
                col for col in all_categorical_columns if col not in auto_detected_ordinal_columns
            ]

            if feature_mode == "Select features manually" and selected_feature_columns:
                selectable_manual_ordinal_columns = [
                    col for col in selectable_manual_ordinal_columns
                    if col in selected_feature_columns
                ]

            if selectable_manual_ordinal_columns:
                user_selected_ordinal_columns = st.multiselect(
                    "Do you want to additionally mark any categorical columns as ordinal?",
                    options=selectable_manual_ordinal_columns,
                    help="Choose extra columns only if their categories have a true order."
                )

                if user_selected_ordinal_columns:
                    st.markdown("### Define category order for selected ordinal columns")
                    st.caption(
                        "For each selected column, enter every category from lowest to highest, separated by commas. Matching is case-insensitive."
                    )

                    for col in user_selected_ordinal_columns:
                        unique_values = (
                            ordinal_source_df[col].dropna().astype(str).str.strip().unique().tolist()
                            if col in ordinal_source_df.columns else []
                        )

                        st.markdown(
                            f"""
                            <div class="order-box">
                                <strong>{col}</strong><br>
                                Available values: {", ".join(unique_values) if unique_values else "No non-null values found"}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        order_text = st.text_input(
                            f"Order for {col}",
                            key=f"order_{col}",
                            placeholder="Example: low, medium, high",
                            help="Write the values in order from lowest to highest."
                        )

                        parsed_order = parse_order_input(order_text)
                        if parsed_order:
                            current_errors = validate_ordinal_order_input(col, unique_values, parsed_order)
                            ordinal_order_errors.extend(current_errors)
                            for error_message in current_errors:
                                st.error(error_message)
                            user_defined_ordinal_mappings[col] = parsed_order
                        else:
                            missing_order_message = f"{col}: enter the full category order before preprocessing."
                            ordinal_order_errors.append(missing_order_message)
                            st.error(missing_order_message)

        if feature_mode == "Select features manually" and selected_feature_columns:
            df_for_size = df[selected_feature_columns + [target_column]]
        else:
            df_for_size = df

        zero_as_missing_columns = []
        zero_value_summary = build_zero_value_summary(
            df,
            target_column=target_column,
            selected_feature_columns=selected_feature_columns
        )

        if not zero_value_summary.empty:
            with st.expander("Zero value check"):
                st.write(
                    "Some numeric columns contain 0 values. In some datasets 0 is a real value; "
                    "in others it is used as a placeholder for a missing value."
                )
                zero_display = zero_value_summary.copy()
                zero_display["Zero Share"] = zero_display["Zero Share"].map(lambda value: f"{value:.1%}")
                show_dataframe(zero_display, use_container_width=True)

                zero_as_missing_columns = st.multiselect(
                    "Which columns have 0 values that should be treated as missing?",
                    options=zero_value_summary["Column"].tolist(),
                    help="Selected columns will have 0 values converted to missing values before median/mode imputation."
                )

        large_dataset_flag = is_large_dataset(df_for_size)
        feature_reduction_available = should_offer_feature_reduction(df_for_size)

        if large_dataset_flag:
            st.warning("This dataset is large. Preprocessing can still run, but model training and explanations may take longer.")

        apply_feature_selection = "No"
        feature_reduction_strategy = "fast_interpretable"
        feature_selection_correlation_threshold = 0.95
        protected_original_features = []

        if feature_reduction_available:
            show_info_box(
                "Feature Reduction for Explainability",
                "Large or very wide datasets can be harder to train and explain. This option removes columns that add little information or repeat nearly the same signal as another column."
            )

            apply_feature_selection = st.radio(
                "Apply feature selection?",
                options=["No", "Yes"],
                index=1,
                help="Keeps the feature set smaller by removing low-information or highly similar columns."
            )

        if st.button("Run Preprocessing", disabled=bool(ordinal_order_errors)):
            reset_training_state()

            if feature_mode == "Select features manually" and (selected_feature_columns is None or len(selected_feature_columns) == 0):
                st.error("Please select at least one feature column.")
            elif ordinal_order_errors:
                st.error("Fix the ordinal category order before running preprocessing.")
            else:
                with st.spinner("Preprocessing running..."):
                    X, y, report, X_explain_reference = preprocess_data(
                        df=df,
                        target_column=target_column,
                        selected_feature_columns=selected_feature_columns,
                        user_selected_ordinal_columns=user_selected_ordinal_columns,
                        user_defined_ordinal_mappings=user_defined_ordinal_mappings,
                        apply_feature_reduction=(apply_feature_selection == "Yes"),
                        feature_reduction_strategy=feature_reduction_strategy,
                        protected_original_features=protected_original_features,
                        zero_as_missing_columns=zero_as_missing_columns,
                        low_variance_threshold=0.0001,
                        high_correlation_threshold=feature_selection_correlation_threshold,
                        top_k_important_features=40
                    )

                st.session_state["X_processed"] = X
                st.session_state["y_processed"] = y
                st.session_state["X_explain_reference"] = X_explain_reference
                st.session_state["preprocessing_report"] = report
                st.session_state["target_column"] = target_column
                st.session_state["large_dataset_flag"] = large_dataset_flag
                queue_toast("Preprocessing completed successfully.")
                st.rerun()

        if "X_processed" in st.session_state and "y_processed" in st.session_state:
            X = st.session_state["X_processed"]
            y = st.session_state["y_processed"]
            X_explain_reference = st.session_state["X_explain_reference"]
            report = st.session_state["preprocessing_report"]

            show_section_divider()
            show_section_header("Preprocessing Review", "Confirm what changed during cleaning before training a model on the processed data.", anchor="preprocessing-review")

            total_dropped_columns = count_total_dropped_columns(report)
            total_missing_filled = len(report.get("filled_missing_numerical", [])) + len(report.get("filled_missing_categorical", []))

            help_duplicates = f"Removed duplicate rows count: {report.get('removed_duplicates', 0)}"
            help_dropped = (
                    "Dropped columns include: " +
                    build_help_text(
                        report.get("dropped_empty_columns", [])
                        + report.get("dropped_high_missing_columns", [])
                        + report.get("dropped_single_value_columns", [])
                        + report.get("dropped_id_columns", [])
                        + report.get("dropped_high_cardinality_columns", [])
                    )
            )
            help_rows_removed = f"Rows removed due to missing-value threshold: {report.get('removed_rows_due_to_missing', 0)}"
            help_missing = (
                    "Missing-handled columns: " +
                    build_help_text(report.get("filled_missing_numerical", []) + report.get("filled_missing_categorical", []))
            )
            help_datetime = (
                    "Created datetime features: " +
                    build_help_text(report.get("created_datetime_features", []))
            )
            help_encoded = (
                    "Encoded columns include ordinal + one-hot encoded columns: " +
                    build_help_text(report.get("ordinal_encoded_columns", []) + report.get("one_hot_encoded_columns", []))
            )

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Duplicates Removed", report.get("removed_duplicates", 0), help=help_duplicates)
            summary_col2.metric("Columns Dropped", total_dropped_columns, help=help_dropped)
            summary_col3.metric("Rows Removed", report.get("removed_rows_due_to_missing", 0), help=help_rows_removed)

            summary_col4, summary_col5, summary_col6 = st.columns(3)
            summary_col4.metric("Missing Columns Handled", total_missing_filled, help=help_missing)
            summary_col5.metric("Datetime Features Created", len(report.get("created_datetime_features", [])), help=help_datetime)
            summary_col6.metric("Encoded Columns", len(report.get("ordinal_encoded_columns", [])) + len(report.get("one_hot_encoded_columns", [])), help=help_encoded)

            if X_explain_reference.empty:
                st.warning("Processed data is empty after preprocessing. Please review the preprocessing report.")
            else:
                st.subheader("Processed Data Review")
                show_chart_note("This is the model-ready version of the dataset. Text categories may now appear as encoded numeric columns.")
                show_dataframe(X_explain_reference.head(), use_container_width=True)

            with st.expander("Detailed preprocessing explanations"):
                show_preprocessing_explanations(report)

            with st.expander("Detailed preprocessing report"):
                st.subheader("Dataset Shape")
                st.write(f"Initial shape: {report.get('initial_shape')}")
                st.write(f"Final feature shape: {report.get('final_shape')}")

                st.subheader("Selected Features")
                if report.get("selected_feature_columns") is None:
                    st.write("All available features were used.")
                else:
                    st.write(report.get("selected_feature_columns"))

                show_list_group("Dropped Columns", [
                    ("Empty Columns", report.get("dropped_empty_columns", [])),
                    ("High-Missing Columns", report.get("dropped_high_missing_columns", [])),
                    ("Single-Value Columns", report.get("dropped_single_value_columns", [])),
                    ("ID Columns", report.get("dropped_id_columns", [])),
                    ("High-Cardinality Columns", report.get("dropped_high_cardinality_columns", [])),
                ])

                show_list_group("Type Conversion", [
                    ("Converted to Numeric", report.get("converted_to_numeric", [])),
                    ("Parsed Datetime Columns", report.get("parsed_datetime_columns", [])),
                    ("Created Datetime Features", report.get("created_datetime_features", [])),
                ])

                missing_entries = [
                    ("Zero-as-Missing Columns", [
                        f"{col}: {count} zero value(s)"
                        for col, count in report.get("zero_as_missing_counts", {}).items()
                    ]),
                    ("Filled Numerical Columns", report.get("filled_missing_numerical", [])),
                    ("Filled Categorical Columns", report.get("filled_missing_categorical", [])),
                ]
                show_list_group("Missing Value Handling", missing_entries)

                show_list_group("Encoding", [
                    ("Auto-detected Ordinal Columns", report.get("auto_detected_ordinal_columns", [])),
                    ("User-selected Ordinal Columns", report.get("user_selected_ordinal_columns", [])),
                    ("User-defined Ordinal Columns", report.get("user_defined_ordinal_columns", [])),
                    ("Ordinal Columns That Could Not Be Safely Encoded", report.get("failed_user_ordinal_columns", [])),
                    ("Ordinal Encoded Columns", report.get("ordinal_encoded_columns", [])),
                    ("One-Hot Encoded Columns", report.get("one_hot_encoded_columns", [])),
                ])

                if report.get("capped_outlier_columns"):
                    st.subheader("Outlier Handling")
                    outlier_df = build_outlier_dataframe(report)
                    show_dataframe(outlier_df, use_container_width=True)
                    show_list("Columns with Capped Extreme Outliers", report.get("capped_outlier_columns", []))

                if report.get("feature_reduction_applied"):
                    st.subheader("Feature Selection")
                    st.write("Strategy: Variance Threshold + Pairwise Correlation")
                    st.write(f"Low variance threshold: {report.get('low_variance_threshold')}")
                    st.write(f"Pairwise correlation threshold: {report.get('high_correlation_threshold')}")
                    show_list("Removed Low-Variance Features", report.get("removed_low_variance_columns", []))
                    show_list("Removed Highly Correlated Features", report.get("removed_high_correlation_columns", []))

                if report.get("target_encoded") or report.get("target_classes") is not None:
                    st.subheader("Target Information")
                    st.write(f"Target encoded: {report.get('target_encoded')}")
                if report.get("target_classes") is not None:
                    st.write("Target classes:")
                    for cls in report.get("target_classes", []):
                        st.write(f"- {cls}")

            show_section_divider()
            show_section_header("Feature Relationship Overview", "A quick pre-model check of which numeric features move together with the target.", anchor="feature-relationships")

            explain_X = X_explain_reference.copy()
            corr_table = build_target_correlation_table(
                explain_X,
                y,
                target_name=st.session_state["target_column"],
                top_n=10
            )

            corr_heatmap_fig = None

            if not corr_table.empty:
                corr_profile_note = get_correlation_profile_interpretation(corr_table)
                show_info_box(
                    "What this section shows",
                    "This area highlights features that move most clearly with the target before any model explanation is generated. Values near 1 or -1 indicate stronger movement together; values near 0 indicate little straight-line relationship."
                )
                show_story_panel("Quick Read", corr_profile_note)

                corr_col1, corr_col2, corr_col3 = st.columns([0.92, 1.0, 1.08])

                with corr_col1:
                    st.subheader("Strongest Relationships")
                    show_dataframe(corr_table[["Feature", "Correlation with Target"]], use_container_width=True)
                    show_chart_note(
                        "Positive values mean the feature tends to rise as the target rises. Negative values mean it tends to move in the opposite direction. This is pattern-finding, not proof of cause."
                    )

                with corr_col2:
                    corr_profile_fig = plot_correlation_profile_figure(corr_table, top_n=8)
                    if corr_profile_fig is not None:
                        show_chart_frame(corr_profile_fig, use_container_width=True)
                    show_chart_note(
                        "This chart is a faster visual read of the table: bars to the right are positive relationships, and bars to the left are negative relationships."
                    )

                with corr_col3:
                    corr_heatmap_fig = plot_correlation_heatmap_figure(
                        explain_X,
                        y,
                        target_name=st.session_state["target_column"],
                        top_n=8
                    )
                    if corr_heatmap_fig is not None:
                        show_chart_frame(corr_heatmap_fig, use_container_width=True)
                    show_chart_note(
                        "The heatmap checks whether the strongest target-related features also move together with each other. Very similar features may tell the model almost the same story."
                    )

                st.session_state["corr_heatmap_fig"] = corr_heatmap_fig
                st.session_state["corr_profile_fig"] = corr_profile_fig
                st.session_state["corr_table_for_report"] = corr_table.copy()
                st.session_state["corr_profile_note"] = corr_profile_note
            else:
                st.info("A correlation-based overview could not be generated because no suitable numeric features were available after preprocessing.")
                st.session_state["corr_heatmap_fig"] = None
                st.session_state["corr_profile_fig"] = None
                st.session_state["corr_table_for_report"] = None
                st.session_state["corr_profile_note"] = None

            show_section_divider()
            show_section_header("Model Training", "Train models on the processed data and compare which approach predicts the target most reliably.", anchor="model-training")

            detected_problem_type = detect_problem_type(y)

            show_info_box(
                "Choose the prediction type",
                "Use Classification when you want the model to predict categories or labels, such as yes/no, low/medium/high, or class names. Use Regression when you want the model to predict a numeric value, such as price, score, temperature, or quality value."
            )

            auto_label = f"Auto-detect (recommended: {detected_problem_type.capitalize()})"

            problem_type_display = st.radio(
                "What kind of prediction do you want?",
                options=[auto_label, "Classification", "Regression"],
                index=0,
                help="Choose Classification when the target is a label or category. Choose Regression when the target is a continuous number."
            )

            if problem_type_display == auto_label:
                chosen_problem_type = detected_problem_type
            elif problem_type_display == "Classification":
                chosen_problem_type = "classification"
            else:
                chosen_problem_type = "regression"

            st.caption(
                f"Selected mode: {chosen_problem_type.capitalize()} — " +
                (
                    "the model will predict categories or classes."
                    if chosen_problem_type == "classification"
                    else "the model will predict a numeric value."
                )
            )

            available_models = get_available_models(chosen_problem_type)

            training_mode_display = st.radio(
                "How would you like to train models?",
                options=["Compare multiple models", "Train a single model"],
                help="Comparing multiple models gives a broader first read. A single model is useful when you already know which method you want to test."
            )

            training_mode = "multiple" if training_mode_display == "Compare multiple models" else "single"

            selected_model_name = None
            if training_mode == "single":
                selected_model_name = st.selectbox(
                    "Select a model",
                    options=list(available_models.keys()),
                    help="Pick one model to train and evaluate on the current dataset."
                )

            if st.button("Train Models"):
                if X is None or X.empty:
                    st.error("Model training cannot start because no usable feature columns are available after preprocessing.")
                else:
                    with st.spinner("Model training running..."):
                        problem_type, results_df, detailed_results = train_and_evaluate_models(
                            X=X,
                            y=y,
                            training_mode=training_mode,
                            selected_model_name=selected_model_name,
                            class_labels=report.get("target_label_mapping") or report.get("target_classes"),
                            forced_problem_type=chosen_problem_type
                        )

                    st.session_state["problem_type"] = problem_type
                    st.session_state["results_df"] = results_df
                    st.session_state["detailed_results"] = detailed_results
                    st.session_state["kfold_df"] = None
                    st.session_state["kfold_fig"] = None
                    st.session_state["kfold_note"] = None
                    st.session_state["selected_training_mode"] = training_mode
                    st.session_state["selected_model_name"] = selected_model_name
                    queue_toast("Model training completed successfully.")
                    st.rerun()

        if (
                "results_df" in st.session_state
                and "problem_type" in st.session_state
                and "detailed_results" in st.session_state
                and "X_explain_reference" in st.session_state
        ):
            results_df = st.session_state["results_df"]
            problem_type = st.session_state["problem_type"]
            detailed_results = st.session_state["detailed_results"]
            X_explain_reference = st.session_state["X_explain_reference"]

            show_section_divider()
            show_section_header("Results Dashboard", "Compare model performance and decide which result is strong enough to explain or report.", anchor="results-dashboard")

            st.write(f"Prediction task: **{problem_type.capitalize()}**")
            best_model_name, best_metric_name, best_metric_value = get_best_model_info(results_df, problem_type)
            show_dataframe(
                style_best_model_row(results_df, best_metric_name),
                use_container_width=True
            )
            model_story = get_model_recommendation_text(results_df, problem_type)
            if best_model_name is not None:
                show_compact_success(f"Leading model: {best_model_name} ({best_metric_name}: {best_metric_value:.4f})")

            show_summary_cards([
                {
                    "label": "Leading model",
                    "value": best_model_name or "-",
                    "note": "Ranked first on the main evaluation metric.",
                    "href": "#model-ranking",
                    "accent": "#3B82F6"
                },
                {
                    "label": "Primary metric",
                    "value": best_metric_name or "-",
                    "note": f"Main score: {format_metric_value(best_metric_value)}",
                    "href": "#metric-comparisons",
                    "accent": "#8B5CF6"
                },
                {
                    "label": "Models compared",
                    "value": str(len(results_df)),
                    "note": "Trained and tested under the same setup.",
                    "accent": "#10B981"
                },
                {
                    "label": "Decision focus",
                    "value": get_metric_focus_label(problem_type),
                    "note": "A plain-language lens for reading the score.",
                    "accent": "#F59E0B"
                }
            ])
            show_story_panel("Executive Summary", model_story)

            leaderboard_fig = plot_model_leaderboard_figure(results_df, problem_type)
            st.session_state["model_leaderboard_fig"] = leaderboard_fig
            st.session_state["model_recommendation_text"] = model_story
            if leaderboard_fig is not None:
                st.markdown('<span id="model-ranking"></span>', unsafe_allow_html=True)
                st.subheader("Model Ranking")
                show_chart_frame(leaderboard_fig, use_container_width=False)
                show_chart_note(
                    "This chart ranks models by the main success metric. The top model is the strongest current candidate, but close scores should be checked with stability and explainability."
                )

            has_roc_auc = "ROC AUC" in results_df.columns
            with st.expander("Metric explanations"):
                show_metric_explanations(problem_type, has_roc_auc=has_roc_auc)

            show_metric_plots(results_df, problem_type)

            with st.expander("Detailed model explanations"):
                if problem_type == "classification":
                    st.write("Classification models predict categories or labels. The goal is not only to be correct often, but also to understand which kinds of mistakes are being made.")
                    st.write("The Confusion Matrix shows correct predictions on the diagonal and mistakes outside the diagonal.")
                    if has_roc_auc:
                        st.write("The ROC Curve shows whether the model can rank positive cases ahead of negative cases across different thresholds.")
                else:
                    st.write("Regression models predict numeric values. The goal is to keep predictions close to the real values.")
                    st.write("R2 Score summarizes overall fit, while MAE and RMSE show how large the prediction errors are.")

            if problem_type == "classification":
                show_info_box(
                    "Model Selection Guide",
                    "Accuracy is the main ranking metric here because it is the easiest overall correctness score to read. ROC AUC answers a different question: how well the model separates two classes before choosing a cutoff. K-fold checks whether performance stays steady when the train/test split changes. If these views disagree, compare the size of the gap and choose the metric that best matches the decision risk."
                )
            else:
                show_info_box(
                    "Model Selection Guide",
                    "R2 Score is the main ranking metric here because it summarizes overall fit. MAE and RMSE translate performance into error size, which is often easier to explain to a non-technical audience. K-fold checks whether the result stays steady when the train/test split changes."
                )

            kfold_df = st.session_state.get("kfold_df")
            kfold_fig = st.session_state.get("kfold_fig")
            kfold_note = st.session_state.get("kfold_note")
            st.subheader("Optional Stability Check")
            kfold_cost, kfold_cost_note = get_kfold_cost_label(X, len(detailed_results))
            show_info_box(
                "K-Fold Cross-Validation",
                f"Runs the selected models across several train/test splits instead of trusting one split only. This helps answer: would the model still look good if different rows were used for testing? Reported stability metric: {get_primary_stability_metric(problem_type)}. Estimated workload: {kfold_cost}. {kfold_cost_note}"
            )
            if st.button("Run K-Fold Stability Check"):
                with st.spinner("Running K-fold stability check..."):
                    try:
                        all_models = get_available_models(problem_type)
                        models_for_stability = {
                            name: all_models[name]
                            for name in detailed_results.keys()
                            if name in all_models
                        }
                        kfold_df = compute_kfold_stability(
                            models=models_for_stability,
                            X=X,
                            y=y,
                            problem_type=problem_type,
                            n_splits=5
                        )
                        selected_kfold_metric = get_primary_stability_metric(problem_type)
                        kfold_fig = plot_kfold_stability_figure(kfold_df, problem_type, metric_name=selected_kfold_metric)
                        kfold_note = get_kfold_interpretation(
                            kfold_df,
                            problem_type,
                            metric_name=selected_kfold_metric,
                            current_leader=best_model_name
                        )
                    except Exception as stability_error:
                        kfold_df = None
                        kfold_fig = None
                        kfold_note = f"Stability check could not be completed: {stability_error}"
                        selected_kfold_metric = None

                st.session_state["kfold_df"] = kfold_df
                st.session_state["kfold_fig"] = kfold_fig
                st.session_state["kfold_note"] = kfold_note
                st.session_state["kfold_metric_name"] = selected_kfold_metric
                st.rerun()

            if kfold_df is not None or kfold_note:
                selected_kfold_metric = get_primary_stability_metric(problem_type)
                if kfold_df is not None and not kfold_df.empty:
                    kfold_fig = plot_kfold_stability_figure(kfold_df, problem_type, metric_name=selected_kfold_metric)
                    kfold_note = get_kfold_interpretation(
                        kfold_df,
                        problem_type,
                        metric_name=selected_kfold_metric,
                        current_leader=best_model_name
                    )
                    st.session_state["kfold_fig"] = kfold_fig
                    st.session_state["kfold_note"] = kfold_note
                    show_chart_note(
                        f"Stability is evaluated with {selected_kfold_metric}. This is a consistency check, not a replacement for the main ranking. ROC AUC can still favor another model because it measures class separation rather than overall correctness."
                        if problem_type == "classification"
                        else "Stability is evaluated with R2 Score, which measures explained variation across repeated train/test splits."
                    )
                if kfold_fig is not None:
                    kf_left, kf_mid, kf_right = st.columns([0.18, 0.64, 0.18])
                    with kf_mid:
                        show_chart_frame(kfold_fig, use_container_width=False)
                if kfold_df is not None and not kfold_df.empty:
                    compact_cols = ["Model", "Folds"]
                    compact_cols.extend([f"{selected_kfold_metric} Mean", f"{selected_kfold_metric} Std"])
                    compact_cols = [col for col in compact_cols if col in kfold_df.columns]
                    compact_kfold_df = kfold_df[compact_cols].round(4)
                    show_dataframe(
                        compact_kfold_df,
                        column_config=build_kfold_column_config(compact_kfold_df, selected_kfold_metric),
                        use_container_width=True,
                        height=160
                    )
                show_story_panel(
                    "Stability assessment",
                    kfold_note or "Stability results are not available for the current configuration."
                )

            if problem_type == "classification":
                st.subheader("Confusion Matrix")

                model_to_show = st.selectbox(
                    "Select a model for confusion matrix",
                    options=list(detailed_results.keys()),
                    help="Choose which model's correct and incorrect class predictions to inspect."
                )

                selected_details = detailed_results[model_to_show]
                cm = selected_details.get("confusion_matrix")
                class_labels = selected_details.get("class_labels")

                if cm is not None and class_labels is not None:
                    show_chart_frame(plot_confusion_matrix_figure(cm, class_labels), use_container_width=False)
                    show_info_box("Confusion Matrix Insight", get_confusion_matrix_interpretation(cm, class_labels))

                roc_fig = plot_roc_curve_figure(detailed_results)
                if roc_fig is not None:
                    st.subheader("ROC Curve")
                    show_chart_frame(roc_fig, use_container_width=False)
                    show_info_box("ROC Curve Insight", get_roc_interpretation(detailed_results))
                    show_chart_note(
                        "ROC AUC and Accuracy can lead to different model preferences. Use ROC AUC when class separation is the priority; use Accuracy when the overall correct prediction rate is the priority."
                    )

            show_section_divider()
            show_section_header("SHAP Explainability", "Explain which features pushed model predictions up, down, toward a class, or away from it.", anchor="shap-explainability")

            show_info_box(
                "How to choose a model for SHAP",
                get_shap_selection_guidance(problem_type, has_roc_auc=has_roc_auc)
            )

            shap_model_name = st.selectbox(
                "Select the model to explain with SHAP",
                options=list(detailed_results.keys()),
                help="Choose the trained model whose behavior you want to explain. Start with the leading model unless another model is more stable or easier to justify."
            )

            if st.button("Generate SHAP Analysis"):
                selected_shap_details = detailed_results[shap_model_name]
                trained_model = selected_shap_details["trained_model"]

                with st.spinner("Generating SHAP explanation..."):
                    shap_outputs = compute_shap_outputs(
                        trained_model=trained_model,
                        X_reference=X_explain_reference,
                        problem_type=problem_type,
                        max_background_samples=100,
                        max_explain_samples=200
                    )

                    importance_df_cached = shap_outputs["feature_importance_df"]
                    bar_fig_cached = plot_shap_importance_bar(importance_df_cached, top_n=12)
                    summary_fig_cached = plot_shap_summary_figure(
                        shap_outputs["shap_values"],
                        shap_outputs["X_explain"],
                        max_display=12
                    )

                st.session_state["shap_model_name"] = shap_model_name
                st.session_state["shap_outputs"] = shap_outputs
                st.session_state["shap_bar_fig"] = bar_fig_cached
                st.session_state["shap_summary_fig"] = summary_fig_cached
                queue_toast("SHAP explanation completed successfully.")
                st.rerun()

            if "shap_outputs" in st.session_state:
                shap_outputs = st.session_state["shap_outputs"]
                shap_model_name = st.session_state.get("shap_model_name", "Selected Model")
                shap_bar_fig = st.session_state.get("shap_bar_fig")
                shap_summary_fig = st.session_state.get("shap_summary_fig")
                explained_model = detailed_results.get(shap_model_name, {}).get("trained_model")
                positive_class_label = get_positive_class_label(problem_type, report, y)
                positive_class_text = format_positive_class_text(positive_class_label)

                st.subheader(f"SHAP Results for {shap_model_name}")
                show_info_box("What SHAP shows", get_shap_intro_text(problem_type, positive_class_label))

                importance_df = shap_outputs["feature_importance_df"]
                top_feature_name = importance_df.iloc[0]["Feature"] if not importance_df.empty else "-"

                show_summary_cards([
                    {
                        "label": "Explained model",
                        "value": shap_model_name,
                        "note": "All explanation charts below describe this model."
                    },
                    {
                        "label": "Top feature",
                        "value": str(top_feature_name),
                        "note": "Has the strongest average influence in this sample."
                    },
                    {
                        "label": "Samples reviewed",
                        "value": str(len(shap_outputs["X_explain"])),
                        "note": "Rows used to estimate the explanation."
                    },
                    {
                        "label": "Reading mode",
                        "value": "Positive class" if problem_type == "classification" else "Prediction value",
                        "note": (
                            f"Positive SHAP values move the model toward {positive_class_text}."
                            if problem_type == "classification"
                            else "Each chart explains what pushed the predicted value up or down."
                        )
                    }
                ])
                show_story_panel(
                    "Interpretation workflow",
                    "Start with the global ranking to see what matters most overall. Then use the feature behavior charts to see whether a feature usually pushes predictions up, down, toward a class, or away from it."
                )

                shap_col1, shap_col2 = st.columns([1.0, 1.15])

                with shap_col1:
                    st.subheader("Top Influential Features")
                    show_dataframe(importance_df.head(8).round(4), use_container_width=True, height=260)
                    show_chart_note(
                        "Features are ranked by average absolute SHAP impact. Higher values mean the feature changed predictions more strongly on average."
                    )

                with shap_col2:
                    if shap_bar_fig is not None:
                        show_chart_frame(shap_bar_fig, use_container_width=True)
                    show_chart_note(
                        "Longer bars mean stronger average contribution across the analyzed sample. This shows importance, not whether the effect is positive or negative."
                    )

                st.subheader("Overall Distribution View")
                sum_left, sum_mid, sum_right = st.columns([0.12, 0.76, 0.12])
                with sum_mid:
                    if shap_summary_fig is not None:
                        show_chart_frame(shap_summary_fig, use_container_width=True)
                show_chart_note(
                    (
                        f"Each point represents one row. Points to the right increase support for {positive_class_text}; points to the left decrease it."
                        if problem_type == "classification"
                        else "Each point represents one row. Points to the right pushed that row's prediction higher; points to the left pushed it lower."
                    )
                )

                st.subheader("How One Feature Changes the Result")
                st.caption(
                    "Select a feature to see how its values relate to the model's output."
                )

                show_chart_note(
                    (
                        f"This chart describes model behavior, not direct causality. For this classification model, positive effects mean stronger movement toward {positive_class_text}."
                        if problem_type == "classification"
                        else "This chart describes model behavior, not direct causality. It shows how the selected feature aligns with higher or lower predicted values."
                    )
                )

                available_effect_features = shap_outputs["feature_importance_df"]["Feature"].head(12).tolist()

                selected_effect_features = st.multiselect(
                    "Choose features for behavior charts and report",
                    options=available_effect_features,
                    default=available_effect_features[:1],
                    help="Select up to three important features to inspect in more detail and include in the exported report."
                )

                if len(selected_effect_features) > 3:
                    st.warning("For readability, only the first three selected features will be shown and added to the report.")
                    selected_effect_features = selected_effect_features[:3]

                effect_fig = None
                effect_note = None
                pdp_ice_fig = None
                pdp_ice_note = None
                feature_detail_reports = []

                if selected_effect_features:
                    show_info_box(
                        "Feature Behavior Charts",
                        (
                            f"The left chart shows observed SHAP contribution: green points increase support for {positive_class_text}, while red points decrease it. The right chart runs controlled what-if checks by moving the selected feature while keeping other values fixed."
                            if problem_type == "classification"
                            else "The left chart shows observed SHAP contribution: green points pushed predictions higher and red points pushed them lower. The right chart runs controlled what-if checks by moving the selected feature while keeping other values fixed."
                        )
                    )

                    feature_tabs = (
                        st.tabs([str(feature)[:28] for feature in selected_effect_features])
                        if len(selected_effect_features) > 1 else [st.container()]
                    )

                    for tab, selected_effect_feature in zip(feature_tabs, selected_effect_features):
                        with tab:
                            current_effect_fig = plot_shap_feature_effect_figure(
                                shap_values=shap_outputs["shap_values"],
                                X_explain=shap_outputs["X_explain"],
                                feature_name=selected_effect_feature,
                                problem_type=problem_type,
                                positive_class_label=positive_class_label
                            )

                            current_effect_note = get_feature_effect_interpretation(
                                shap_values=shap_outputs["shap_values"],
                                X_explain=shap_outputs["X_explain"],
                                feature_name=selected_effect_feature,
                                problem_type=problem_type,
                                positive_class_label=positive_class_label
                            )

                            current_pdp_ice_fig = None
                            current_pdp_ice_note = None
                            if explained_model is not None:
                                pdp_ice_data = compute_pdp_ice_data(
                                    trained_model=explained_model,
                                    X=shap_outputs["X_explain"],
                                    feature_name=selected_effect_feature,
                                    problem_type=problem_type,
                                    grid_points=12,
                                    ice_samples=30,
                                    positive_class_label=positive_class_label
                                )
                                current_pdp_ice_fig = plot_pdp_ice_figure(pdp_ice_data)
                                current_pdp_ice_note = get_pdp_ice_interpretation(pdp_ice_data)

                            chart_left, chart_right = responsive_pair_columns([1, 1])
                            with chart_left:
                                st.markdown("**Observed contribution**")
                                if current_effect_fig is not None:
                                    show_chart_frame(current_effect_fig, use_container_width=True)
                                show_chart_note(
                                    (
                                        f"Use this chart to see whether the feature mostly moves predictions toward {positive_class_text}, away from it, or behaves differently across rows."
                                        if problem_type == "classification"
                                        else "Use this chart to see whether the feature mostly pushes predicted values upward, downward, or behaves differently across rows."
                                    )
                                )
                            with chart_right:
                                st.markdown("**Controlled feature movement**")
                                if current_pdp_ice_fig is not None:
                                    show_chart_frame(current_pdp_ice_fig, use_container_width=True)
                                show_chart_note(
                                    "Use this chart to see how the model response changes when this feature is moved across its observed value range."
                                )

                            note_left, note_right = responsive_pair_columns([1, 1])
                            with note_left:
                                show_info_box("SHAP Effect Summary", current_effect_note)
                            with note_right:
                                show_story_panel("PDP / ICE Assessment", current_pdp_ice_note or "PDP / ICE assessment is not available for this feature.")

                            feature_detail_reports.append({
                                "feature_name": selected_effect_feature,
                                "effect_fig": current_effect_fig,
                                "effect_note": current_effect_note,
                                "pdp_ice_fig": current_pdp_ice_fig,
                                "pdp_ice_note": current_pdp_ice_note,
                            })

                            if effect_fig is None:
                                effect_fig = current_effect_fig
                                effect_note = current_effect_note
                                pdp_ice_fig = current_pdp_ice_fig
                                pdp_ice_note = current_pdp_ice_note

                behavior_df = build_feature_behavior_summary(shap_outputs, top_n=8, problem_type=problem_type)
                behavior_fig = plot_feature_behavior_summary_figure(behavior_df, top_n=8)

                behavior_col1, behavior_col2 = st.columns([1.05, 0.95])
                with behavior_col1:
                    st.subheader("Overall Feature Behavior")
                    if behavior_fig is not None:
                        show_chart_frame(behavior_fig, use_container_width=True)
                    show_chart_note(
                        (
                            f"This summary chart shows whether high-impact features usually move predictions toward {positive_class_text}, away from it, or behave in a mixed way."
                            if problem_type == "classification"
                            else "This summary chart shows whether high-impact features usually raise the predicted value, lower it, or behave in a mixed way."
                        )
                    )
                with behavior_col2:
                    st.subheader("Behavior Summary Table")
                    show_dataframe(behavior_df.head(6).round(4), use_container_width=True, height=240)
                    show_chart_note(
                        "The table summarizes how strongly each feature tends to matter and the dominant direction seen in the analysis sample."
                    )

                st.session_state["shap_effect_fig"] = effect_fig
                st.session_state["shap_effect_note"] = effect_note
                st.session_state["feature_behavior_df"] = behavior_df.copy()
                st.session_state["feature_behavior_fig"] = behavior_fig
                st.session_state["pdp_ice_fig"] = pdp_ice_fig
                st.session_state["pdp_ice_note"] = pdp_ice_note
                st.session_state["feature_detail_reports"] = feature_detail_reports

                show_section_divider()
                show_download_section(
                    target_column=st.session_state.get("target_column", "target"),
                    report=st.session_state.get("preprocessing_report", {}),
                    results_df=results_df,
                    problem_type=problem_type,
                    shap_outputs=shap_outputs,
                    shap_model_name=shap_model_name,
                    shap_bar_fig=shap_bar_fig,
                    shap_summary_fig=shap_summary_fig,
                    corr_table_for_report=st.session_state.get("corr_table_for_report"),
                    corr_heatmap_fig=st.session_state.get("corr_heatmap_fig"),
                    corr_profile_fig=st.session_state.get("corr_profile_fig"),
                    shap_effect_fig=st.session_state.get("shap_effect_fig"),
                    effect_note=st.session_state.get("shap_effect_note"),
                    model_leaderboard_fig=st.session_state.get("model_leaderboard_fig"),
                    model_recommendation_text=st.session_state.get("model_recommendation_text"),
                    corr_profile_note=st.session_state.get("corr_profile_note"),
                    feature_behavior_df=st.session_state.get("feature_behavior_df"),
                    feature_behavior_fig=st.session_state.get("feature_behavior_fig"),
                    kfold_df=st.session_state.get("kfold_df"),
                    kfold_fig=st.session_state.get("kfold_fig"),
                    kfold_note=st.session_state.get("kfold_note"),
                    pdp_ice_fig=st.session_state.get("pdp_ice_fig"),
                    pdp_ice_note=st.session_state.get("pdp_ice_note"),
                    feature_detail_reports=st.session_state.get("feature_detail_reports"),
                )

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")
