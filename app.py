"""MentorMatch Analytics Dashboard — multipage router (flat layout, no folders).

Every tab is a root-level file registered with st.navigation, so the repo
needs no pages/ or utils/ directories — a flat GitHub upload just works.
"""
import streamlit as st

st.set_page_config(page_title="MentorMatch Analytics", page_icon="🧭",
                   layout="wide", initial_sidebar_state="expanded")

pages = [
    st.Page("tab0_home.py",                       title="Home / Overview",                icon="🧭", default=True),
    st.Page("tab1_data_cleaning.py",              title="1 · Data Cleaning",              icon="🧹"),
    st.Page("tab2_descriptive.py",                title="2 · Descriptive Analytics",      icon="📊"),
    st.Page("tab3_diagnostic.py",                 title="3 · Diagnostic Analytics",       icon="🔬"),
    st.Page("tab4_classification_outcome.py",     title="4 · Career Outcome Model",       icon="🎯"),
    st.Page("tab5_regression_income.py",          title="5 · Income Growth Regression",   icon="📈"),
    st.Page("tab6_classification_conversion.py",  title="6 · Conversion Model",           icon="💳"),
    st.Page("tab7_paid_users.py",                 title="7 · Paid Users",                 icon="💎"),
    st.Page("tab8_clustering.py",                 title="8 · Engagement Segments",        icon="🧩"),
    st.Page("tab9_association_rules.py",          title="9 · Association Rules",          icon="🕸️"),
    st.Page("tab10_findings.py",                  title="10 · Findings & Actions",        icon="✅"),
]
st.navigation(pages).run()
