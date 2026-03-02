import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ElectionIQ · Predicting the Vote",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour  
DEM_BLUE = "#1a6fb5"
REP_RED  = "#c0392b"
NEUTRAL  = "#6c757d"

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 40px 48px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.hero p  { font-size: 1.15rem; opacity: 0.82; margin-top: 8px; }

.card {
    background: white;
    border-radius: 10px;
    padding: 22px 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    margin-bottom: 16px;
    border-left: 5px solid #1a6fb5;
}
.card.red  { border-color: #c0392b; }
.card.gray { border-color: #6c757d; }

.sec-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 3px solid #1a6fb5;
    padding-bottom: 6px;
    margin-bottom: 20px;
}

[data-testid="stSidebar"] { background: #1a1a2e !important; }
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
## Welcome to ElectionIQ

Every four years, billions of dollars are spent trying to answer one question:
**who is going to win?** Campaigns pour resources into swing states, pollsters
survey thousands of voters, and analysts build ever-more-complex models — yet
elections still surprise us.

ElectionIQ takes a different approach. Instead of asking *who people say they'll
vote for*, we ask: **can the demographics of a county alone tell us how it will vote?**

Using US Census data alongside county-level results from the 2016 and 2020
presidential elections, we built a linear regression model that predicts the
Republican vote share of any US county based on socioeconomic indicators —
income, poverty rate, racial composition, employment sector, and more.

---

### 📦 The Data

Our analysis draws on three datasets:

- **County Statistics** — 4,800+ US counties with voting results from 2016 and
  2020 merged with US Census demographics (income, race, employment, poverty)
- **Trump–Biden Polls (2020)** — state-level polling averages from FiveThirtyEight,
  covering pollster grades, sample sizes, and methodology
- **Trump–Clinton Polls (2016)** — equivalent polling data from the 2016 race,
  allowing us to study how polling accuracy and voter sentiment shifted across cycles

All datasets are publicly available — county voting data from the MIT Election Lab,
demographic data from the US Census Bureau, and polling data from FiveThirtyEight.

---

### 🔬 Our Approach

We treat this as a **supervised regression problem**. The target variable is the
Republican presidential vote share at county level. Our features are purely
socioeconomic — no prior vote history, no polling data — to test how much
demographics alone can explain electoral outcomes.

We use **Ordinary Least Squares Linear Regression** for its interpretability:
every coefficient has a clear meaning, making it easy to explain *why* the model
predicts what it does. This matters in a political context where understanding
the drivers is just as valuable as the prediction itself.
""")

#  Data loading 
@st.cache_data
def load_data():
    county     = pd.read_csv("county_statistics.csv")
    polls_biden   = pd.read_csv("trump_biden_polls.csv")
    polls_clinton = pd.read_csv("trump_clinton_polls.csv")
    return county, polls_biden, polls_clinton

try:
    county_df, polls_biden, polls_clinton = load_data()
except Exception as e:
    st.error(f"Could not load data files. Make sure the CSVs are in the same folder as this script.\n\nError: {e}")
    st.stop()

#  Sidebar 
st.sidebar.markdown("## 🗳️ ElectionIQ")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Business Case & Data", "📊 Data Visualization", "🤖 Prediction Model"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Datasets**")
st.sidebar.markdown("• County Statistics (4,800+ counties)\n• Trump–Biden 2020 Polls\n• Trump–Clinton 2016 Polls")
st.sidebar.markdown("---")
st.sidebar.markdown("*NYU · Group Project · 2026*")
st.sidebar.markdown("*NYU · Group Project · Aimee P, Nicole Z, Olivia P*")



if page == "🏠 Business Case & Data":

    st.markdown("""
    <div class="hero">
        <h1>🗳️ ElectionIQ</h1>
        <p>Can socioeconomic demographics predict how a county votes?<br>
        A data-driven approach to understanding American elections.</p>
    </div>
    """, unsafe_allow_html=True)

    # Problem statement
    st.markdown('<div class="sec-title">📌 The Business Problem</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        Political campaigns, news organisations, and policy researchers share a common challenge:
        **predicting electoral outcomes before Election Day**. Doing so helps campaigns allocate
        resources efficiently, journalists set realistic expectations, and academics study the
        structural forces that drive voting behaviour.

        **Our question:** Can we predict the *Republican vote share* in a US county using
        observable socioeconomic indicators — income, race, education, and employment?

        A linear regression model trained on county-level data from the **2016 and 2020
        presidential elections** gives us a transparent, interpretable answer.
        """)

    with col2:
        st.markdown("""
        <div class="card">
            <b>🎯 Target Variable</b><br>
            Republican presidential vote share (%) at county level
        </div>
        <div class="card red">
            <b>📥 Key Predictors</b><br>
            Income, Poverty rate, Race demographics, Employment sector, Unemployment
        </div>
        <div class="card gray">
            <b>💡 Value Delivered</b><br>
            Identify "persuadable" counties where demographics suggest a different outcome is possible
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset overview
    st.markdown('<div class="sec-title">📂 Dataset Overview</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([
        "🏘️ County Statistics", "📋 Trump–Biden 2020 Polls", "📋 Trump–Clinton 2016 Polls"
    ])

    with tab1:
        st.markdown(f"**{len(county_df):,} counties across the United States**")
        st.markdown("Combines voting results (2016 & 2020) with US Census demographics.")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Counties", f"{len(county_df):,}")
        col_b.metric("States", county_df['state'].nunique())
        col_c.metric("Features", len(county_df.columns))
        st.dataframe(county_df.head(8), use_container_width=True)

    with tab2:
        st.markdown(f"**{len(polls_biden):,} poll entries — 2020 presidential race**")
        col_a, col_b = st.columns(2)
        col_a.metric("Unique Polls", polls_biden['poll_id'].nunique())
        col_b.metric("States Covered", polls_biden['state'].nunique())
        st.dataframe(polls_biden.head(8), use_container_width=True)

    with tab3:
        st.markdown(f"**{len(polls_clinton):,} poll entries — 2016 presidential race**")
        col_a, col_b = st.columns(2)
        has_poll_id = 'poll_id' in polls_clinton.columns
        col_a.metric("Unique Polls", polls_clinton['poll_id'].nunique() if has_poll_id else "N/A")
        col_b.metric("States Covered", polls_clinton['state'].nunique())
        st.dataframe(polls_clinton.head(8), use_container_width=True)

    st.markdown("---")

    # Methodology
    st.markdown('<div class="sec-title">🔬 Methodology</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Data Cleaning** — drop nulls, normalise column names, engineer the target variable
    2. **EDA** — visualise distributions, correlations, and geographic patterns
    3. **Feature Selection** — choose socioeconomic predictors; drop collinear variables
    4. **Model Training** — Ordinary Least Squares Linear Regression (Scikit-Learn)
    5. **Evaluation** — R², MAE, residual analysis
    6. **Application** — interactive county-level predictor on the model page
    """)


# data viz
elif page == "📊 Data Visualization":

    st.markdown("""
    <div class="hero">
        <h1>📊 Data Visualization</h1>
        <p>Exploring the socioeconomic patterns behind the American vote.</p>
    </div>
    """, unsafe_allow_html=True)

    df = county_df.copy().dropna(subset=[
        'percentage20_Donald_Trump', 'percentage16_Donald_Trump',
        'Income', 'Poverty', 'White', 'Black', 'Unemployment'
    ])

    # Distribution of Trump vote share 
    st.markdown('<div class="sec-title">1 · Distribution of Republican Vote Share (2020)</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['percentage20_Donald_Trump'], bins=40,
                color=REP_RED, edgecolor='white', alpha=0.88)
        mean_val = df['percentage20_Donald_Trump'].mean()
        ax.axvline(mean_val, color='black', linestyle='--',
                   label=f"Mean = {mean_val:.1%}")
        ax.set_xlabel("Trump Vote Share (2020)", fontsize=12)
        ax.set_ylabel("Number of Counties", fontsize=12)
        ax.set_title("How Republican is a Typical US County?", fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("""
        **Key Insight**

        The distribution is **right-skewed** — most counties vote Republican.
        However, Democratic support concentrates in high-population urban counties,
        so the national popular vote does not reflect this county-level majority.

        > Most US counties are red — but most US *voters* live in blue counties.
        """)
        st.metric("Median Trump Share", f"{df['percentage20_Donald_Trump'].median():.1%}")
        st.metric("Counties > 60% Trump", f"{(df['percentage20_Donald_Trump'] > 0.6).sum():,}")

    # ── Chart 2: 2016 → 2020 swing ────────────────────────────────────────────
    st.markdown('<div class="sec-title">2 · Vote Swing: 2016 → 2020</div>',
                unsafe_allow_html=True)
    df['swing'] = df['percentage20_Donald_Trump'] - df['percentage16_Donald_Trump']
    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['swing'], bins=50, color=REP_RED, alpha=0.6)
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax.set_xlabel("Change in Republican Share (2016 → 2020)", fontsize=12)
        ax.set_ylabel("Number of Counties", fontsize=12)
        ax.set_title("Which Direction Did Counties Swing?", fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("""
        **Key Insight**

        Most counties experienced a **slight shift toward Trump** from 2016 to 2020 —
        even as he lost the national popular vote by a wider margin.

        Republicans made gains in rural and small-town America while losing ground
        in suburbs and large metros.
        """)
        st.metric("Counties swung >2% toward Trump", f"{(df['swing'] > 0.02).sum():,}")
        st.metric("Counties swung >2% toward Biden", f"{(df['swing'] < -0.02).sum():,}")

    # Income vs vote share 
    st.markdown('<div class="sec-title">3 · Income vs Republican Vote Share</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sc = ax.scatter(df['Income'], df['percentage20_Donald_Trump'],
                        alpha=0.25, s=18, c=df['percentage20_Donald_Trump'],
                        cmap='RdBu_r', vmin=0.2, vmax=0.8)
        plt.colorbar(sc, ax=ax, label="Trump Share")
        m, b = np.polyfit(df['Income'], df['percentage20_Donald_Trump'], 1)
        x_line = np.linspace(df['Income'].min(), df['Income'].max(), 200)
        ax.plot(x_line, m * x_line + b, color='black', linewidth=2, label='Trend')
        ax.set_xlabel("Median Household Income ($)", fontsize=12)
        ax.set_ylabel("Trump Vote Share (2020)", fontsize=12)
        ax.set_title("Wealthier Counties ≠ More Republican", fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        corr = df['Income'].corr(df['percentage20_Donald_Trump'])
        st.markdown(f"""
        **Key Insight**

        There is a **negative correlation (r = {corr:.2f})** between income and
        Republican vote share at the county level.

        The old "Republicans = wealthy" pattern has reversed: lower-income rural
        counties now lean strongly Republican, while high-income urban and suburban
        counties have shifted toward Democrats.
        """)
        st.metric("Correlation: Income ↔ Trump%", f"r = {corr:.2f}")

    # Poverty & White share scatter 
    st.markdown('<div class="sec-title">4 · Demographic Breakdown</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df['Poverty'], df['percentage20_Donald_Trump'],
                   alpha=0.2, s=15, color=REP_RED)
        m, b = np.polyfit(df['Poverty'], df['percentage20_Donald_Trump'], 1)
        x_line = np.linspace(df['Poverty'].min(), df['Poverty'].max(), 200)
        ax.plot(x_line, m * x_line + b, color='black', linewidth=2)
        ax.set_xlabel("Poverty Rate (%)", fontsize=11)
        ax.set_ylabel("Trump Vote Share", fontsize=11)
        ax.set_title("Poverty Rate vs Vote Share", fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df['White'], df['percentage20_Donald_Trump'],
                   alpha=0.2, s=15, color=DEM_BLUE)
        m, b = np.polyfit(df['White'], df['percentage20_Donald_Trump'], 1)
        x_line = np.linspace(df['White'].min(), df['White'].max(), 200)
        ax.plot(x_line, m * x_line + b, color='black', linewidth=2)
        ax.set_xlabel("White Population Share (%)", fontsize=11)
        ax.set_ylabel("Trump Vote Share", fontsize=11)
        ax.set_title("White Share vs Vote Share", fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        fig.tight_layout()
        st.pyplot(fig)

    # Correlation heatmap 
    st.markdown('<div class="sec-title">5 · Correlation Heatmap</div>',
                unsafe_allow_html=True)
    feat_cols = [
        'percentage20_Donald_Trump', 'Income', 'Poverty', 'ChildPoverty',
        'White', 'Black', 'Hispanic', 'Unemployment',
        'Professional', 'Service', 'Construction', 'Production'
    ]
    corr_matrix = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r',
                center=0, ax=ax, annot_kws={"size": 9}, linewidths=0.5)
    ax.set_title("Feature Correlations with Trump Vote Share", fontsize=14, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    **White share, Production workers, Construction workers** → positively correlated with Trump vote  
    **Black share, Income, Hispanic share, Professional workers** → negatively correlated
    """)

    # 2020 state-level polling averages 
    st.markdown('<div class="sec-title">6 · Average Trump Poll Numbers by State (2020)</div>',
                unsafe_allow_html=True)

    trump_polls = polls_biden[polls_biden['candidate_name'].str.contains('Trump', na=False)].copy()
    state_avg = (trump_polls.groupby('state')['pct']
                             .mean()
                             .reset_index()
                             .rename(columns={'pct': 'polled_trump_pct'})
                             .sort_values('polled_trump_pct', ascending=True)
                             .tail(20))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [REP_RED if p > 50 else DEM_BLUE for p in state_avg['polled_trump_pct']]
    ax.barh(state_avg['state'], state_avg['polled_trump_pct'],
            color=colors, edgecolor='white', alpha=0.88)
    ax.axvline(50, color='black', linestyle='--', linewidth=1.2)
    ax.set_xlabel("Average Polled Trump % (2020)", fontsize=12)
    ax.set_title("Top 20 States · Average Trump Poll Numbers (2020)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)

elif page == "🤖 Prediction Model":

    st.markdown("""
    <div class="hero">
        <h1>🤖 Prediction Model</h1>
        <p>Linear regression to predict Republican vote share from county demographics.</p>
    </div>
    """, unsafe_allow_html=True)

    # prepare & train 
    FEATURES = [
        'Income', 'Poverty', 'White', 'Black', 'Hispanic',
        'Unemployment', 'Professional', 'Service',
        'Construction', 'Production', 'MeanCommute'
    ]
    TARGET = 'percentage20_Donald_Trump'

    model_df = county_df[FEATURES + [TARGET, 'county', 'state']].dropna()
    X = model_df[FEATURES]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # metrics
    st.markdown('<div class="sec-title">📈 Model Performance</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.3f}", help="Proportion of variance explained by the model")
    c2.metric("Mean Abs. Error", f"{mae:.1%}", help="Average prediction error in percentage points")
    c3.metric("Training Samples", f"{len(X_train):,}")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.35, s=20, color=REP_RED)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
        ax.set_xlabel("Actual Trump Share", fontsize=12)
        ax.set_ylabel("Predicted Trump Share", fontsize=12)
        ax.set_title("Actual vs Predicted", fontsize=13, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(residuals, bins=40, color=NEUTRAL, edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', linestyle='--')
        ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Residual Distribution", fontsize=13, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown('<div class="sec-title">🔍 Feature Importance (Standardised Coefficients)</div>',
                unsafe_allow_html=True)

    coeff_df = (pd.DataFrame({'Feature': FEATURES, 'Coefficient': model.coef_})
                  .sort_values('Coefficient'))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [REP_RED if c > 0 else DEM_BLUE for c in coeff_df['Coefficient']]
    ax.barh(coeff_df['Feature'], coeff_df['Coefficient'],
            color=colors, edgecolor='white', alpha=0.88)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_xlabel("Standardised Coefficient", fontsize=12)
    ax.set_title("What Drives the Republican Vote Share?", fontsize=14, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**
    - 🔴 **Positive** (more Republican): higher White share, Production & Construction workers
    - 🔵 **Negative** (more Democratic): higher Black share, Income, Professional workers, Hispanic share
    """)

    st.markdown("---")

    #  Interactive predictor 
    st.markdown('<div class="sec-title">🎮 Interactive County Predictor</div>',
                unsafe_allow_html=True)
    st.markdown("Adjust the sliders to represent a hypothetical county and see the predicted Republican vote share.")

    col1, col2, col3 = st.columns(3)
    with col1:
        income       = st.slider("Median Income ($)",        20000, 130000, 55000, step=1000)
        poverty      = st.slider("Poverty Rate (%)",         0,     50,     15)
        white_share  = st.slider("White Population (%)",     0,     100,    70)
        black_share  = st.slider("Black Population (%)",     0,     100,    12)
    with col2:
        hispanic     = st.slider("Hispanic Population (%)",  0,     100,    10)
        unemployment = st.slider("Unemployment Rate (%)",    0,     25,     6)
        professional = st.slider("Professional Workers (%)", 10,    60,     30)
        service      = st.slider("Service Workers (%)",      5,     50,     17)
    with col3:
        construction = st.slider("Construction Workers (%)", 0,     30,     8)
        production   = st.slider("Production Workers (%)",   0,     40,     10)
        commute      = st.slider("Mean Commute (min)",       5,     60,     25)

    input_arr = np.array([[income, poverty, white_share, black_share, hispanic,
                           unemployment, professional, service,
                           construction, production, commute]])
    pred = float(np.clip(model.predict(scaler.transform(input_arr))[0], 0, 1))

    st.markdown("---")
    col_pred, col_viz = st.columns([1, 2])

    with col_pred:
        colour = "#c0392b" if pred > 0.5 else "#1a6fb5"
        label  = "🔴 Republican-leaning county" if pred > 0.5 else "🔵 Democratic-leaning county"
        card_class = "card red" if pred > 0.5 else "card"
        st.markdown(f"""
        <div class="{card_class}" style="text-align:center;">
            <div style="font-size:2.5rem; font-weight:800; color:{colour};">{pred:.1%}</div>
            <div style="font-size:1rem; margin-top:4px;">Predicted Republican Share</div>
            <br>
            <div style="font-size:1rem; color:#555;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_viz:
        fig, ax = plt.subplots(figsize=(7, 1.8))
        ax.barh(['Republican', 'Democratic'], [pred, 1 - pred],
                color=[REP_RED, DEM_BLUE], alpha=0.9)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_title("Predicted Vote Share Split", fontsize=12, fontweight='bold')
        for i, v in enumerate([pred, 1 - pred]):
            ax.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=11, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)

    # similar countries
    st.markdown("**📍 Real counties with similar predicted vote share:**")
    model_df = model_df.copy()
    model_df['predicted'] = model.predict(scaler.transform(model_df[FEATURES]))
    similar = (model_df[np.abs(model_df['predicted'] - pred) < 0.05]
               [['county', 'state', TARGET, 'Income', 'Poverty']]
               .head(5))

    if len(similar):
        similar = similar.copy()
        similar.columns = ['County', 'State', 'Actual Trump %', 'Income', 'Poverty %']
        similar['Actual Trump %'] = similar['Actual Trump %'].map(lambda x: f"{x:.1%}")
        similar['Income'] = similar['Income'].map(lambda x: f"${x:,.0f}")
        st.dataframe(similar, use_container_width=True, hide_index=True)
    else:
        st.info("No closely matching counties found for these slider values — try adjusting them.")