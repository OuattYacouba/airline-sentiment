import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud

# BERT (Hugging Face) reste optionnel : lourd (PyTorch), il n'est pas installé dans la version en ligne.
try:
    from transformers import pipeline
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

DATA_PATH = Path(__file__).parent / "Tweets.csv"
SENTIMENTS = ["negative", "neutral", "positive"]
LABELS_FR = {"negative": "Négatif", "neutral": "Neutre", "positive": "Positif"}
COLORS = {"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#27AE60"}
COLORS_FR = {LABELS_FR[k]: v for k, v in COLORS.items()}
# Thème Plotly sombre à fond transparent, pour se fondre dans le thème Streamlit
pio.templates["dashboard"] = go.layout.Template(pio.templates["plotly_dark"])
pio.templates["dashboard"].layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
PLOT = dict(template="dashboard")

# ==========================================================
# CONFIGURATION ET STYLE
# ==========================================================
st.set_page_config(page_title="Airline Sentiment Analyzer", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #21262D;
        border-radius: 12px;
        border: 1px solid #30363D;
        border-left: 5px solid #4B9CD3;
        padding: 12px 16px;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================================
# CHARGEMENT & PRÉPARATION (mis en cache : calculé une seule fois)
# ==========================================================
def clean_text(text):
    """Texte pour le modèle : sans liens ni @mentions (le nom de la compagnie biaiserait l'apprentissage)."""
    text = re.sub(r"https?://\S+", " ", str(text))
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip().lower()


@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()


@st.cache_data(show_spinner="Chargement et analyse des 14 640 tweets…")
def load_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text"])
    df["clean"] = df["text"].map(clean_text)
    df["date"] = pd.to_datetime(df["tweet_created"], utc=True).dt.tz_convert("US/Pacific").dt.date
    blobs = df["text"].map(lambda t: TextBlob(t).sentiment)
    df["polarity"] = blobs.map(lambda s: s.polarity)
    df["subjectivity"] = blobs.map(lambda s: s.subjectivity)
    vader = get_vader()
    df["vader"] = df["text"].map(lambda t: vader.polarity_scores(t)["compound"])
    df["sentiment_fr"] = df["airline_sentiment"].map(LABELS_FR)
    return df


@st.cache_resource(show_spinner="Entraînement du modèle TF-IDF…")
def train_model(texts, labels):
    """TF-IDF (mots et paires de mots) + régression logistique, classes rééquilibrées.
    Entraîné une fois sur tout le jeu de données, indépendamment des filtres."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, C=2, class_weight="balanced")
    model.fit(vectorizer.fit_transform(X_train), y_train)
    y_pred = model.predict(vectorizer.transform(X_test))
    report = classification_report(y_test, y_pred, labels=SENTIMENTS, output_dict=True)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "report": report,
        "cm": confusion_matrix(y_test, y_pred, labels=SENTIMENTS),
        "n_test": len(y_test),
    }
    return vectorizer, model, metrics


@st.cache_resource(show_spinner="Chargement du modèle BERT (première fois : 1 à 2 minutes)…")
def get_bert():
    return pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")


def lexicon_label(score, threshold=0.05):
    return "positive" if score > threshold else "negative" if score < -threshold else "neutral"


df = load_data()
vectorizer, model, metrics = train_model(tuple(df["clean"]), tuple(df["airline_sentiment"]))

# ==========================================================
# SIDEBAR (FILTRES)
# ==========================================================
st.sidebar.title("🎯 Filtres")
st.sidebar.markdown("Les graphiques suivent la sélection ; le modèle est entraîné sur l'ensemble des tweets.")

airlines = sorted(df["airline"].unique().tolist())
selected_airlines = st.sidebar.multiselect("Compagnies aériennes :", options=airlines, default=airlines)

st.sidebar.divider()
st.sidebar.caption(
    "Données : *Twitter US Airline Sentiment* (Kaggle / Figure Eight), "
    "14 640 tweets adressés à 6 compagnies américaines du 16 au 24 février 2015, "
    "sentiment annoté par des humains.")

if not selected_airlines:
    st.title("✈️ Airline Sentiment & NLP Analysis")
    st.warning("Sélectionnez au moins une compagnie dans le panneau de gauche.")
    st.stop()

df_filtered = df[df["airline"].isin(selected_airlines)]

# ==========================================================
# HEADER & KPI
# ==========================================================
st.title("✈️ Airline Sentiment & NLP Analysis")
st.markdown("Ce que les voyageurs disent des compagnies aériennes sur Twitter : "
            "analyse via **TF-IDF**, **TextBlob**, **VADER** et **Machine Learning**.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tweets analysés", f"{len(df_filtered):,}".replace(",", " "))
c2.metric("Part de tweets négatifs", f"{(df_filtered['airline_sentiment'] == 'negative').mean():.0%}")
c3.metric("Polarité moyenne (TextBlob)", f"{df_filtered['polarity'].mean():.2f}",
          help="De -1 (très négatif) à +1 (très positif).")
c4.metric("Précision du modèle", f"{metrics['accuracy']:.0%}",
          help=f"Mesurée sur {metrics['n_test']:,} tweets jamais vus à l'entraînement.".replace(",", " ", 1))

# ==========================================================
# NAVIGATION PAR ONGLETS
# ==========================================================
# Lien direct vers un onglet : ?onglet=global | scores | modele | nuage
TABS = {"global": "📊 Analyse globale", "scores": "🧪 Scores textuels",
        "modele": "🤖 Modèle & simulateur", "nuage": "☁️ Nuage de mots"}
tab1, tab2, tab3, tab4 = st.tabs(list(TABS.values()), default=TABS.get(st.query_params.get("onglet")))

# ----------------------------------------------------------
# TAB 1 : ANALYSE GLOBALE & RAISONS
# ----------------------------------------------------------
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_sent = px.pie(df_filtered, names="sentiment_fr", title="Répartition des sentiments",
                          color="sentiment_fr", color_discrete_map=COLORS_FR, hole=0.45, **PLOT)
        st.plotly_chart(fig_sent, width="stretch")

    with col_b:
        neg_reasons = (df_filtered.loc[df_filtered["airline_sentiment"] == "negative", "negativereason"]
                       .value_counts().sort_values().reset_index())
        fig_reason = px.bar(neg_reasons, x="count", y="negativereason", orientation="h",
                            title="Causes d'insatisfaction (tweets négatifs)",
                            labels={"count": "Tweets", "negativereason": ""},
                            color_discrete_sequence=[COLORS["negative"]], **PLOT)
        st.plotly_chart(fig_reason, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        by_airline = (df_filtered.groupby(["airline", "sentiment_fr"]).size()
                      .groupby(level=0).transform(lambda s: s / s.sum()).rename("part").reset_index())
        order = (by_airline[by_airline["sentiment_fr"] == "Négatif"].sort_values("part")["airline"].tolist())
        fig_air = px.bar(by_airline, x="part", y="airline", color="sentiment_fr", orientation="h",
                         title="Sentiment par compagnie", color_discrete_map=COLORS_FR,
                         category_orders={"airline": order, "sentiment_fr": list(COLORS_FR)},
                         labels={"part": "", "airline": "", "sentiment_fr": ""}, **PLOT)
        fig_air.update_layout(xaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_air, width="stretch")

    with col_d:
        by_day = df_filtered.groupby(["date", "sentiment_fr"]).size().rename("tweets").reset_index()
        fig_day = px.bar(by_day, x="date", y="tweets", color="sentiment_fr", title="Tweets par jour",
                         color_discrete_map=COLORS_FR, category_orders={"sentiment_fr": list(COLORS_FR)},
                         labels={"date": "", "tweets": "Tweets", "sentiment_fr": ""}, **PLOT)
        fig_day.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_day, width="stretch")

# ----------------------------------------------------------
# TAB 2 : POLARITÉ, SUBJECTIVITÉ ET LIMITES DES LEXIQUES
# ----------------------------------------------------------
with tab2:
    st.subheader("Distribution des scores sémantiques")
    st.caption("**Polarité** : de -1 (négatif) à +1 (positif). **Subjectivité** : de 0 (factuel) à 1 (opinion).")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_pol = px.histogram(df_filtered, x="polarity", nbins=50, title="Polarité (TextBlob)",
                               labels={"polarity": "Polarité"}, color_discrete_sequence=["#4B9CD3"], **PLOT)
        fig_pol.add_vline(x=0, line_dash="dash", line_color="white")
        st.plotly_chart(fig_pol, width="stretch")
    with col_p2:
        fig_sub = px.histogram(df_filtered, x="subjectivity", nbins=50, title="Subjectivité (TextBlob)",
                               labels={"subjectivity": "Subjectivité"}, color_discrete_sequence=["#27AE60"], **PLOT)
        st.plotly_chart(fig_sub, width="stretch")

    st.divider()
    st.subheader("🏁 Benchmark par compagnie")
    comp_stats = df_filtered.groupby("airline")[["polarity", "subjectivity"]].mean().reset_index()
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_bp = px.bar(comp_stats.sort_values("polarity"), x="polarity", y="airline", orientation="h",
                        title="Polarité moyenne par compagnie", color="polarity",
                        color_continuous_scale="RdYlGn", labels={"polarity": "Polarité", "airline": ""}, **PLOT)
        st.plotly_chart(fig_bp, width="stretch")
    with col_b2:
        fig_bs = px.bar(comp_stats.sort_values("subjectivity"), x="subjectivity", y="airline", orientation="h",
                        title="Degré d'émotion (subjectivité) par compagnie", color="subjectivity",
                        color_continuous_scale="Purples", labels={"subjectivity": "Subjectivité", "airline": ""}, **PLOT)
        st.plotly_chart(fig_bs, width="stretch")

    st.divider()
    st.subheader("Les lexiques suffisent-ils ?")
    agree = {
        "TextBlob": (df_filtered["polarity"].map(lexicon_label) == df_filtered["airline_sentiment"]).mean(),
        "VADER": (df_filtered["vader"].map(lexicon_label) == df_filtered["airline_sentiment"]).mean(),
        "TF-IDF + régression logistique": metrics["accuracy"],
    }
    l1, l2, l3 = st.columns(3)
    for col, (name, value) in zip((l1, l2, l3), agree.items()):
        col.metric(f"Accord avec les annotations — {name}", f"{value:.0%}")
    st.info(
        "Les lexiques notent les mots un par un : « flight delayed 3 hours » ne contient aucun mot négatif "
        "et passe pour neutre, alors que c'est une plainte. Un modèle entraîné sur des tweets annotés apprend "
        "ce vocabulaire propre au transport aérien. (Le modèle est mesuré sur des tweets jamais vus, "
        "les lexiques sur l'ensemble de la sélection.)")

# ----------------------------------------------------------
# TAB 3 : MACHINE LEARNING (TF-IDF) & SIMULATEUR
# ----------------------------------------------------------
with tab3:
    st.subheader("Classifieur TF-IDF + régression logistique")
    n_test = f"{metrics['n_test']:,}".replace(",", " ")
    st.caption(f"Entraîné sur 80 % des tweets, évalué sur les {n_test} restants. "
               "Classes rééquilibrées : les tweets négatifs représentent 63 % des données.")

    m1, m2 = st.columns(2)
    m1.metric("Précision globale (accuracy)", f"{metrics['accuracy']:.1%}")
    m2.metric("F1-score moyen (macro)", f"{metrics['f1_macro']:.3f}",
              help="Moyenne des F1 des trois classes : ne laisse pas la classe majoritaire masquer les autres.")

    col_r, col_cm = st.columns(2)
    with col_r:
        rep = pd.DataFrame(metrics["report"]).T.loc[SENTIMENTS, ["precision", "recall", "f1-score", "support"]]
        rep.index = [LABELS_FR[s] for s in rep.index]
        rep.columns = ["Précision", "Rappel", "F1", "Tweets"]
        st.markdown("**Performance par classe**")
        st.dataframe(rep.style.format({"Précision": "{:.0%}", "Rappel": "{:.0%}", "F1": "{:.2f}", "Tweets": "{:.0f}"}),
                     width="stretch")
        st.caption("**Précision** : quand le modèle annonce cette classe, à quelle fréquence a-t-il raison. "
                   "**Rappel** : part des tweets de cette classe qu'il retrouve.")
    with col_cm:
        names = [LABELS_FR[s] for s in SENTIMENTS]
        fig_cm = px.imshow(metrics["cm"], text_auto=True, x=names, y=names, color_continuous_scale="Blues",
                           labels={"x": "Prédit", "y": "Réel", "color": "Tweets"}, title="Matrice de confusion", **PLOT)
        st.plotly_chart(fig_cm, width="stretch")

    with st.expander("🔍 Les expressions qui pèsent le plus dans la décision"):
        terms = np.array(vectorizer.get_feature_names_out())
        cols = st.columns(3)
        for col, cls in zip(cols, model.classes_):
            coefs = model.coef_[list(model.classes_).index(cls)]
            top = terms[np.argsort(coefs)[::-1][:12]]
            col.markdown(f"**{LABELS_FR[cls]}**")
            col.markdown("\n".join(f"- {t}" for t in top))

    st.divider()
    st.subheader("🔮 Simulateur de tweet")
    with st.expander("📝 Exemples à tester"):
        st.markdown(
            "- **Positif :** *Amazing flight! The cabin crew was super professional and we landed ahead of schedule.*\n"
            "- **Négatif :** *Worst service ever. Flight cancelled without any notice. Never flying with them again!*\n"
            "- **Sans mot négatif :** *Still waiting for my bag, 3 hours after landing.*\n"
            "- **Sarcasme (limite des modèles) :** *Oh great, another 3 hours delay. Thanks for nothing!*")

    user_tweet = st.text_input("Saisissez un tweet en anglais :", "The flight was delayed but the staff was very helpful.")

    if user_tweet.strip():
        probas = model.predict_proba(vectorizer.transform([clean_text(user_tweet)]))[0]
        prediction = model.classes_[int(np.argmax(probas))]
        blob = TextBlob(user_tweet).sentiment
        vader_score = get_vader().polarity_scores(user_tweet)["compound"]

        r1, r2, r3 = st.columns(3)
        r1.markdown(f"**Sentiment prédit (TF-IDF)**<br><span style='color:{COLORS[prediction]}; font-size:26px; "
                    f"font-weight:600'>{LABELS_FR[prediction]}</span> — confiance {probas.max():.0%}",
                    unsafe_allow_html=True)
        r2.metric("Polarité TextBlob", f"{blob.polarity:.2f}", help=f"→ {LABELS_FR[lexicon_label(blob.polarity)]}")
        r3.metric("Score VADER", f"{vader_score:.2f}", help=f"→ {LABELS_FR[lexicon_label(vader_score)]}")

        proba_df = pd.DataFrame({"Sentiment": [LABELS_FR[c] for c in model.classes_], "Probabilité": probas})
        fig_p = px.bar(proba_df, x="Probabilité", y="Sentiment", orientation="h", color="Sentiment",
                       color_discrete_map=COLORS_FR, range_x=[0, 1], height=200, **PLOT)
        fig_p.update_layout(showlegend=False, xaxis_tickformat=".0%", margin=dict(t=10, b=10))
        st.plotly_chart(fig_p, width="stretch")

    st.divider()
    st.subheader("🤖 Pour aller plus loin : BERT (Deep Learning)")
    st.markdown(
        "Le TF-IDF compte les mots ; **BERT** lit la phrase dans son ensemble et saisit mieux le contexte "
        "(négations, « but », parfois le sarcasme). Le modèle utilisé, *BERTweet*, a été entraîné sur des tweets.")
    if BERT_AVAILABLE:
        if user_tweet.strip() and st.button("Comparer avec BERT"):
            bert_res = get_bert()(user_tweet)[0]
            bert_labels = {"POS": "Positif", "NEG": "Négatif", "NEU": "Neutre"}
            st.success(f"**BERT prédit : {bert_labels.get(bert_res['label'], bert_res['label'])}** "
                       f"— confiance {bert_res['score']:.0%}")
    else:
        st.info("BERT n'est pas activé dans la version en ligne : avec PyTorch, il dépasse la mémoire de "
                "l'hébergement gratuit. En local : `pip install -r requirements-bert.txt`, puis relancer l'application.")

# ----------------------------------------------------------
# TAB 4 : NUAGES DE MOTS
# ----------------------------------------------------------
with tab4:
    st.subheader("Visualisation des mots dominants")
    present = [s for s in SENTIMENTS if s in set(df_filtered["airline_sentiment"])]
    selected_sent = st.selectbox("Sentiment :", present, format_func=LABELS_FR.get)

    text_cloud = " ".join(df_filtered.loc[df_filtered["airline_sentiment"] == selected_sent, "clean"])
    stopwords = STOPWORDS | {"flight", "flights", "amp", "will", "now", "get", "us", "im", "u"}
    color_map = {"negative": "Reds", "neutral": "Greys", "positive": "Greens"}
    wordcloud = WordCloud(background_color="#161B22", width=1200, height=550, stopwords=stopwords,
                          colormap=color_map[selected_sent], random_state=42).generate(text_cloud)
    fig_wc, ax = plt.subplots(figsize=(12, 5.5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    fig_wc.patch.set_facecolor("#161B22")
    st.pyplot(fig_wc)
    st.caption("Mentions @compagnie, liens et mots trop fréquents (« flight ») retirés pour faire ressortir le fond.")

st.markdown("---")
st.caption("Projet Master 2 Data Management — Yacouba Ouattara · NLP Airline Sentiment Dashboard")
