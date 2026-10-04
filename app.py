"""Streamlit dashboard for the Financial News Sentiment project.

Run with:  streamlit run app.py

Needs the files saved by the main notebook (models/ and data/ folders next to this file).
FinBERT is optional: if you trained it in FinBERT_Colab.ipynb and copied the results here,
a second model appears in the sidebar automatically.

Layout:  header  ->  KPI cards  ->  3 tabs (Predict / Data / Models)
"""
import html
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

st.set_page_config(page_title="Financial Sentiment", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- colours and icons
# One fixed colour per sentiment, used in every chart so the meaning never changes.
# Each one also has an icon and a word, so colour is never the only clue.
COLORS = {"Bullish": "#1f9d55", "Bearish": "#e34948", "Neutral": "#8a8a85"}
ICONS = {"Bullish": "🟢 ▲", "Bearish": "🔴 ▼", "Neutral": "⚪ ●"}
ORDER = ["Bearish", "Neutral", "Bullish"]
BLUE = "#2a78d6"

# ---------------------------------------------------------------- styling (works in light and dark mode)
st.markdown("""
<style>
.hero {
    padding: 26px 30px; border-radius: 16px; color: white; margin-bottom: 18px;
    background: linear-gradient(120deg, #0f2a4d 0%, #1f5fae 55%, #2a9d8f 100%);
}
.hero h1 { margin: 0; font-size: 2rem; color: white; }
.hero p  { margin: 6px 0 0 0; opacity: 0.9; font-size: 1.02rem; }
.card {
    padding: 16px 18px; border-radius: 14px;
    background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.25);
}
.card .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7; }
.card .value { font-size: 1.7rem; font-weight: 700; line-height: 1.25; }
.card .sub   { font-size: 0.82rem; opacity: .65; }
.result {
    padding: 22px; border-radius: 16px; text-align: center; color: white;
}
.result .big { font-size: 2.2rem; font-weight: 800; }
.result .small { font-size: 1rem; opacity: .92; }
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    color: white; font-size: 0.8rem; font-weight: 600; white-space: nowrap;
}
.sentence-row { padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,0.2); }
</style>
""", unsafe_allow_html=True)


def card(label, value, sub=""):
    """Small KPI card."""
    st.markdown('<div class="card"><div class="label">' + label + '</div>'
                '<div class="value">' + value + '</div>'
                '<div class="sub">' + sub + '</div></div>', unsafe_allow_html=True)


def show_chart(fig):
    """Draw a plotly chart. Newer Streamlit uses width='stretch', older uses use_container_width."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- load models and data (cached)
@st.cache_resource
def get_baseline():
    return utils.load_baseline()


@st.cache_resource
def get_bert():
    return utils.load_bert()


@st.cache_data
def get_validation_data():
    path = os.path.join("data", "validation_clean.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["sentiment"] = df["label"].map(utils.LABEL_MAP)
    df["word_count"] = df["clean_text"].astype(str).str.split().str.len()
    return df


@st.cache_data
def get_comparison():
    return utils.load_comparison()


# ---------------------------------------------------------------- sidebar: choose the model
st.sidebar.header("⚙️ Model")

model_choices = []
if utils.baseline_available():
    model_choices.append("RNN / LSTM / GRU (best baseline)")
if utils.bert_available():
    model_choices.append("FinBERT (fine-tuned)")

if len(model_choices) == 0:
    st.error("No trained model found. Run the main notebook first so that the `models/` folder is created "
             "(or copy the `models/` folder from the Colab download next to this file).")
    st.stop()

chosen_model = st.sidebar.radio("Which model?", model_choices)
use_bert = chosen_model.startswith("FinBERT")

if use_bert:
    try:
        bert_model, bert_tokenizer = get_bert()
    except ImportError:
        st.error("FinBERT needs the `transformers` library. Run `pip install transformers` and restart the app.")
        st.stop()
    model_name = "FinBERT"
    st.sidebar.info("Using the fine-tuned **FinBERT** model (trained in FinBERT_Colab.ipynb).")
else:
    model, vocab, config = get_baseline()
    model_name = config["rnn_type"].upper()
    st.sidebar.info("Model type: **" + model_name + "**  \n"
                    "(the best of RNN, LSTM and GRU on the validation data)")
    st.sidebar.write("Vocabulary size:", config["vocab_size"])
    st.sidebar.write("Max tweet length:", config["max_len"], "tokens")

if not utils.bert_available():
    st.sidebar.caption("FinBERT is not installed. To add it, train `FinBERT_Colab.ipynb` on Google Colab "
                       "and copy `models/finbert_sentiment` here.")

# ---------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>📈 Financial News Sentiment Predictor</h1>
  <p>Paste a finance tweet, headline or paragraph and find out if the market mood is
  <b>Bullish</b>, <b>Bearish</b> or <b>Neutral</b>.</p>
</div>
""", unsafe_allow_html=True)

val_df = get_validation_data()
comparison = get_comparison()

# ---------------------------------------------------------------- KPI cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    card("Active model", model_name, "used for predictions below")
if comparison is not None:
    best = comparison.iloc[0]
    with k2:
        card("Best macro F1", "{:.3f}".format(best["Val Macro F1"]), best["Model"])
    with k3:
        card("Best accuracy", "{:.1f}%".format(comparison["Val Accuracy"].max() * 100), "validation set")
else:
    with k2:
        card("Best macro F1", "-", "run the notebook")
    with k3:
        card("Best accuracy", "-", "run the notebook")
with k4:
    card("Validation tweets", "{:,}".format(len(val_df)) if val_df is not None else "-", "3 classes")

st.write("")

tab_predict, tab_data, tab_models = st.tabs(["🔮 Predict", "📊 The data", "🏆 Model results"])

# ================================================================= TAB 1: PREDICT
with tab_predict:
    examples = {
        "(write my own)": "",
        "Bullish example": "$TSLA - Tesla upgraded to Buy at Goldman, price target raised.",
        "Bearish example": "$AAPL - Apple downgraded at Morgan Stanley on weak iPhone demand.",
        "Neutral example": "$MSFT to report quarterly earnings on Tuesday after the close.",
        "Paragraph example": "Shares surged after record quarterly profit. Analysts raised their targets. "
                             "However the CFO warned about weaker demand next year.",
    }
    left, right = st.columns([1, 1])
    with left:
        example_name = st.selectbox("Try an example", list(examples.keys()))
        user_text = st.text_area("Your text", value=examples[example_name], height=150,
                                 placeholder="e.g. $NVDA - Nvidia beats estimates and raises guidance")
        predict_clicked = st.button("Predict sentiment", type="primary")

    with right:
        if not predict_clicked:
            st.info("👈 Pick an example or type your own text, then press **Predict sentiment**.")
        elif user_text.strip() == "":
            st.warning("Please type some text first.")
        else:
            if use_bert:
                probs_function = lambda s: utils.bert_probs(s, bert_model, bert_tokenizer)
            else:
                probs_function = lambda s: utils.baseline_probs(s, model, vocab, config["max_len"])

            label, probs, per_sentence = utils.predict_paragraph(user_text, probs_function)

            if label is None:
                st.warning("Could not find any sentence in the text.")
            else:
                confidence = probs[label] * 100
                st.markdown('<div class="result" style="background:' + COLORS[label] + '">'
                            '<div class="small">Predicted sentiment</div>'
                            '<div class="big">' + ICONS[label] + " " + label + '</div>'
                            '<div class="small">' + "{:.1f}".format(confidence) + '% confident</div></div>',
                            unsafe_allow_html=True)

    # ---- details below the two columns (only after a prediction)
    if predict_clicked and user_text.strip() != "" and label is not None:
        st.write("")
        g1, g2 = st.columns(2)

        with g1:
            # gauge: how sure is the model?
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "%", "font": {"size": 44}},
                title={"text": "Model confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS[label], "thickness": 0.35},
                    "steps": [{"range": [0, 50], "color": "rgba(128,128,128,0.12)"},
                              {"range": [50, 75], "color": "rgba(128,128,128,0.22)"},
                              {"range": [75, 100], "color": "rgba(128,128,128,0.32)"}],
                },
            ))
            gauge.update_layout(height=300)
            show_chart(gauge)
            if confidence < 50:
                st.caption("Below 50%: the three classes are close, so treat this prediction as uncertain.")

        with g2:
            # horizontal bars: probability of each class
            bars = go.Figure(go.Bar(
                x=[probs[c] * 100 for c in ORDER],
                y=ORDER,
                orientation="h",
                marker_color=[COLORS[c] for c in ORDER],
                text=["{:.1f}%".format(probs[c] * 100) for c in ORDER],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            bars.update_layout(title="Class probabilities", height=300,
                               xaxis=dict(range=[0, 115], title="probability (%)", showgrid=False),
                               yaxis=dict(title=""))
            show_chart(bars)

        if len(per_sentence) > 1:
            st.subheader("Sentence by sentence")
            st.caption("The overall result is the average of the sentences. "
                       "Look for the sentence that pulls it one way.")
            for sentence, sentence_label in per_sentence:
                st.markdown('<div class="sentence-row"><span class="pill" style="background:'
                            + COLORS[sentence_label] + '">' + ICONS[sentence_label] + " " + sentence_label
                            + '</span> &nbsp; ' + html.escape(sentence) + '</div>', unsafe_allow_html=True)

# ================================================================= TAB 2: DATA
with tab_data:
    if val_df is None:
        st.info("`data/validation_clean.csv` not found. Run the main notebook to create it.")
    else:
        d1, d2 = st.columns(2)

        with d1:
            counts = val_df["sentiment"].value_counts().reindex(ORDER)
            donut = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.6,
                marker=dict(colors=[COLORS[c] for c in counts.index], line=dict(color="rgba(0,0,0,0)", width=2)),
                textinfo="label+percent", sort=False,
                hovertemplate="%{label}: %{value} tweets (%{percent})<extra></extra>",
            ))
            donut.update_layout(title="Class distribution (validation set)", height=380, showlegend=False,
                                annotations=[dict(text="<b>{:,}</b><br>tweets".format(counts.sum()),
                                                  x=0.5, y=0.5, showarrow=False, font=dict(size=18))])
            show_chart(donut)
            st.caption("Neutral is the biggest class, so the data is imbalanced. "
                       "That is why macro F1 is used, and why the model leans towards Neutral.")

        with d2:
            box = px.box(val_df, x="sentiment", y="word_count", color="sentiment",
                         category_orders={"sentiment": ORDER}, color_discrete_map=COLORS,
                         title="Tweet length by sentiment", labels={"word_count": "words", "sentiment": ""})
            box.update_layout(height=380, showlegend=False)
            show_chart(box)
            st.caption("Most tweets are short (about 8 to 15 words), so a maximum length of 40 tokens is plenty.")

        st.subheader("Sample validation tweets")
        pick = st.multiselect("Show only", ORDER, default=ORDER)
        sample = val_df[val_df["sentiment"].isin(pick)]
        if len(sample) > 0:
            show_table(sample[["clean_text", "sentiment"]].sample(min(10, len(sample)), random_state=1)
                       .rename(columns={"clean_text": "Tweet", "sentiment": "True label"}))

# ================================================================= TAB 3: MODEL RESULTS
with tab_models:
    if comparison is None:
        st.info("`data/model_comparison.csv` not found. Run the main notebook to create it.")
    else:
        m1, m2 = st.columns([3, 2])

        with m1:
            table = comparison.sort_values("Val Macro F1")
            names = table["Model"].tolist()
            fig = go.Figure()
            fig.add_trace(go.Bar(y=names, x=table["Val Accuracy"], orientation="h", name="Accuracy",
                                 marker_color="#9ec5f0", text=["{:.3f}".format(v) for v in table["Val Accuracy"]],
                                 textposition="outside"))
            fig.add_trace(go.Bar(y=names, x=table["Val Macro F1"], orientation="h", name="Macro F1",
                                 marker_color=BLUE, text=["{:.3f}".format(v) for v in table["Val Macro F1"]],
                                 textposition="outside"))
            fig.update_layout(title="Accuracy vs macro F1 (validation set)", barmode="group", height=380,
                              xaxis=dict(range=[0, 1.1], showgrid=False), legend=dict(orientation="h", y=-0.15))
            show_chart(fig)

        with m2:
            st.write("**Scores**")
            shown = comparison.copy()
            shown.insert(0, "Rank", range(1, len(shown) + 1))
            show_table(shown.round(3))
            st.caption("Macro F1 treats the three classes equally, so it is the fairer score here. "
                       "A model that always says Neutral gets about 66% accuracy but a very low macro F1.")
            if not os.path.exists(utils.BERT_RESULT_FILE):
                st.caption("FinBERT is not in this table yet (it is trained separately on Colab).")

    # pictures saved by the notebook
    pictures = [("loss_curves.png", "Training and validation loss"),
                ("confusion_matrices.png", "Confusion matrices"),
                ("bert_confusion_matrix.png", "FinBERT confusion matrix")]
    available = [(f, c) for f, c in pictures if os.path.exists(os.path.join("data", f))]
    if available:
        st.subheader("Charts from the notebook")
        for file_name, caption in available:
            st.image(os.path.join("data", file_name), caption=caption)