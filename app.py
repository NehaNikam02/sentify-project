
from flask import Flask
import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

app = Flask(__name__)
from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/product_select")
def product_select():
    return render_template("product_select.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")
@app.route("/purchase")
def purchase():
    return render_template("purchase.html")

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets")

# ---------- SENTIMENT ANALYZER ----------
sia = SentimentIntensityAnalyzer()


# ---------- CORE ANALYSIS FUNCTION ----------
def analyze_sentiment_and_emotion(df, text_column):
    positive = negative = neutral = 0
    happy = sad = angry = emotion_neutral = 0

    for text in df[text_column].dropna():
        compound = sia.polarity_scores(str(text))["compound"]

        if compound >= 0.05:
            positive += 1
        elif compound <= -0.05:
            negative += 1
        else:
            neutral += 1

        if compound >= 0.6:
            happy += 1
        elif compound <= -0.6:
            angry += 1
        elif -0.6 < compound <= -0.05:
            sad += 1
        else:
            emotion_neutral += 1

    total = positive + negative + neutral
    if total == 0:
        return None

    score = ((positive - negative) / total) * 100 + 50
    score = max(0, min(100, round(score, 2)))

    return {
        "positive": round((positive / total) * 100, 2),
        "negative": round((negative / total) * 100, 2),
        "neutral": round((neutral / total) * 100, 2),
        "happy": round((happy / total) * 100, 2),
        "sad": round((sad / total) * 100, 2),
        "angry": round((angry / total) * 100, 2),
        "emotion_neutral": round((emotion_neutral / total) * 100, 2),
        "score": score,
        "total": total
    }


# ---------- HOME ----------
@app.route("/results/<product>/<brand>")
def results(product, brand):

    product = product.lower()
    brand = brand.lower()

    # ---------- LOAD DATA ----------
    if product == "mobile":
        df = pd.read_csv(os.path.join(DATASET_PATH, "mobile_reviews.csv"))
        df = df[df["brand"].str.lower() == brand]
        text_column = "review_text"

    elif product == "laptop":
        df = pd.read_csv(os.path.join(DATASET_PATH, "laptop_reviews.csv"))
        df = df[df["product_name"].str.contains(brand, case=False, na=False)]
        text_column = "review"

    elif product == "headphones":
        df = pd.read_csv(os.path.join(DATASET_PATH, "headphones_reviews.csv"))
        df = df[df["Product"].str.contains(brand, case=False, na=False)]
        text_column = df.columns[1]

    elif product == "smart":
        df = pd.read_csv(os.path.join(DATASET_PATH, "amazon_alexa.tsv"), sep="\t")
        text_column = "verified_reviews"

    else:
        return "Invalid product ❌"

    if df.empty:
        return "No reviews found ❌"

    r = analyze_sentiment_and_emotion(df, text_column)

    # ---------- DECISION LOGIC ----------
    if r["negative"] > 25:
        decision_title = "Not Recommended ❌"
        decision_color = "#EF4444"
        decision_text = (
            "A significant portion of users have expressed dissatisfaction. "
            "Negative sentiment outweighs trust signals, indicating potential concerns "
            "related to performance, pricing, or reliability."
        )
    else:
        decision_title = "Recommended to Buy ✅"
        decision_color = "#22C55E"
        decision_text = (
            "Overall public sentiment shows strong satisfaction and confidence in the product. "
            "Positive emotional signals outweigh negative feedback, indicating trust in performance "
            "and user experience."
        )

    category_map = {
        "mobile": "Mobile Phones",
        "laptop": "Laptops",
        "headphones": "Headphones",
        "smart": "Smart Devices"
    }

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SENTIFY | Sentiment Results</title>

<style>
* {{
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: "Segoe UI", system-ui, sans-serif;
}}

body {{
  min-height: 100vh;
  background: #020617;
  color: #E5E7EB;
}}

.main {{
  padding: 90px 7%;
}}

.header h1 {{
  font-size: 3rem;
  color: #38BDF8;
  margin-bottom: 10px;
}}

.header p {{
  max-width: 900px;
  font-size: 1.05rem;
  color: #CBD5F5;
  line-height: 1.85;
}}

.product-box {{
  margin-top: 40px;
  padding: 24px 28px;
  border-radius: 20px;
  background: rgba(15,23,42,0.85);
  display: flex;
  gap: 40px;
  flex-wrap: wrap;
}}

.product-box span {{
  font-size: 0.85rem;
  color: #93C5FD;
}}

.decision {{
  margin-top: 70px;
  padding: 45px;
  border-radius: 28px;
  background: rgba(15,23,42,0.9);
}}

.decision h2 {{
  font-size: 2.5rem;
  color: {decision_color};
  margin-bottom: 12px;
}}

.metrics {{
  margin-top: 60px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px,1fr));
  gap: 26px;
}}

.metric {{
  padding: 26px;
  border-radius: 22px;
  background: rgba(15,23,42,0.85);
}}

.metric span {{
  font-size: 0.85rem;
  color: #93C5FD;
}}

.metric h3 {{
  margin-top: 10px;
  font-size: 2.2rem;
}}

.emotions {{
  margin-top: 70px;
}}

.emotion-bar {{
  margin-bottom: 18px;
}}

.bar {{
  height: 14px;
  border-radius: 10px;
  background: rgba(148,163,184,0.2);
  overflow: hidden;
}}

.fill {{
  height: 100%;
}}

.happy {{ background:#22C55E; }}
.neutral {{ background:#38BDF8; }}
.angry {{ background:#EF4444; }}
.sad {{ background:#F59E0B; }}

.insight {{
  margin-top: 80px;
  padding: 35px;
  border-radius: 24px;
  background: rgba(15,23,42,0.85);
}}

.actions {{
  margin-top: 60px;
  display: flex;
  gap: 24px;
}}

.primary {{
  background: #38BDF8;
  padding: 14px 38px;
  border-radius: 40px;
  border: none;
  cursor: pointer;
}}

.secondary {{
  background: transparent;
  border: 2px solid #38BDF8;
  color: #38BDF8;
  padding: 14px 38px;
  border-radius: 40px;
  cursor: pointer;
}}
</style>
</head>

<body>
<div class="main">

<div class="header">
  <h1>Sentiment Analysis Results</h1>
  <p>Based on large-scale review and emotional analysis, the following insights summarize user perception.</p>
</div>

<div class="product-box">
  <div><span>Product Category</span><br><strong>{category_map[product]}</strong></div>
  <div><span>Brand / Product</span><br><strong>{brand.capitalize()}</strong></div>
  <div><span>Data Source</span><br><strong>Public Reviews</strong></div>
</div>

<div class="decision">
  <h2>{decision_title}</h2>
  <p>{decision_text}</p>
</div>

<div class="metrics">
  <div class="metric"><span>Overall Sentiment Score</span><h3>{r["score"]} / 100</h3></div>
  <div class="metric"><span>Positive Mentions</span><h3>{r["positive"]}%</h3></div>
  <div class="metric"><span>Negative Mentions</span><h3>{r["negative"]}%</h3></div>
  <div class="metric"><span>Total Reviews Analyzed</span><h3>{r["total"]}</h3></div>
</div>

<div class="emotions">
  <h3>Emotion Distribution</h3>

  <div class="emotion-bar">
    Happy {r["happy"]}%
    <div class="bar"><div class="fill happy" style="width:{r["happy"]}%"></div></div>
  </div>

  <div class="emotion-bar">
    Neutral {r["emotion_neutral"]}%
    <div class="bar"><div class="fill neutral" style="width:{r["emotion_neutral"]}%"></div></div>
  </div>

  <div class="emotion-bar">
    Angry {r["angry"]}%
    <div class="bar"><div class="fill angry" style="width:{r["angry"]}%"></div></div>
  </div>

  <div class="emotion-bar">
    Sad {r["sad"]}%
    <div class="bar"><div class="fill sad" style="width:{r["sad"]}%"></div></div>
  </div>
</div>

<div class="insight">
  <h3>Decision Insight</h3>
  <p>{decision_text}</p>
</div>

<div class="actions">
  <button class="primary" onclick="window.location.href='/dashboard/{product}/{brand}'">

    View Detailed Dashboard
  </button>
  <button class="secondary" onclick="window.location.href='http://127.0.0.1:5500/analysis.html'">
    Analyze Another Product
  </button>

  <button class="primary" 
  onclick="window.location.href='http://127.0.0.1:5500/purchase.html?product={product}&brand={brand}'"

  style="background:#22C55E;">
    Purchase Product 
  </button>
</div>

</div>
</body>
</html>
"""

@app.route("/dashboard/<product>/<brand>")
def dashboard(product, brand):

    product = product.lower()
    brand = brand.lower()

    if product == "mobile":
        df = pd.read_csv(os.path.join(DATASET_PATH, "mobile_reviews.csv"))
        df = df[df["brand"].str.lower() == brand]
        text_column = "review_text"

    elif product == "laptop":
        df = pd.read_csv(os.path.join(DATASET_PATH, "laptop_reviews.csv"))
        df = df[df["product_name"].str.contains(brand, case=False, na=False)]
        text_column = "review"

    elif product == "headphones":
        df = pd.read_csv(os.path.join(DATASET_PATH, "headphones_reviews.csv"))
        df = df[df["Product"].str.contains(brand, case=False, na=False)]
        text_column = df.columns[1]

    elif product == "smart":
        df = pd.read_csv(os.path.join(DATASET_PATH, "amazon_alexa.tsv"), sep="\t")
        text_column = "verified_reviews"

    else:
        return "Invalid product ❌"

    if df.empty:
        return "No reviews found ❌"

    r = analyze_sentiment_and_emotion(df, text_column)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SENTIFY | Analytics Dashboard</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
* {{
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: "Segoe UI", system-ui, sans-serif;
}}

body {{
  min-height: 100vh;
  background: url("/static/bg_image.png") no-repeat center/cover fixed;
  color: #E5E7EB;
}}

.overlay {{
  position: fixed;
  inset: 0;
  background: linear-gradient(rgba(3,7,18,0.9),rgba(3,7,18,0.95));
  z-index: 0;
}}

.main {{
  position: relative;
  z-index: 1;
  padding: 90px 6%;
}}

.header h1 {{
  font-size: 3rem;
  color: #38BDF8;
  margin-bottom: 16px;
}}

.header p {{
  font-size: 1.05rem;
  color: #CBD5F5;
  line-height: 1.8;
}}

.kpi-grid {{
  margin-top: 60px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 26px;
}}

.kpi {{
  background: rgba(15,23,42,0.8);
  border-radius: 22px;
  padding: 26px;
  backdrop-filter: blur(14px);
}}

.kpi span {{
  font-size: 0.85rem;
  color: #93C5FD;
}}

.kpi h2 {{
  margin-top: 12px;
  font-size: 2.4rem;
}}

.grid {{
  margin-top: 70px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 32px;
}}

.panel {{
  background: rgba(15,23,42,0.85);
  border-radius: 26px;
  padding: 28px;
  backdrop-filter: blur(14px);
}}

.panel h3 {{
  margin-bottom: 16px;
  color: #38BDF8;
}}

canvas {{
  width: 100% !important;
  height: 220px !important;
}}

.footer {{
  margin-top: 80px;
}}

.footer a {{
  color: #38BDF8;
  text-decoration: none;
}}
</style>
</head>

<body>
<div class="overlay"></div>

<div class="main">

<div class="header">
  <h1>Sentiment Analytics Dashboard</h1>
  <p>
    Detailed analytics visualization for {brand.capitalize()} ({product.capitalize()} category).
  </p>
</div>

<div class="kpi-grid">
  <div class="kpi">
    <span>Total Reviews</span>
    <h2>{r["total"]}</h2>
  </div>
  <div class="kpi">
    <span>Positive Sentiment</span>
    <h2>{r["positive"]}%</h2>
  </div>
  <div class="kpi">
    <span>Negative Sentiment</span>
    <h2>{r["negative"]}%</h2>
  </div>
  <div class="kpi">
    <span>Neutral Sentiment</span>
    <h2>{r["neutral"]}%</h2>
  </div>
</div>

<div class="grid">

  <div class="panel">
    <h3>Sentiment Distribution</h3>
    <canvas id="sentimentChart"></canvas>
  </div>

  <div class="panel">
    <h3>Emotion Breakdown</h3>
    <canvas id="emotionChart"></canvas>
  </div>

  <div class="panel">
    <h3>Overall Sentiment Score Trend</h3>
    <canvas id="trendChart"></canvas>
  </div>

  <div class="panel">
    <h3>Insight Summary</h3>
    <p>
      Product shows <strong>{r["positive"]}% positive</strong> sentiment.
      Emotional analysis indicates <strong>{r["happy"]}% happiness</strong> among users.
      Negative signals remain at <strong>{r["negative"]}%</strong>, requiring monitoring.
    </p>
  </div>
  <div class="panel">
  <h3>AI Purchase Confidence Meter</h3>
  <canvas id="confidenceGauge"></canvas>
</div>

<div class="panel">
  <h3>Risk Radar Analysis</h3>
  <canvas id="riskRadar"></canvas>
</div>


</div>

<div class="footer">
  <a href="/results/{product}/{brand}">← Back to Results</a>
</div>

</div>

<script>

new Chart(document.getElementById("sentimentChart"), {{
    type: "pie",
    data: {{
        labels: ["Positive","Negative","Neutral"],
        datasets: [{{
            data: [{r["positive"]}, {r["negative"]}, {r["neutral"]}],
            backgroundColor: ["#22C55E","#EF4444","#FACC15"]
        }}]
    }},
    options: {{
        plugins: {{
            legend: {{
                labels: {{ color:"#E5E7EB" }}
            }}
        }}
    }}
}});

new Chart(document.getElementById("emotionChart"), {{
    type: "bar",
    data: {{
        labels: ["Happy","Neutral","Angry","Sad"],
        datasets: [{{
            label: "Emotion %",
            data: [{r["happy"]}, {r["emotion_neutral"]}, {r["angry"]}, {r["sad"]}],
            backgroundColor: "#38BDF8"
        }}]
    }},
    options: {{
        scales: {{
            x: {{ ticks: {{ color:"#E5E7EB" }} }},
            y: {{ ticks: {{ color:"#E5E7EB" }} }}
        }},
        plugins: {{
            legend: {{
                labels: {{ color:"#E5E7EB" }}
            }}
        }}
    }}
}});

new Chart(document.getElementById("trendChart"), {{
    type: "line",
    data: {{
        labels: ["Start","Mid","Current"],
        datasets: [{{
            label: "Sentiment Score",
            data: [50, 60, {r["score"]}],
            borderColor: "#38BDF8",
            backgroundColor: "rgba(56,189,248,0.2)",
            fill: true,
            tension: 0.4
        }}]
    }},
    options: {{
        scales: {{
            x: {{ ticks: {{ color:"#E5E7EB" }} }},
            y: {{ ticks: {{ color:"#E5E7EB" }} }}
        }},
        plugins: {{
            legend: {{
                labels: {{ color:"#E5E7EB" }}
            }}
        }}
    }}
}});
// CONFIDENCE GAUGE
new Chart(document.getElementById("confidenceGauge"), {{
    type: "doughnut",
    data: {{
        labels: ["Confidence","Remaining Risk"],
        datasets: [{{
            data: [{r["score"]}, {100 - r["score"]}],
            backgroundColor: ["#22C55E","#1E293B"],
            borderWidth: 0
        }}]
    }},
    options: {{
        cutout: "75%",
        plugins: {{
            legend: {{ display:false }}
        }}
    }}
}});

// RISK RADAR
new Chart(document.getElementById("riskRadar"), {{
    type: "radar",
    data: {{
        labels: ["Negative","Anger","Sadness","Neutral","Positive"],
        datasets: [{{
            label: "Risk Distribution",
            data: [
                {r["negative"]},
                {r["angry"]},
                {r["sad"]},
                {r["neutral"]},
                {r["positive"]}
            ],
            backgroundColor: "rgba(56,189,248,0.2)",
            borderColor: "#38BDF8"
        }}]
    }},
    options: {{
        scales: {{
            r: {{
                angleLines: {{ color:"#334155" }},
                grid: {{ color:"#334155" }},
                pointLabels: {{ color:"#E5E7EB" }},
                ticks: {{ display:false }}
            }}
        }},
        plugins: {{
            legend: {{
                labels: {{ color:"#E5E7EB" }}
            }}
        }}
    }}
}});



</script>

</body>
</html>
"""




# ---------- RUN ----------
if __name__ == "__main__":

    app.run(debug=True)




