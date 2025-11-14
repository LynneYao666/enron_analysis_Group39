# =====================================================
#  ENRON — Insider Trading Detection Pipeline (CSV Version)
#  For CSV format: file, message  (raw email text)
# =====================================================

import pandas as pd
import numpy as np
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -------------------------------------
# Phase 0 — Load CSV (YOUR FORMAT)
# -------------------------------------

csv_path = "emails.csv"   # <-- MODIFY THIS PATH

df = pd.read_csv(csv_path)
print("✅ CSV loaded, rows:", len(df))
print(df.head())

# Expecting columns: file, message
if not all(col in df.columns for col in ["file", "message"]):
    raise ValueError("Error: CSV must contain 'file' and 'message' columns")


# -------------------------------------
# Phase 1 — Parse Email Headers
# -------------------------------------

def parse_fields(raw_text):

    # Extract basic headers using regex
    def extract(pattern):
        match = re.search(pattern, raw_text, re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else None

    from_field = extract(r"^From:\s*(.*)$")
    to_field = extract(r"^To:\s*(.*)$")
    subject_field = extract(r"^Subject:\s*(.*)$")
    date_field = extract(r"^Date:\s*(.*)$")

    # Extract body (after first blank line)
    parts = re.split(r"\n\s*\n", raw_text, 1)
    body = parts[1] if len(parts) > 1 else ""

    return pd.Series({
        "From": from_field,
        "To": to_field,
        "Subject": subject_field,
        "Date": date_field,
        "Message": body
    })


print(" Parsing all emails (this may take 10–20 seconds)...")

parsed = df["message"].apply(parse_fields)
emails = parsed.copy()

# Drop empty senders/recipients
emails = emails.dropna(subset=["From", "To"])
emails["From"] = emails["From"].str.lower().str.strip()
emails["To"] = emails["To"].str.lower().str.strip()
emails["Message"] = emails["Message"].astype(str)

print("Parsed emails:", len(emails))
print(emails.head())


# -------------------------------------
# Phase 1 — Build Communication Graph
# -------------------------------------

records = []
for _, row in emails.iterrows():
    tos = str(row["To"]).split(",")
    for t in tos:
        t = t.strip()
        if t:
            records.append({"from": row["From"], "to": t})

edges_df = pd.DataFrame(records)
print("🔗 Edges:", len(edges_df))

G = nx.from_pandas_edgelist(edges_df, "from", "to", create_using=nx.DiGraph())
print(" Graph nodes:", len(G.nodes()), "edges:", len(G.edges()))

centrality = nx.betweenness_centrality(G)
top_people = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]

print("\n Top 15 central people:")
for p, c in top_people:
    print(f"{p:<40} {c:.5f}")

important_people = [p for p, _ in top_people]
filtered = emails[emails["From"].isin(important_people) | emails["To"].isin(important_people)]

filtered.to_csv("phase1_filtered_emails.csv", index=False)
print("\n✅ Phase 1 complete — saved phase1_filtered_emails.csv with", len(filtered), "emails")


# -------------------------------------
# Phase 2 — Keyword Detection
# -------------------------------------

emails = filtered.copy()

# Clean text
emails["clean"] = (
    emails["Message"]
    .str.lower()
    .str.replace(r"[^a-z\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
)

insider_terms = [
    "insider", "sell shares", "buy shares", "buy stock", "sell stock",
    "market rumor", "material nonpublic", "non public",
    "earnings release", "quiet period", "sec", "tip off",
    "inside info", "insider info", "confidential"
]

def score(text):
    return sum(term in text for term in insider_terms)

emails["keyword_score"] = emails["clean"].apply(score)
suspicious = emails[emails["keyword_score"] > 0]

print("\n⚠ Emails with insider keywords:", len(suspicious))


# -------------------------------------
# Phase 3 — Time Filter (2001 Crisis)
# -------------------------------------

emails["Date"] = pd.to_datetime(emails["Date"], errors="ignore")
suspicious["Date"] = pd.to_datetime(suspicious["Date"], errors="ignore")

time_filtered = suspicious[
    suspicious["Date"].between("2001-04-01", "2001-12-31", inclusive="both")
]

print("Emails in crisis window:", len(time_filtered))


if len(time_filtered) >= 10:
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, stop_words="english")
    X = vectorizer.fit_transform(time_filtered["clean"])

    lda = LatentDirichletAllocation(n_components=4, random_state=42)
    topics = lda.fit_transform(X)

    vocab = np.array(vectorizer.get_feature_names_out())

    print("\n LDA Topics:")
    for i, comp in enumerate(lda.components_):
        print(f"Topic {i}: {', '.join(vocab[np.argsort(comp)][-10:])}")

    time_filtered["topic"] = topics.argmax(axis=1)
else:
    time_filtered["topic"] = 0
    print("\n Skipping topic modeling — too few emails.")


# -------------------------------------
# Final — Top 5 Insider Trading Emails
# -------------------------------------

time_filtered["final_score"] = time_filtered["keyword_score"] * 2 + 1

top5 = time_filtered.sort_values("final_score", ascending=False).head(5)
top5.to_csv("top5_insider_trading.csv", index=False)

print("\n TOP 5 insider trading emails saved → top5_insider_trading.csv\n")

for _, row in top5.iterrows():
    print("------------------------------------------------------")
    print("From:", row["From"])
    print("To:", row["To"])
    print("Subject:", row["Subject"])
    print("Date:", row["Date"])
    print("Score:", row["final_score"])
    print("Snippet:", row["Message"][:350].replace("\n", " "))
    print()
