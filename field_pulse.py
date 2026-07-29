"""
Field Pulse — a daily digest of new research papers in your niche,
ranked by relevance to papers you already care about, with an AI
note on why each one matters (not just a summary).

Costs $0 to run: OpenAlex is free and needs no key, embeddings run
locally on your Mac for free, and Gemini's API free tier (no credit
card, no expiry) covers the "why it matters" step at this volume.

Run manually:   python field_pulse.py
"""

import os
import json
import sqlite3
import datetime
import webbrowser
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# ---------- CONFIG (safe to tweak) ----------
SEED_FILE = "seed.json"
DB_FILE = "field_pulse.db"
OUTPUT_DIR = "digests"
TOP_N = 5                        # how many papers to include per digest
LOOKBACK_DAYS = 2                # how many days back counts as "new"
MODEL_NAME = "all-MiniLM-L6-v2"  # local, free embedding model
GEMINI_MODEL = "gemini-3.5-flash"
CONTACT_EMAIL = "you@example.com"  # any email; OpenAlex gives faster, more reliable responses to identified requests — a courtesy header, not a login

# ---------- SETUP ----------
embedder = SentenceTransformer(MODEL_NAME)
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_papers (
            openalex_id TEXT PRIMARY KEY,
            seen_on TEXT
        )
    """)
    conn.commit()
    return conn


def reconstruct_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]} to save space.
    This puts the words back in order so we have real text to embed."""
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def load_seed():
    with open(SEED_FILE) as f:
        return json.load(f)


def fetch_openalex_work_by_doi(doi):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    r = requests.get(url, params={"mailto": CONTACT_EMAIL}, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_recent_papers(search_terms, lookback_days):
    since = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
    query = " OR ".join(search_terms)
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": f"from_publication_date:{since}",
        "sort": "publication_date:desc",
        "per-page": 50,
        "mailto": CONTACT_EMAIL,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def embed_text(text):
    return embedder.encode(text, normalize_embeddings=True)


def cosine(a, b):
    return float(np.dot(a, b))


def build_seed_centroid(seed_dois):
    vectors, titles = [], []
    for doi in seed_dois:
        work = fetch_openalex_work_by_doi(doi)
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        text = (work.get("title") or "") + ". " + abstract
        vectors.append(embed_text(text))
        titles.append(work.get("title"))
    centroid = np.mean(vectors, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return centroid, titles


def rank_new_papers(papers, centroid, conn):
    scored = []
    for p in papers:
        oid = p.get("id")
        already_seen = conn.execute(
            "SELECT 1 FROM seen_papers WHERE openalex_id = ?", (oid,)
        ).fetchone()
        if already_seen:
            continue
        abstract = reconstruct_abstract(p.get("abstract_inverted_index"))
        if not abstract:
            continue  # can't judge relevance without an abstract, so skip it
        text = (p.get("title") or "") + ". " + abstract
        vec = embed_text(text)
        score = cosine(vec, centroid)
        scored.append((score, p, abstract))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:TOP_N]


def summarize_relevance(field_name, seed_titles, paper_title, paper_abstract):
    import time
    from google.genai import errors

    prompt = f"""You help a researcher working on: {field_name}.
Papers they already care about: {"; ".join(t for t in seed_titles if t)}.

New paper:
Title: {paper_title}
Abstract: {paper_abstract}

In 2 concise sentences, explain specifically why this new paper matters
given the papers they already follow - not a generic summary of the abstract.
If it's genuinely minor or off-topic, say so plainly instead of overselling it."""

    # Models to try in order if Google's servers return a 503 error
models_to_try = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                resp = gemini_client.models.generate_content(
                    model=model_name, 
                    contents=prompt
                )
                return resp.text.strip()
            except errors.APIError as e:
                print(f"API busy for {model_name} (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                break

    raise RuntimeError("All Gemini model attempts failed.")


def build_html(field_name, entries, out_path):
    rows = ""
    for score, p, note in entries:
        title = p.get("title") or "(untitled)"
        link = p.get("id")
        rows += f"""
        <div style="margin-bottom:24px;padding:16px;border:1px solid #ddd;border-radius:8px;">
            <div style="font-size:12px;color:#888;">match score: {score:.2f}</div>
            <a href="{link}" style="font-size:17px;font-weight:600;color:#111;text-decoration:none;">{title}</a>
            <p style="margin-top:8px;color:#333;line-height:1.5;">{note}</p>
        </div>"""
    html = f"""<html><head><meta charset="utf-8"><title>Field Pulse — {field_name}</title></head>
    <body style="font-family:-apple-system,sans-serif;max-width:680px;margin:40px auto;padding:0 16px;">
    <h2>Field Pulse — {field_name}</h2>
    <div style="color:#888;margin-bottom:20px;">{datetime.date.today().isoformat()}</div>
    {rows if rows else "<p>No new relevant papers found today — try widening LOOKBACK_DAYS or search_terms.</p>"}
    </body></html>"""
    with open(out_path, "w") as f:
        f.write(html)


def mark_seen(conn, entries):
    today = datetime.date.today().isoformat()
    for _, p, _ in entries:
        conn.execute(
            "INSERT OR IGNORE INTO seen_papers (openalex_id, seen_on) VALUES (?, ?)",
            (p.get("id"), today),
        )
    conn.commit()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed = load_seed()
    conn = init_db()

    print("Building your seed profile...")
    centroid, seed_titles = build_seed_centroid(seed["seed_dois"])

    print("Fetching recent papers from OpenAlex...")
    candidates = fetch_recent_papers(seed["search_terms"], LOOKBACK_DAYS)
    print(f"Found {len(candidates)} candidates, ranking against your seed...")

    top = rank_new_papers(candidates, centroid, conn)

    print(f"Summarizing top {len(top)} matches with Gemini...")
    entries = []
    for score, p, abstract in top:
        note = summarize_relevance(seed["field_name"], seed_titles, p.get("title"), abstract)
        entries.append((score, p, note))

    out_path = os.path.join(OUTPUT_DIR, f"{datetime.date.today().isoformat()}.html")
    build_html(seed["field_name"], entries, out_path)
    mark_seen(conn, entries)

    print(f"Done. Opening {out_path}")
    webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
