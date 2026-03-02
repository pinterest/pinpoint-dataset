#!/usr/bin/env python3
"""Generate an interactive HTML visualization of retrieval results."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from utils.data_utils import normalize_query_id


def sig_to_url(sig: str) -> str:
    return f"https://i.pinimg.com/736x/{sig[0:2]}/{sig[2:4]}/{sig[4:6]}/{sig}.jpg"


def load_results(file_path):
    """Load a canonical results JSON file."""
    with open(file_path) as f:
        return json.load(f)


def build_query_data(results_path, gt_path):
    results = load_results(results_path)
    gt_df = pd.read_parquet(gt_path)

    queries = []
    for _, row in gt_df.iterrows():
        qid = normalize_query_id(row["query_id"])
        if qid not in results:
            continue
        retrieved = results[qid].get("retrieved_items", [])
        if not retrieved:
            continue

        positives = row["positive_candidates"]
        if hasattr(positives, "tolist"):
            positives = positives.tolist()
        elif not isinstance(positives, list):
            positives = []
        pos_set = set(positives)

        negatives = row.get("negative_candidates", [])
        if hasattr(negatives, "tolist"):
            negatives = negatives.tolist()
        elif not isinstance(negatives, list):
            negatives = []
        neg_set = set(negatives)

        top10 = retrieved[:10]
        top50 = retrieved[:50]
        hits1 = 1 if (top10[:1] and top10[0] in pos_set) else 0
        hits10 = sum(1 for r in top10 if r in pos_set)
        hits50 = sum(1 for r in top50 if r in pos_set)
        neg10 = sum(1 for r in top10 if r in neg_set)

        first_hit = None
        for rank, item in enumerate(retrieved, 1):
            if item in pos_set:
                first_hit = rank
                break

        retrieved_annotated = []
        for sig in top10:
            if sig in pos_set:
                label = "positive"
            elif sig in neg_set:
                label = "negative"
            else:
                label = "neutral"
            retrieved_annotated.append({"sig": sig, "label": label})

        gt_sample = positives[:6]

        queries.append({
            "qid": qid,
            "instruction": row.get("instruction", ""),
            "query_sig": row.get("query_image_signature", ""),
            "query_sig2": row.get("query_image_signature2", None),
            "category": row.get("query_category", ""),
            "interest": row.get("l1_interest", ""),
            "n_pos": len(positives),
            "n_neg": len(negatives),
            "hits1": hits1,
            "hits10": hits10,
            "hits50": hits50,
            "neg10": neg10,
            "first_hit": first_hit,
            "retrieved": retrieved_annotated,
            "gt_sample": gt_sample,
        })

    return queries


def compute_stats(queries):
    total = len(queries)
    hit1 = sum(1 for q in queries if q["hits1"] > 0)
    hit10 = sum(1 for q in queries if q["hits10"] > 0)
    hit50 = sum(1 for q in queries if q["hits50"] > 0)
    zero = sum(1 for q in queries if q["hits50"] == 0)
    neg_contaminated = sum(1 for q in queries if q["neg10"] > 0)
    mean_p10 = np.mean([q["hits10"] / 10 for q in queries])
    mean_r50 = np.mean([q["hits50"] / q["n_pos"] for q in queries if q["n_pos"] > 0])
    first_hits = [q["first_hit"] for q in queries if q["first_hit"] is not None]
    median_first = np.median(first_hits) if first_hits else None

    cats = Counter(q["category"] for q in queries)

    return {
        "total": total,
        "hit1": hit1, "hit1_pct": hit1 / total * 100,
        "hit10": hit10, "hit10_pct": hit10 / total * 100,
        "hit50": hit50, "hit50_pct": hit50 / total * 100,
        "zero": zero, "zero_pct": zero / total * 100,
        "neg_contaminated": neg_contaminated,
        "neg_pct": neg_contaminated / total * 100,
        "mean_p10": mean_p10,
        "mean_r50": mean_r50,
        "median_first_hit": median_first,
        "categories": dict(cats.most_common()),
    }


def generate_html(queries, stats, output_path):
    categories = sorted(stats["categories"].keys())
    cat_options = "".join(
        f'<option value="{c}">{c} ({stats["categories"][c]})</option>'
        for c in categories
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Retrieval Results — MagicPins PE</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f6fa; color: #2d3436; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 6px; font-size: 1.6em; }}
.subtitle {{ text-align: center; color: #636e72; margin-bottom: 24px; font-size: 0.95em; }}

.stats-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px; max-width: 1200px; margin: 0 auto 28px;
}}
.stat-card {{
  background: #fff; border-radius: 10px; padding: 16px 14px;
  text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.08);
}}
.stat-card .val {{ font-size: 1.8em; font-weight: 700; }}
.stat-card .lbl {{ font-size: .82em; color: #636e72; margin-top: 2px; }}
.stat-card.good .val {{ color: #00b894; }}
.stat-card.bad .val {{ color: #d63031; }}
.stat-card.warn .val {{ color: #fdcb6e; }}
.stat-card.info .val {{ color: #0984e3; }}

.controls {{
  max-width: 1200px; margin: 0 auto 18px;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
}}
.controls select, .controls input {{
  padding: 8px 12px; border: 1px solid #dfe6e9; border-radius: 6px;
  font-size: .9em; background: #fff;
}}
.controls select {{ min-width: 160px; }}
.controls input {{ min-width: 200px; }}
.controls .count {{ margin-left: auto; font-size: .9em; color: #636e72; }}

.query-card {{
  max-width: 1200px; margin: 0 auto 20px; background: #fff;
  border-radius: 12px; box-shadow: 0 1px 6px rgba(0,0,0,.07);
  overflow: hidden;
}}
.card-header {{
  display: flex; align-items: center; gap: 14px;
  padding: 14px 18px; border-bottom: 1px solid #f0f0f0;
  flex-wrap: wrap;
}}
.card-header img {{
  width: 80px; height: 80px; object-fit: cover; border-radius: 8px;
  border: 2px solid #dfe6e9; flex-shrink: 0;
}}
.card-meta {{ flex: 1; min-width: 200px; }}
.card-meta .qid {{ font-size: .78em; color: #b2bec3; }}
.card-meta .instruction {{ font-size: 1.05em; font-weight: 600; margin: 4px 0; }}
.card-meta .tags {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.tag {{
  font-size: .72em; padding: 2px 8px; border-radius: 20px;
  background: #dfe6e9; color: #636e72;
}}
.tag.hit {{ background: #00b89433; color: #00b894; }}
.tag.miss {{ background: #d6303133; color: #d63031; }}
.tag.neg {{ background: #fdcb6e55; color: #e17055; }}

.card-badges {{
  display: flex; gap: 8px; flex-shrink: 0;
}}
.badge {{
  padding: 6px 12px; border-radius: 8px; font-size: .82em;
  font-weight: 600; text-align: center; line-height: 1.2;
}}
.badge.green {{ background: #00b89422; color: #00b894; }}
.badge.red {{ background: #d6303122; color: #d63031; }}
.badge.orange {{ background: #e1705522; color: #e17055; }}

.card-body {{ padding: 14px 18px; }}
.section-label {{
  font-size: .78em; font-weight: 600; color: #b2bec3;
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px;
}}
.img-row {{
  display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px;
}}
.img-cell {{
  flex-shrink: 0; width: 90px; text-align: center;
}}
.img-cell img {{
  width: 86px; height: 86px; object-fit: cover; border-radius: 6px;
  border: 3px solid #dfe6e9;
}}
.img-cell.positive img {{ border-color: #00b894; }}
.img-cell.negative img {{ border-color: #d63031; }}
.img-cell.neutral img {{ border-color: #dfe6e9; }}
.img-cell .rank {{
  font-size: .7em; color: #636e72; margin-top: 2px;
}}
.img-cell .rank.hit {{ color: #00b894; font-weight: 700; }}
.img-cell .rank.neg-hit {{ color: #d63031; font-weight: 700; }}

.gt-section {{ margin-top: 12px; }}
.gt-section .img-cell img {{ border-color: #0984e3; opacity: .85; }}

.legend {{
  max-width: 1200px; margin: 0 auto 14px;
  display: flex; gap: 16px; font-size: .82em; color: #636e72;
  flex-wrap: wrap;
}}
.legend span {{ display: flex; align-items: center; gap: 4px; }}
.legend .dot {{
  width: 14px; height: 14px; border-radius: 3px; display: inline-block;
}}

.pagination {{
  max-width: 1200px; margin: 10px auto 30px;
  display: flex; justify-content: center; gap: 8px; align-items: center;
}}
.pagination button {{
  padding: 8px 16px; border: 1px solid #dfe6e9; border-radius: 6px;
  background: #fff; cursor: pointer; font-size: .9em;
}}
.pagination button:hover {{ background: #f0f0f0; }}
.pagination button:disabled {{ opacity: .4; cursor: default; }}
.pagination .page-info {{ font-size: .9em; color: #636e72; }}

.hidden {{ display: none !important; }}
</style>
</head>
<body>

<h1>MagicPins PE — Retrieval Analysis</h1>
<p class="subtitle">{stats['total']:,} queries evaluated &middot; precision@10 = {stats['mean_p10']:.4f} &middot; recall@50 = {stats['mean_r50']:.4f}</p>

<div class="stats-grid">
  <div class="stat-card good">
    <div class="val">{stats['hit1']}</div>
    <div class="lbl">Hit @1 ({stats['hit1_pct']:.1f}%)</div>
  </div>
  <div class="stat-card good">
    <div class="val">{stats['hit10']}</div>
    <div class="lbl">Hit @10 ({stats['hit10_pct']:.1f}%)</div>
  </div>
  <div class="stat-card good">
    <div class="val">{stats['hit50']}</div>
    <div class="lbl">Hit @50 ({stats['hit50_pct']:.1f}%)</div>
  </div>
  <div class="stat-card bad">
    <div class="val">{stats['zero']}</div>
    <div class="lbl">Zero hits ({stats['zero_pct']:.1f}%)</div>
  </div>
  <div class="stat-card warn">
    <div class="val">{stats['neg_contaminated']}</div>
    <div class="lbl">Neg in top-10 ({stats['neg_pct']:.1f}%)</div>
  </div>
  <div class="stat-card info">
    <div class="val">{stats['median_first_hit']:.0f}</div>
    <div class="lbl">Median 1st-hit rank</div>
  </div>
</div>

<div class="legend">
  <span><span class="dot" style="background:#00b894"></span> Positive (correct)</span>
  <span><span class="dot" style="background:#d63031"></span> Negative (wrong)</span>
  <span><span class="dot" style="background:#dfe6e9"></span> Neutral (unlabeled)</span>
  <span><span class="dot" style="background:#0984e3"></span> Ground truth sample</span>
</div>

<div class="controls">
  <select id="filterResult">
    <option value="all">All results</option>
    <option value="hit1">Hit @1</option>
    <option value="hit10">Hit @10</option>
    <option value="hit50">Hit @50 only (miss@10)</option>
    <option value="zero">Zero hits</option>
    <option value="has_neg">Has negatives in top-10</option>
  </select>
  <select id="filterCat">
    <option value="all">All categories</option>
    {cat_options}
  </select>
  <input type="text" id="searchBox" placeholder="Search instructions...">
  <select id="sortBy">
    <option value="qid">Sort: Query ID</option>
    <option value="hits10_desc">Sort: Hits@10 desc</option>
    <option value="hits10_asc">Sort: Hits@10 asc</option>
    <option value="neg_desc">Sort: Neg@10 desc</option>
    <option value="first_hit">Sort: First hit rank</option>
  </select>
  <span class="count" id="countLabel"></span>
</div>

<div id="cardContainer"></div>

<div class="pagination">
  <button id="prevBtn" onclick="changePage(-1)">&#8592; Prev</button>
  <span class="page-info" id="pageInfo"></span>
  <button id="nextBtn" onclick="changePage(1)">Next &#8594;</button>
</div>

<script>
const DATA = {json.dumps(queries, separators=(',', ':'))};
const PER_PAGE = 30;
let filtered = [...DATA];
let page = 0;

function sigUrl(sig) {{
  return `https://i.pinimg.com/736x/${{sig.slice(0,2)}}/${{sig.slice(2,4)}}/${{sig.slice(4,6)}}/${{sig}}.jpg`;
}}

function applyFilters() {{
  const rf = document.getElementById('filterResult').value;
  const cf = document.getElementById('filterCat').value;
  const search = document.getElementById('searchBox').value.toLowerCase();
  const sort = document.getElementById('sortBy').value;

  filtered = DATA.filter(q => {{
    if (rf === 'hit1' && q.hits1 === 0) return false;
    if (rf === 'hit10' && q.hits10 === 0) return false;
    if (rf === 'hit50' && (q.hits50 === 0 || q.hits10 > 0)) return false;
    if (rf === 'zero' && q.hits50 > 0) return false;
    if (rf === 'has_neg' && q.neg10 === 0) return false;
    if (cf !== 'all' && q.category !== cf) return false;
    if (search && !q.instruction.toLowerCase().includes(search)) return false;
    return true;
  }});

  if (sort === 'hits10_desc') filtered.sort((a,b) => b.hits10 - a.hits10);
  else if (sort === 'hits10_asc') filtered.sort((a,b) => a.hits10 - b.hits10);
  else if (sort === 'neg_desc') filtered.sort((a,b) => b.neg10 - a.neg10);
  else if (sort === 'first_hit') filtered.sort((a,b) => (a.first_hit||9999) - (b.first_hit||9999));
  else filtered.sort((a,b) => a.qid.localeCompare(b.qid));

  page = 0;
  render();
}}

function changePage(d) {{
  page += d;
  render();
}}

function render() {{
  const total = filtered.length;
  const maxPage = Math.max(0, Math.ceil(total / PER_PAGE) - 1);
  if (page > maxPage) page = maxPage;
  if (page < 0) page = 0;

  document.getElementById('countLabel').textContent = `${{total}} queries`;
  document.getElementById('pageInfo').textContent = `Page ${{page+1}} of ${{maxPage+1}}`;
  document.getElementById('prevBtn').disabled = page === 0;
  document.getElementById('nextBtn').disabled = page >= maxPage;

  const start = page * PER_PAGE;
  const slice = filtered.slice(start, start + PER_PAGE);
  const container = document.getElementById('cardContainer');

  container.innerHTML = slice.map(q => {{
    const queryImgs = [q.query_sig, q.query_sig2].filter(Boolean).map(s =>
      `<img src="${{sigUrl(s)}}" alt="query" onerror="this.style.display='none'">`
    ).join('');

    const hitBadge = q.hits10 > 0
      ? `<div class="badge green">${{q.hits10}} hit${{q.hits10>1?'s':''}}<br>@10</div>`
      : `<div class="badge red">0 hits<br>@10</div>`;
    const negBadge = q.neg10 > 0
      ? `<div class="badge orange">${{q.neg10}} neg<br>@10</div>`
      : '';
    const firstHitBadge = q.first_hit
      ? `<div class="badge green">1st hit<br>#${{q.first_hit}}</div>`
      : `<div class="badge red">No hit<br>in top-50</div>`;

    const retImgs = q.retrieved.map((r, i) => {{
      const cls = r.label;
      const rankCls = r.label === 'positive' ? 'hit' : (r.label === 'negative' ? 'neg-hit' : '');
      const lbl = r.label === 'positive' ? '&#10003;' : (r.label === 'negative' ? '&#10007;' : '');
      return `<div class="img-cell ${{cls}}">
        <img src="${{sigUrl(r.sig)}}" alt="r${{i+1}}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2286%22 height=%2286%22><rect fill=%22%23eee%22 width=%2286%22 height=%2286%22/><text x=%2243%22 y=%2248%22 text-anchor=%22middle%22 fill=%22%23999%22 font-size=%2211%22>N/A</text></svg>'">
        <div class="rank ${{rankCls}}">#${{i+1}} ${{lbl}}</div>
      </div>`;
    }}).join('');

    const gtImgs = q.gt_sample.map(s =>
      `<div class="img-cell">
        <img src="${{sigUrl(s)}}" alt="gt" loading="lazy" onerror="this.style.display='none'" style="border-color:#0984e3">
      </div>`
    ).join('');

    return `<div class="query-card">
      <div class="card-header">
        ${{queryImgs}}
        <div class="card-meta">
          <div class="qid">Query ${{q.qid}}</div>
          <div class="instruction">${{q.instruction}}</div>
          <div class="tags">
            <span class="tag">${{q.category}}</span>
            <span class="tag">${{q.interest}}</span>
            <span class="tag">${{q.n_pos}} positives</span>
            <span class="tag">${{q.n_neg}} negatives</span>
            ${{q.hits10 > 0 ? '<span class="tag hit">has hits</span>' : '<span class="tag miss">no hits @10</span>'}}
            ${{q.neg10 > 0 ? '<span class="tag neg">' + q.neg10 + ' neg in top10</span>' : ''}}
          </div>
        </div>
        <div class="card-badges">
          ${{hitBadge}}${{negBadge}}${{firstHitBadge}}
        </div>
      </div>
      <div class="card-body">
        <div class="section-label">Top-10 Retrieved</div>
        <div class="img-row">${{retImgs}}</div>
        <div class="gt-section">
          <div class="section-label">Ground Truth Positives (sample)</div>
          <div class="img-row">${{gtImgs}}</div>
        </div>
      </div>
    </div>`;
  }}).join('');
}}

document.getElementById('filterResult').addEventListener('change', applyFilters);
document.getElementById('filterCat').addEventListener('change', applyFilters);
document.getElementById('searchBox').addEventListener('input', applyFilters);
document.getElementById('sortBy').addEventListener('change', applyFilters);

applyFilters();
</script>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"Saved HTML to {output_path}")


if __name__ == "__main__":
    results_path = "standardized_results/bge_vl_mllm_s1_retrieval_results_licensed.json"
    gt_path = "pinpoint_licensed.parquet"

    print("Loading and analyzing per-query results...")
    queries = build_query_data(results_path, gt_path)
    print(f"Analyzed {len(queries)} queries")

    stats = compute_stats(queries)
    print(f"  Hit@1:  {stats['hit1']:,} ({stats['hit1_pct']:.1f}%)")
    print(f"  Hit@10: {stats['hit10']:,} ({stats['hit10_pct']:.1f}%)")
    print(f"  Hit@50: {stats['hit50']:,} ({stats['hit50_pct']:.1f}%)")
    print(f"  Zero:   {stats['zero']:,} ({stats['zero_pct']:.1f}%)")
    print(f"  Neg contaminated: {stats['neg_contaminated']:,} ({stats['neg_pct']:.1f}%)")

    generate_html(queries, stats, "retrieval_analysis.html")
