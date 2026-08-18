# Hindi Government-Schemes RAG Corpus

`build_corpus.py` builds a small, source-attributed corpus from official Hindi
government sites. It respects `robots.txt`, uses a conservative rate limit, and
does not crawl domains outside its explicit allowlist.

Run a small smoke collection first:

```bash
python build_corpus.py --sources myscheme cooperation --limit-per-source 10
```

Reviewed source documents can also be kept in `seeds/` and built without web
collection. This is the preferred path for official pages that are client-rendered
or do not permit crawling:

```bash
python build_corpus.py --no-crawl --seed-documents corpus/seeds/pmjdy.jsonl
```

The output files are:

* `processed/documents.jsonl` — one cleaned source page per line.
* `processed/chunks.jsonl` — overlapping retrieval chunks with document and
  source metadata.

Inspect and deduplicate the result before embedding. Use `chunks.jsonl` for the
retrieval index; keep `source_url`, `publisher`, and `retrieved_at` so answers
can cite their sources and the corpus can be refreshed.

The crawler retains attribution but does not establish that every page can be
redistributed. Before publishing a corpus, verify the licence/terms for each
source; prefer records explicitly released under the Government Open Data
License–India when redistribution is needed.
