"""
Shared Lucene search service for WebShop env workers (memory-saving infra patch, no behavioural change).

Each WebAgentTextEnv normally starts its own pyserini LuceneSearcher, i.e. one JVM per env worker
(~480 MB anon RSS regardless of heap size). With 256 env workers that alone is ~120 GB, which does
not fit a 200 GB cgroup next to FSDP model loading. When WEBSHOP_SHARED_SEARCH=1, init_search_engine()
returns a SharedLuceneSearcher proxy that forwards queries to a small pool of named Ray actors, each
holding one real LuceneSearcher over the SAME index. Results (ranking, docids, raw docs) are identical.
"""
import os
from collections import namedtuple

import ray

_Hit = namedtuple("Hit", ["docid", "score"])


class _Doc:
    def __init__(self, raw):
        self._raw = raw

    def raw(self):
        return self._raw


@ray.remote(num_cpus=0.5)
class LuceneSearchService:
    def __init__(self, index_path):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(index_path)

    def search(self, query, k):
        hits = self.searcher.search(query, k=k)
        return [(h.docid, h.score, self.searcher.doc(h.docid).raw()) for h in hits]


class SharedLuceneSearcher:
    def __init__(self, index_path):
        index_path = os.path.normpath(index_path)
        n = int(os.environ.get("WEBSHOP_SHARED_SEARCH_N", "4"))
        shard = os.getpid() % n
        name = f"webshop_lucene_{os.path.basename(index_path)}_{shard}"
        self.actor = LuceneSearchService.options(
            name=name, get_if_exists=True, lifetime="detached"
        ).remote(index_path)
        self._docs = {}

    def search(self, query, k=10):
        res = ray.get(self.actor.search.remote(query, k))
        self._docs = {docid: raw for docid, _, raw in res}
        return [_Hit(docid, score) for docid, score, _ in res]

    def doc(self, docid):
        return _Doc(self._docs[docid])
