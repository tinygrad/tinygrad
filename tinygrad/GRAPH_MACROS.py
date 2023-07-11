import os, functools

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

GRAPH, PRUNEGRAPH, GRAPHPATH = getenv("GRAPH", 0), getenv("PRUNEGRAPH", 0), getenv("GRAPHPATH", "/tmp/net")
