import re
import string

def levenshtein(a, b):
  n, m = len(a), len(b)
  if n > m:
    a, b, n, m = b, a, m, n

  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)

  return current[n]

def word_error_rate(x, y):
  scores = words = 0
  for h, r in zip(x, y):
    h_list = h.split()
    r_list = r.split()
    words += len(r_list)
    scores += levenshtein(h_list, r_list)
  return float(scores) / words, float(scores), words

def normalize_string(s):
  s = "".join(c for c in s.lower() if c not in string.punctuation)
  s = re.sub(r'\b(a|an|the)\b', ' ', s)
  return " ".join(s.split())

def f1_score(x, y):
  xt = normalize_string(x).split()
  yt = normalize_string(y).split()
  ct = set(xt) & set(yt)
  if (ns := len(ct)) == 0:
    return 0.0
  p = ns / len(xt)
  r = ns / len(yt)
  return 2 * p * r / (p + r)
