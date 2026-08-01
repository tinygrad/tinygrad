#!/usr/bin/env python3
# Sticky PR comment via the REST API: find an existing comment containing MARKER and PATCH it, or POST a new one.
# Works on GitHub and Gitea (stdlib only, replaces marocchino/sticky-pull-request-comment which needs GraphQL).
# Env vars: GITHUB_TOKEN, GITHUB_API_URL, GITHUB_REPOSITORY (set by the runner), PR_NUMBER, MARKER, and BODY_FILE or MESSAGE.
import json, os, sys, urllib.request

api, repo = os.environ["GITHUB_API_URL"], os.environ["GITHUB_REPOSITORY"]
pr, marker = os.environ["PR_NUMBER"], os.environ["MARKER"]
body = open(os.environ["BODY_FILE"]).read() if os.environ.get("BODY_FILE") else os.environ["MESSAGE"]

if not body.strip():
  print("comment body is empty, not posting")
  sys.exit(0)

def req(url, method="GET", payload=None):
  r = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), method=method,
    headers={"Authorization": f"token {os.environ['GITHUB_TOKEN']}", "Accept": "application/json", "Content-Type": "application/json"})
  return json.load(urllib.request.urlopen(r))

# find the latest sticky comment (paginate, 100 comments per page)
existing, page = None, 1
while True:
  comments = req(f"{api}/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}")
  stickies = [c for c in comments if marker in (c.get("body") or "")]
  if stickies: existing = stickies[-1]
  if not comments or len(comments) < 100: break
  page += 1

if existing is not None and existing["body"] == body:
  print("comment is already up to date")
  sys.exit(0)
url = f"{api}/repos/{repo}/issues/comments/{existing['id']}" if existing is not None else f"{api}/repos/{repo}/issues/{pr}/comments"
resp = req(url, 'PATCH' if existing is not None else 'POST', {'body': body})
print(f"{'updated' if existing is not None else 'created'} comment {resp['id']}")
