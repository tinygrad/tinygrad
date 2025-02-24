#!/usr/bin/env bash
npm init -y && \
npm install --save-dev webpack webpack-cli && \
npm install tiktoken && \
# insert step here to add text to package.json
jq '.scripts.build = "webpack"' package.json > package.tmp.json && \
mv package.tmp.json package.json && \
npm run build && \
mv dist/*.wasm ./tiktoken_bg.wasm
mv dist/* ./
rm -rf dist
