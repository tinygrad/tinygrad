# fswatch -0or tinygrad | xargs -0 -n 1 -I {} -t extra/pyodide/package.sh
echo "Current dir: $PWD"
source ~/gbin/ml/bin/activate
pip wheel . 