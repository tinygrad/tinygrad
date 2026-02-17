#!/bin/sh
install_loc="$HOME/.local/bin"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/Dockerfile" <<'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 \
        cuda-nvdisasm-12-8 \
        cuda-cuobjdump-12-8 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:$PATH
EOF
docker build --platform=linux/amd64 -t cuda-nvcc:12.8 "$tmpdir"

mkdir -p "$install_loc"
tee "$install_loc/nvccshim" >/dev/null <<'EOF'
#!/bin/sh
cmd=$(basename "$0")
if [ "$cmd" = "nvdisasm" ]; then
  docker run --rm --platform=linux/amd64 -i cuda-nvcc:12.8 sh -c 'cat>/tmp/d;nvdisasm /tmp/d' <"$1"
else
  while [ $# -gt 1 ]; do
    [ "$1" = "-o" ] && { o="$2"; shift 2; } || { f="$f $1"; shift; }
  done
  docker run --rm --platform=linux/amd64 -i cuda-nvcc:12.8 sh -c "cat>/tmp/s.cu;$cmd$f -o /tmp/o /tmp/s.cu;cat /tmp/o" <"$1" >"$o"
fi
EOF
chmod +x "$install_loc/nvccshim"
for t in nvcc nvdisasm ptxas cuobjdump nvlink; do
  ln -sf "$install_loc/nvccshim" "$install_loc/$t"
done
