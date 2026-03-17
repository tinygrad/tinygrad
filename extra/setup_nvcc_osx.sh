#!/bin/sh
install_loc="$HOME/.local/bin"
docker build --platform=linux/arm64 -t cuda-nvcc:12.8 - <<'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-nvdisasm-12-8 cuda-cuobjdump-12-8 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:$PATH
EOF

mkdir -p "$install_loc"
tee "$install_loc/nvccshim" >/dev/null <<'EOF'
#!/bin/sh
set -eu
vols=""
for arg in "$@"; do
  case "$arg" in /*)
    d=$(dirname -- "$arg")
    case "$vols" in *" $d:"*) ;; *) vols="$vols -v $d:$d" ;; esac
  ;; esac
done
exec docker run --rm --platform=linux/arm64 $vols cuda-nvcc:12.8 "$(basename "$0")" "$@"
EOF
chmod +x "$install_loc/nvccshim"
for t in nvcc nvdisasm; do
  ln -sf "$install_loc/nvccshim" "$install_loc/$t"
done
