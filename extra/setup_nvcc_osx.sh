#!/bin/sh
install_loc="$HOME/.local/bin"
docker build --platform=linux/amd64 -t cuda-nvcc:12.8 - <<'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-nvdisasm-12-8 cuda-cuobjdump-12-8 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:$PATH
EOF

mkdir -p "$install_loc"
tee "$install_loc/nvccshim" >/dev/null <<'EOF'
#!/bin/sh
set -eu
# assume the final arg is the input path
# mount it so that container can read it
dir=$(dirname "${@: -1}")
exec docker run --rm --platform=linux/amd64 -v "$dir":"$dir" cuda-nvcc:12.8 "$(basename "$0")" "$@"
EOF
chmod +x "$install_loc/nvccshim"
for t in nvcc nvdisasm; do
  ln -sf "$install_loc/nvccshim" "$install_loc/$t"
done
