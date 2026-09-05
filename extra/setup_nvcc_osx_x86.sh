#!/bin/sh
# x86_64 variant of setup_nvcc_osx.sh. The upstream script pins --platform=linux/arm64 and the CUDA
# "sbsa" (ARM server) repository, which only fits Apple Silicon. On an Intel Mac the container must be
# linux/amd64 with the ubuntu2204/x86_64 CUDA repo, which then runs natively instead of under emulation.
# Note: NVCCCompiler compiles through tempfile.NamedTemporaryFile, so TMPDIR must point somewhere the
# container can see -- colima mounts $HOME but not /var/folders. Use TMPDIR=$HOME/.tinygrad-tmp.
set -eu
install_loc="$HOME/.local/bin"
docker build --platform=linux/amd64 -t cuda-nvcc-x86:12.8 - <<'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-nvdisasm-12-8 cuda-cuobjdump-12-8 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:$PATH
EOF

mkdir -p "$install_loc" "$HOME/.tinygrad-tmp"
tee "$install_loc/nvccshim" >/dev/null <<'EOF'
#!/bin/sh
set -eu
cname="cuda-nvcc-persistent-x86"
if ! docker inspect --format='{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; then
  docker rm -f "$cname" 2>/dev/null || true
  docker run -d --platform=linux/amd64 --name "$cname" -v "$HOME":"$HOME" \
    cuda-nvcc-x86:12.8 sleep infinity >/dev/null
fi
exec docker exec "$cname" "$(basename "$0")" "$@"
EOF
chmod +x "$install_loc/nvccshim"
for t in nvcc nvdisasm cuobjdump; do ln -sf "$install_loc/nvccshim" "$install_loc/$t"; done
echo "installed: $install_loc/{nvcc,nvdisasm,cuobjdump}"
