#!/bin/sh
install_loc="$HOME/.local/bin"
docker build --platform=linux/amd64 -t rocm-hipcc:7.2 - <<'EOF'
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates gnupg tzdata && \
  wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb && \
  apt-get install -y ./amdgpu-install_7.2.70200-1_all.deb && \
  amdgpu-install -y --usecase=rocm --no-dkms --no-32 && \
  rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/rocm/bin:$PATH
EOF

mkdir -p "$install_loc"
tee "$install_loc/hipccshim" >/dev/null <<'EOF'
#!/bin/sh
set -eu
cname="rocm-hipcc-persistent"
if ! docker inspect --format='{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; then
  docker rm -f "$cname" 2>/dev/null || true
  docker run -d --platform=linux/amd64 --name "$cname" \
    -v /var/folders:/var/folders -v "$HOME":"$HOME" \
    rocm-hipcc:7.2 sleep 300 >/dev/null
fi
exec docker exec "$cname" "$(basename "$0")" "$@"
EOF
chmod +x "$install_loc/hipccshim"
for t in hipcc hipconfig; do
  ln -sf "$install_loc/hipccshim" "$install_loc/$t"
done
