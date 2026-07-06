#!/bin/bash
# finish_npu_fix.sh
# Restores the kernel headers/build tree that the Ubuntu 24.04.4 point-release
# removed, then rebuilds and reloads the patched aipu (NPU) DKMS module.
# The source patch (install_npu_fix_v4.sh) is already applied at this point.
set -e

K="$(uname -r)"                                   # 6.6.89-cix
HDR_DIR="/usr/src/linux-headers-${K}"
HDR_DEB="/opt/linux-headers-next-cix_1.0.2_arm64.deb"
MNT_HDR="/mnt/tmp/usr/src/linux-headers-${K}"

echo "=== finish_npu_fix: kernel $K ==="

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo $0"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Restore kernel headers / build tree
# ---------------------------------------------------------------------------
if [ -f "${HDR_DIR}/Makefile" ]; then
    echo "[1/5] Kernel headers already present at ${HDR_DIR}"
elif [ -f "$HDR_DEB" ]; then
    echo "[1/5] Installing kernel headers from ${HDR_DEB}..."
    dpkg -i "$HDR_DEB"
elif [ -f "${MNT_HDR}/Makefile" ]; then
    echo "[1/5] Copying kernel headers from ${MNT_HDR}..."
    cp -a "$MNT_HDR" "$HDR_DIR"
else
    echo "ERROR: No kernel headers source found (deb or /mnt/tmp)."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Ensure /lib/modules/$K/build points at the headers
# ---------------------------------------------------------------------------
if [ ! -e "/lib/modules/${K}/build/Makefile" ]; then
    echo "[2/5] Creating /lib/modules/${K}/build -> ${HDR_DIR}"
    rm -f "/lib/modules/${K}/build"
    ln -s "$HDR_DIR" "/lib/modules/${K}/build"
else
    echo "[2/5] /lib/modules/${K}/build already valid"
fi
ls -l "/lib/modules/${K}/build" | sed 's/^/      /'

# ---------------------------------------------------------------------------
# 3. Rebuild the DKMS module against the restored headers
# ---------------------------------------------------------------------------
echo "[3/5] Rebuilding aipu/5.11.0 via DKMS..."
dkms remove aipu/5.11.0 --all 2>/dev/null || true
dkms add /usr/src/aipu-5.11.0 2>/dev/null || true
dkms build aipu/5.11.0
dkms install aipu/5.11.0 --force

# ---------------------------------------------------------------------------
# 4. Reload the patched module
# ---------------------------------------------------------------------------
echo "[4/5] Reloading aipu module..."
rmmod aipu 2>/dev/null || true
modprobe aipu
echo "on" > /sys/devices/platform/CIXH4000:00/power/control 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
echo "[5/5] Done."
echo "      dkms status:"; dkms status | grep -i aipu | sed 's/^/        /'
echo "      loaded module:"; lsmod | grep -i aipu | sed 's/^/        /'
echo ""
echo "=== finish_npu_fix complete ==="
