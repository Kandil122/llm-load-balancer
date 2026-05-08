# NVIDIA MX350 on Arch Linux — Personal Setup Guide

## System Info

| Component | Details |
|---|---|
| Machine | Dell Laptop |
| iGPU | Intel Iris Xe Graphics (TigerLake-LP GT2) |
| dGPU | NVIDIA GeForce MX350 (GP107M) |
| Kernel | `7.0.3-arch1-2` (stock Arch `linux`) |
| OS | Arch Linux |

---

## Current Status

- **Intel Iris Xe** → active, handles all display and desktop (driver: `i915`)
- **NVIDIA MX350** → active, available on demand (driver: `nouveau`)
- **Mode:** PRIME offloading — Intel is default, NVIDIA invoked manually with `DRI_PRIME=1`

---

## Driver History — What Was Tried

### 1. `nvidia-open` (failed)
The modern open-source NVIDIA kernel module available in Arch repos (`extra/nvidia-open 595.71.05`).

**Why it failed:** `nvidia-open` only supports Turing (RTX 20xx) and newer architectures. The MX350 is based on **Pascal (GP107)** — too old. The driver loaded but could not bind to the device:

```
NVRM: The NVIDIA GPU 0000:01:00.0 (PCI ID: 10de:1c94)
NVRM: nvidia.ko because it does not include the required GPU
nvidia 0000:01:00.0: probe with driver nvidia failed with error -1
```

> Note: The legacy `nvidia` package (closed-source blob) has been **removed from Arch repos entirely** as of 2025. Only `nvidia-open` remains.

### 2. `nvidia-470xx-dkms` from AUR (failed)
The 470xx legacy branch supports Pascal GPUs including the MX350.

**Why it failed:** Caused system crashes on this machine. Abandoned.

### 3. `nouveau` (working ✓)
The open-source reverse-engineered NVIDIA driver, built into the Linux kernel.

**Extra step required:** Nouveau needs firmware blobs for 3D acceleration on Pascal. The `nouveau-fw` package (AUR) was needed. Without it, the driver loaded but 3D init failed:

```
nvc0_screen_create:805 - Base screen init failed: -19
failed to load driver: nouveau
```

After installing firmware and rebooting, the GPU initialized correctly and reports as `NV137`.

---

## Current Driver Setup

- **MX350 driver:** `nouveau`
- **Blacklist file:** `/etc/modprobe.d/blacklist-nouveau.conf` — **deleted** (was left over from a previous nvidia installation)
- **`/etc/mkinitcpio.conf` MODULES line:** cleared of all nvidia entries, set to `MODULES=()`
- **Firmware:** `nouveau-fw` installed from AUR

---

## How to Run Apps on the NVIDIA GPU

Prefix any command with `DRI_PRIME=1`:

```bash
DRI_PRIME=1 <command>
```

### Examples

```bash
DRI_PRIME=1 firefox        # Firefox on NVIDIA
DRI_PRIME=1 blender        # Blender on NVIDIA
DRI_PRIME=1 glmark2        # GPU benchmark on NVIDIA
DRI_PRIME=1 mpv video.mkv  # Video playback on NVIDIA
```

---

## Verify Which GPU is Active

```bash
# Should show NV137 (MX350 via nouveau)
DRI_PRIME=1 glxinfo | grep "OpenGL renderer"

# Should show Intel Iris Xe (default)
glxinfo | grep "OpenGL renderer"
```

---

## Useful Diagnostic Commands

```bash
# Check which driver is bound to each GPU
lspci -k | grep -A 3 -E "VGA|3D"

# Check loaded GPU modules
lsmod | grep -E "nouveau|nvidia|i915"

# Check kernel messages for GPU errors
journalctl -b | grep -iE "nvidia|nouveau" | head -30

# Monitor GPU usage (install nvtop)
sudo pacman -S nvtop
nvtop
```

---

## Notes

- Without `DRI_PRIME=1`, everything runs on Intel — better battery life for daily use
- `nouveau` does not support CUDA — if CUDA is needed, a working proprietary driver is required (not currently possible on this machine)
- The Intel iGPU handles the display output; the MX350 renders and passes frames back via PRIME (render offload)
