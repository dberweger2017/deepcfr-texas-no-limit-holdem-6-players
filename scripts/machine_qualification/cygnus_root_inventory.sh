#!/bin/bash
# Owner-run inventory and optional memory-lock launcher. No device/service changes.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $EUID -eq 0 ]] || { echo 'Run as root using sudo bash SCRIPT'; exit 1; }
[[ $(hostname) == cygnus-x1 ]] || { echo 'Wrong host'; exit 1; }
if [[ ${1:-inventory} == session ]]; then
  root=/home/dberweger/cygnus-qualification-20261014
  cd "$root"
  # Resource and render access apply only to this isolated unprivileged
  # process tree. No persistent groups, limits, device ACLs or drivers change.
  ulimit -l unlimited
  groups=$(id -G dberweger | tr ' ' ',')
  render=$(getent group render | cut -d: -f3)
  [[ -n $render ]] || { echo 'Missing render group'; exit 1; }
  setpriv --reuid=dberweger --regid=dberweger --groups "$groups,$render" \
    env HOME=/home/dberweger USER=dberweger LOGNAME=dberweger \
    PATH=/home/dberweger/.cargo/bin:/home/dberweger/.local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    QUALIFICATION_DEADLINE_UTC=2026-10-15T04:50:00Z \
    /usr/bin/tmux -L cygnus-qualification new-session -d -s keepalive \
    "cd '$root' && python3 idle_until.py"
  echo 'Isolated user qualification session started; cutoff 2026-10-15 04:50 UTC.'
  exit 0
fi
if [[ ${1:-inventory} == memory ]]; then
  root=/home/dberweger/cygnus-qualification-20261014
  [[ ! -e $root/memory-locked ]] || { echo 'Preserving prior memory attempt'; exit 1; }
  cd "$root"
  # Only this process tree receives a larger lock limit. Drop root before any
  # testing; neither service settings nor persistent user limits are changed.
  ulimit -l unlimited
  export QUALIFICATION_ROOT="$root" QUALIFICATION_MEMORY_LABEL=memory-locked-instances
  setsid -f setpriv --reuid=dberweger --regid=dberweger --init-groups \
    python3 watchdog.py --output memory-locked --seconds 10800 -- python3 memory_pass.py \
    > "$root/memory-locked-supervisor.log" 2>&1 < /dev/null
  chown dberweger:dberweger "$root/memory-locked-supervisor.log"
  echo 'Detached memory test launched as dberweger; watchdog logs every 2 seconds.'
  exit 0
fi
[[ ${1:-inventory} == inventory ]] || { echo 'Usage: SCRIPT [inventory|memory|session]'; exit 1; }
out=/home/dberweger/cygnus-qualification-20261014/root-inventory
[[ ! -e $out ]] || { echo "$out exists; preserving earlier evidence"; exit 1; }
install -d -m 0755 "$out"
exec > >(tee "$out/inventory.log") 2>&1
printf 'UTC: '; date -u --iso-8601=seconds
printf '\n=== DMI system, board, BIOS, memory ===\n'
dmidecode --type 0,1,2,16,17
printf '\n=== CPU, block devices, PCI and sensors ===\n'
lscpu
lsblk -e7 -O
lspci -nnvvk
sensors
printf '\n=== SMART on every detected physical disk (read only) ===\n'
failures=0
while read -r device kind; do
  [[ $kind == disk ]] || continue
  printf '\n--- %s ---\n' "$device"
  set +e
  smartctl -x "$device"
  rc=$?
  set -e
  printf 'smartctl_exit=%s device=%s\n' "$rc" "$device"
  # Bits 0-2 indicate command/device failure; all health bits are preserved too.
  (( rc == 0 )) || failures=$((failures + 1))
  if [[ $device == /dev/nvme*n* ]]; then
    nvme smart-log "$device" || failures=$((failures + 1))
    nvme id-ctrl "$device" || failures=$((failures + 1))
  fi
done < <(lsblk -dnpo NAME,TYPE)
printf '\n=== Power limits, thermal counters and fan inputs (no writes) ===\n'
for path in /sys/class/powercap/intel-rapl*/{name,enabled,constraint_*} \
  /sys/devices/system/cpu/cpu*/thermal_throttle/* \
  /sys/class/hwmon/hwmon*/{name,fan*_input,fan*_label,pwm*,temp*_label,temp*_input}; do
  [[ -f $path ]] || continue
  printf '%s: ' "$path"
  cat "$path" || true
done
printf '\n=== Kernel storage / hardware-error messages this boot ===\n'
journalctl -k -b --no-pager | grep -Ei 'ata[0-9]|nvme|scsi|sd[a-z]|hardware error|mce|edac|thermal|throttl|nouveau' || true
printf '\n=== EFI Secure Boot variable ===\n'
for path in /sys/firmware/efi/efivars/SecureBoot-*; do
  [[ -f $path ]] && od -An -t u1 "$path"
done
printf '\nNONZERO_SMART_OR_NVME_RESULTS=%s\n' "$failures"
# Only newly created evidence is assigned to the owner; no device permissions change.
chown -R dberweger:dberweger "$out"
printf '\nInventory complete: %s/inventory.log\n' "$out"
(( failures == 0 ))
