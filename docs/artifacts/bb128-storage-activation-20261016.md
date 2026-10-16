# bb128 storage activation — October 16, 2026

Observed at `2026-10-16T07:30:32.384810+00:00` over `ssh bb128`.

- `rclone v1.60.1-DEV`.
- Required `rclone lsd gdrive:` exited **0**, listing 107 folders,
  including 94 per-PR folders. Authentication/root access is active.
- `rclone lsjson gdrive: --dirs-only` confirms these known existing folder IDs:

| Folder | Drive ID |
|---|---|
| `PR-163-equity-buckets` | `1gwungsUq-b9uey8InGy0yOD2GbaMRhp-` |
| `PR-292-nplayer-blueprint-phase2` | `1O7yc5RzlIRGD0iWn8nSXoyElcBWdriI9` |
| `PR-301-hu200-capacity-cygnus` | `159JoTfd8HUyblui5QgSMc6w1JNz-z7px` |

Mount and persistent configuration (read-only observations):

```text
TARGET       SOURCE    FSTYPE OPTIONS
/mnt/data2tb /dev/sda1 ext4   rw,relatime
TARGET       SOURCE                                    FSTYPE OPTIONS
/mnt/data2tb UUID=b37333ea-7bc8-435f-9179-7a868c0e5a8b ext4   defaults,nofail,x-systemd.device-timeout=10s
Filesystem         1B-blocks     Used     Available Use% Mounted on
/dev/sda1      1967845998592 52137984 1967777083392   1% /mnt/data2tb
```

The owner supplied the Seagate designation, approximate 150–200 MB/s throughput,
shared 3–4 MB/s uplink and remote root configuration; this verification performed
no disk benchmark or credential/config read. Activation does not itself accept
any research archive. No upload, archive relocation, cleanup, scientific work,
credential or sharing change was performed by this documentation task.

A read-only check of the owner's existing smoke-test folder also exited **0**
at 07:32:30 UTC (host log uses CEST). No second test upload was created:

```text
$ rclone check --one-way /mnt/data2tb/research-cloud/_rclone-tests gdrive:_rclone-tests
2026/10/16 09:32:30 NOTICE: Google drive root '_rclone-tests': 0 differences found
2026/10/16 09:32:30 NOTICE: Google drive root '_rclone-tests': 1 matching files
```

The local test file was 50,000,000 bytes. This validates the existing smoke test,
not acceptance of any campaign's archive; each archive still needs its own receipt.
