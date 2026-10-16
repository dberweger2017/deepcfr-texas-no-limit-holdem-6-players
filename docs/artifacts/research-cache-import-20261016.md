# Research cache tool import — October 16, 2026

Imported over SSH from bb128, without reading or copying rclone credentials:

| Deployed source | SHA256 at import |
|---|---|
| `/mnt/data2tb/research-cloud/.tools/research_cache.py` | `d03af1d5b0d3161723d44cbadf57cee62f5df60baaf3bd7b1fc32e340d49dcce` |
| `/mnt/data2tb/research-cloud/README.md` | `21363a9bd97e425343b19d9c72e2e4cf3a444be313173ecbb53783c84934a049` |

The initial import commit contains a byte-identical `scripts/research_cache.py`
and `docs/research-cache-deployed-readme.md`. Later repository fixes are separate
commits; this source hash remains the deployed baseline, not a claim that a
patched repository copy is already deployed.

Read-only activation observations: `/research` is the owner's root-owned symlink
to `/mnt/data2tb/research-cloud`; the deployed tool is executable. The owner's
user crontab contains:

```text
0 4 * * * /usr/bin/python3 /mnt/data2tb/research-cloud/.tools/research_cache.py evict >> /mnt/data2tb/research-cloud/.tools/cron.log 2>&1
```

No deployment, cron/configuration/credential change, live upload or eviction is
part of the import. Eviction changes in this PR require owner review before
any deployment. Existing PR301/306 and other running campaigns keep their
current archival method until they finish. The private M4 cable is not relied on.
