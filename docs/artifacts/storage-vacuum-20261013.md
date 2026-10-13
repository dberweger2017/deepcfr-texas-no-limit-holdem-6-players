# M1/M4 storage cleanup — October 13, 2026

The owner requested another pass after the overnight research PRs merged. This
pass retired **88.890 GB of existing research copies**:
13 native cache releases and 36 unchanged archived originals.
The separately authorized personal backup offload released **19.529 GB**
on M1 after its upload completed; exact personal filenames remain in a private
local receipt and are omitted from this repository. Total allocated copies
released: **108.419 GB**. Free space at verification:
**M1 65.986 GB /M4 104.323 GB**.

| Operation | Allocated bytes released or retired |
| --- | ---: |
| M1 uploaded research archive cache | 4,932,796,416 |
| M4 uploaded research archive cache | 60,486,582,272 |
| M4 unchanged #246/#252 originals | 23,470,350,336 |
| M1 uploaded personal backup cache | 19,529,351,168 |

Filesystem free space is a separate time-stamped observation and can change
while other work runs. Retired originals are counted by their allocated blocks;
cache reclamation uses each placeholder's observed before/after block delta.

## Preservation and acceptance

- #246 and #252 were confirmed MERGED immediately before their own unchanged
  archived evidence was removed. All 36 exact original paths, sizes and source
  mtimes match their existing manifests; recorded member digests provide the
  restoration locators. Ctimes precede the local archive-readback seal. Fresh identity/open-handle checks and singly
  linked-file checks preceded unlink. Changed, uncertain or omitted files stay.
- Google Drive's current native uploaded/no-pending/no-conflicts state and
  independent file ID/name/size/parent metadata accept the retained archives.
  Existing manifests and provenance are trusted without downloading archives
  or repeating archive/member payload hash audits. Only small manifest metadata
  was read and compared with its existing recorded digest.
- Cache release used normal `FileManager.evictUbiquitousItem`, preserving cloud
  files, identities and local placeholders. Some M4 placeholders retain small
  provider bookkeeping allocations, excluded from the released-byte count.
  No synced deletion, forced offload or unpinning occurred.
- The incomplete #246 ZIP prefix remains failure evidence in Drive; only its
  uploaded local cache was released. No original retirement relies on it.
- The live M1 Slumbot analysis still reads the #248 HU200 average, so that model
  and its work root remain. Source, shared Git, tables, other retained inputs,
  installed release caches and other agents' work were preserved. No research,
  default, release, application-shutdown or unattended-work change was made.

## Restoration

[All 36 removed paths and exact archive/member/size/SHA256 locators](storage-vacuum-20261013-locators.md).
Use the retained archives in Research-Cloud or their observed Drive links.
For future research use, download to a fresh ignored nonsynced path, verify the
whole archive SHA256 and its manifest, extract only the named member, and verify
that member's bytes/SHA256. Never overwrite an active input.

```sh
shasum -a 256 ARCHIVE.zip
python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2],sys.argv[3])' ARCHIVE.zip MEMBER NEW_IGNORED_ROOT
shasum -a 256 NEW_IGNORED_ROOT/MEMBER
```

Cache-only files restore by opening/downloading their unchanged Drive file.
The two scientific source archives for retired originals are #246 full evidence
and #252 full evidence; archive and member hashes are in the locator record.

[Full private cleanup receipt](https://drive.google.com/file/d/1SDqMIa6b-aBNfGqvG7Wx9iUgUzrA01gM/view?usp=drivesdk), Drive ID `1SDqMIa6b-aBNfGqvG7Wx9iUgUzrA01gM`,
**66,745 bytes**, SHA256 `307e8062e2c182514983ca21ef25816e692b319dc178a2a7830b43d71fd5ec03`.
Independent connector readback confirms its name/size and Research-Cloud parent
`188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. The receipt contains every research cache
identity, original stat acceptance, restoration locator and preservation scope.
Earlier originals-retained statements are superseded only for the listed paths.
