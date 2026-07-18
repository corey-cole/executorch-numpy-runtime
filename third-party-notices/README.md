# Third-Party Notices

These license/notice files come from the prebuilt ExecuTorch runtime tarball
and cover the third-party libraries statically linked into `_core`.

## Update procedure

When the pinned ExecuTorch runtime version changes (bump in
`cmake/RuntimePin.cmake`), replace these files from the new tarball:

```bash
rm -rf third-party-notices/
cp -r third_party/executorch-runtime-*-<platform>/THIRD-PARTY-NOTICES third-party-notices/
```

Commit the changes alongside the `RuntimePin.cmake` bump.
