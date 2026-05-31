#!/usr/bin/env python3
"""Patch a freshly-converted SIAM n3d .mnn so it works on small patches.

Background: MNNConvert in this branch populates 3 trailing flatbuffer
field-offset values in the Net's vtable that MNN's runtime then trips
over during resizeSession() for any shape other than the model's
default (256x256x192). The May 29 build of MNNConvert left those 3
field offsets at the flatbuffers "absent field" marker 0xFFFFFFFF and
the resulting .mnn loads cleanly. Clearing those bytes to 0xff in any
freshly-converted .mnn restores the working behavior.

We could not reproduce the May 29 MNNConvert output despite checking
out the same git commit; the divergence appears to be in some
non-deterministic state inside MNN's converter that we have not been
able to pin down. The post-processor below is the practical fix until
the upstream non-determinism is identified.

The patch:
  - The 12 bytes at byte offset (filesize - 56) get rewritten to 0xff.
  - Pre-patch they look like 0001000000010000c0000000.
  - Post-patch they match what the May 29 working file has.
  - Bytes elsewhere are left untouched, including the bizCode +
    mnn_uuid strings.

Usage:
  python3 patch_n3d_for_small_patch.py <mnn_file> [<mnn_file> ...]

Idempotent: if the bytes are already 0xff (already patched), the
script is a no-op.
"""
import sys, os

PATCH_BYTES = bytes([0xFF] * 12)
EXPECTED_PRE = bytes.fromhex("0001000000010000c0000000")


def patch_one(path: str) -> None:
    with open(path, "rb") as f:
        data = bytearray(f.read())
    sz = len(data)
    start = sz - 56
    end = start + 12
    cur = bytes(data[start:end])
    if cur == PATCH_BYTES:
        print(f"  {path}: already patched -- skipped")
        return
    if cur != EXPECTED_PRE:
        print(
            f"  {path}: bytes at [{start},{end}) are {cur.hex()} -- not the expected "
            f"pre-patch pattern {EXPECTED_PRE.hex()}; skipped"
        )
        return
    data[start:end] = PATCH_BYTES
    with open(path, "wb") as f:
        f.write(data)
    print(
        f"  {path}: patched [{start},{end}) {EXPECTED_PRE.hex()} -> {PATCH_BYTES.hex()}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        patch_one(arg)


if __name__ == "__main__":
    main()
