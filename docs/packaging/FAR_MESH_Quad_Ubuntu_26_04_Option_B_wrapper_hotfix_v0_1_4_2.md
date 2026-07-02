# FAR MESH Quad v0.1.4.2 Ubuntu wrapper hotfix

This hotfix corrects the Ubuntu native launcher wrapper behavior.

## Problem

The first Ubuntu native Option B wrapper launched Python with argv[0] set to:

    far-mesh-quad-native

This allowed Python multiprocessing / background task routing to treat the GUI wrapper as the Python executable. When a FAR MESH task started a worker process, the wrapper could relaunch the full GUI instead of starting a worker, creating repeated new FAR MESH windows.

## Fix

The Ubuntu wrapper now executes:

    /usr/bin/python3 -m far_mesh.main

with argv[0] intentionally set to:

    /usr/bin/python3

This preserves Python executable identity for multiprocessing and background task safety.

## Expected Ubuntu process identity

After launching FAR MESH Quad on Ubuntu:

    ps aux | grep -E "far_mesh.main|far-mesh-quad-native" | grep -v grep

The main process should appear as:

    /usr/bin/python3 -m far_mesh.main

This is intentional on Ubuntu.

## Validation

The fixed wrapper was tested on Ubuntu 26.04. FAR MESH Quad opened normally, and task execution no longer created a loop of new GUI instances.
