# RMSD plateau Docker runner

This package runs `rmsd_plateau.py` in an isolated Python 3.11 Docker image.

## Build

```bash
chmod +x build_image.sh run_rmsd_plateau.sh
./build_image.sh
```

Default image name:

```text
rmsd-plateau:1.0
```

To use another name:

```bash
IMAGE_NAME=my-rmsd:latest ./build_image.sh
```

## Run

Place one production `.xtc` and its topology (`.tpr` preferred) in one directory:

```text
replica_1/
├── md_200ns.xtc
└── md_200ns.tpr
```

Then run:

```bash
./run_rmsd_plateau.sh /absolute/or/relative/path/to/replica_1
```

Additional analysis options are forwarded to `rmsd_plateau.py`:

```bash
./run_rmsd_plateau.sh replica_1 \
  --stride 10 \
  --bootstrap 10000 \
  --equiv-drift-A 0.1 \
  --equiv-window-ns 100
```

For C-alpha RMSD:

```bash
./run_rmsd_plateau.sh replica_1 --selection "protein and name CA"
```

The generated CSV, TXT, JSON, and PNG files are written back into the input directory.

## Important

The runner deliberately refuses ambiguous inputs. If a directory contains multiple `.xtc` files or multiple plausible `.tpr` files, split them into separate directories or rename/clean the directory first. This avoids silently analyzing the wrong trajectory/topology pair.
