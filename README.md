# Astrophotography Stacker

Stack multiple astrophotography frames into a single noise-reduced image.

## Features
- Star detection and alignment using **astroalign**
- Sigma-clipping stack for outlier rejection
- Histogram stretch for visibility
- Supports: PNG, JPG, TIFF, DNG, CR2, NEF, ARW

## How to use

### Option 1: GitHub Actions (Recommended)
1. Upload your files somewhere (Google Drive, Dropbox, etc.)
2. Get shareable URLs for each file
3. Go to **Actions** > **Stack Astrophotography** > **Run workflow**
4. Paste your file URLs (one per line)
5. Wait for the job to finish, download from **Artifacts**

### Option 2: Local
```bash
pip install numpy Pillow scipy astroalign rawpy
python3 stack.py /path/to/your/astro/files /path/to/output
```

## Output
- `stacked_raw.png` — Raw stacked result
- `stacked_stretched.png` — With histogram stretch
- `stacked_enhanced.png` — With contrast/saturation boost
