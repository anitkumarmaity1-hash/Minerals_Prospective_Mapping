Minerals_Prospective_Mapping
<br><br>
Mineral Region Segmentation Using Multispectral Remote Sensing Images Based on CNN.
<br><br>
Anit Kumar Maity · Reg. No. 24MCA00PY0055 · Dept. of Computer Science, Pondicherry University · April 2026
<br><br>
Python 3.10 PyTorch 2.1 Landsat-8 + ASTER Binary Segmentation Google Colab T4 GPU MCA Project 2026
<br><br>
This project presents a deep learning framework for binary mineral prospectivity mapping in the Salem District, Tamil Nadu, which hosts magnetite and hematite occurrences. Three encoder-decoder architectures (U-Net, Attention U-Net, ResUNet) are trained on multi-source Landsat-8 inputs (7 bands, PCA, geological rasters). A separate ASTER 14-band variant employs a Spectral-Spatial Transformer (SST) — a ViT-based HSI model — to capture long-range spectral and spatial correlations under extreme class imbalance (approx 2,509: 1).
<br><br>
01 · PROJECT OVERVIEW
Study Area — Salem District, Tamil Nadu — five sub-districts (Attur, Mettur, Omalur, Salem, Sankari) covering ~120×100 km at 30 m resolution. Documented magnetite, hematite and alteration mineral occurrences.
Satellite Data — Landsat-8 L2 surface reflectance (7 bands, 30 m, 13 Feb 2023, 99.6% cloud-free). ASTER 14-band stack (B1–B9 reflectance + B10–B14 brightness temperature) for the SST variant.
Ground Truth — Mineralisation shapefile → 90 m buffer → rasterised onto 30 m grid → uint8 binary label. Saved as .npy + .tif.
Class Imbalance — Only 2,309 mineral pixels from 5,794,980 valid pixels (0.04%). Background-to-mineral ratio ~2,509:1. Drives all loss weighting and augmentation decisions.
<br><br>
02 · END-TO-END PIPELINE
1. Data Acquisition - Landsat-8 L2 from USGS EarthExplorer · ASTER 14-band stack · GADM L3 shapefile · Lithology & fault shapefiles.
2. Preprocessing - Radiometric scaling (DN × 0.0000275 − 0.2) · QA_PIXEL cloud masking · Spatial sub-setting to Salem boundary · Valid-pixel mask.
3. Feature Engineering7-band raw · PCA top-3 (97.68% variance) · 5 geological rasters (Lithology, Intrusive, Age, Group Name, D2Fault) · 5 configs: 7ch/3ch/12ch/8ch/15ch 
4. Ground Truth - Shapefile → 90 m buffer → rasterise → crop to Salem → binary uint8 label
5. Patch Extraction & Aug.128×128 patches · stride 64 · NaN thresh 30% · 10× mineral aug / 3× BG aug (flips, rotate, brightness) · zero-overlap split 70/15/15
6. Model TrainingU-Net-PCA3 · AttentionUNet-8ch · ResUNet-15ch · SST-ASTER14 · AdamW · Weighted BCE+Dice · Early stopping7Evaluation & OutputMineral IoU · F1 · Precision · Recall · Confusion matrices · Threshold sweep · Prospectivity map GeoTIFF
<br><br>
03 · MODEL ARCHITECTURES
1. U-Net PCA3 - 3ch PCA - Deep supervision (aux heads D4, D3) - 92(epoch) - 90.67% - 95.11%
2. Attention U-Net - 8ch PCA+Geo - Soft attention gates on skip connections - 120(epoch) - 88.84% - 94.09%
3. ResUNet - 15ch All - Residual blocks + identity shortcuts - 120(epoch) - 88.51% - 93.90%
