# Datasets

Each dataset uses `trainA / trainB / valA / valB` (A = source domain, B = target domain).

```
datasets/
├── maps/               # Google Maps <-> Aerial Photo (pix2pix "maps", 1,096 + 1,096 images)
│   ├── trainA/ (Google Maps)   trainB/ (Aerial Photo)   valA/   valB/
├── cityscapes2cbcl/    # Cityscapes (2,500) -> CBCL StreetScenes (3,000)
│   ├── trainA/ (Cityscapes)    trainB/ (CBCL)           valA/   valB/
├── edges2shoes/        # optional extra task
└── day2night/          # optional extra task
```

GTA → Cityscapes follows the original VSAIT layout used by `configs/gta2cityscapes.yaml`:
`./data/source/{train,val}` (GTA, 10,000 images) and `./data/target/{train,val}` (Cityscapes, 2,500 images).

Download:
- GTA5: https://download.visinf.tu-darmstadt.de/data/from_games/
- Cityscapes (`leftImg8bit`): https://www.cityscapes-dataset.com/
- Google Maps / Aerial (pix2pix `maps`): https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
- CBCL StreetScenes (MIT CBCL, Boston street scenes)
