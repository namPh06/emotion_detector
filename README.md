# He thong Nhan dien Cam xuc Lop hoc
## Emotion Recognition for Learning Analytics

**Dataset chinh:** DAiSEE  
**Pretrain tuy chon:** FER2013  
**Backbone khuyen nghi:** EfficientNetV2B0  
**Training:** Python script hoac Google Colab GPU  
**Inference:** Webcam real-time tren may ca nhan

---

## Cau truc project

```text
TTCS1/
├── README.md
├── main.py
├── train_best_model.py
├── requirements.txt
├── colab/
│   ├── 01_DAiSEE_Training.ipynb
│   ├── 02_FER2013_Pretrain.ipynb
│   ├── 03_DAiSEE_Training_Improved.ipynb
│   └── 04_Full_Pipeline_Best_Accuracy.ipynb
├── emotion_results/
├── models/
│   └── emotion_model_daisee.h5
└── utils/
    └── label_mapping.py
```

## Pipeline train khuyen nghi

File nen dung mac dinh la `train_best_model.py`. Script nay chuan hoa pipeline train de cho ra model moi on dinh hon cac notebook cu:

1. Dung `EfficientNetV2B0` pretrained ImageNet.
2. Pretrain them tren FER2013 neu ban co dataset nay.
3. Fine-tune tren DAiSEE voi:
   - augmentation manh nhung an toan
   - `CategoricalFocalCrossentropy`
   - class weights xu ly mat can bang
   - early stopping + reduce LR + checkpoint
4. Xuat model tot nhat vao `training_runs/best_model/` va tu copy sang `models/emotion_model_daisee.h5`.

## Chuan bi du lieu

### DAiSEE

Script moi can bo du lieu DAiSEE da duoc chuan bi theo 4 lop:

```text
data/daisee_4class/
├── train/
│   ├── 0  1  2  3
├── val/
│   ├── 0  1  2  3
└── test/
    ├── 0  1  2  3
```

Y nghia lop:
- `0`: Buon chan
- `1`: Tap trung
- `2`: Hung thu
- `3`: Binh thuong

### FER2013

Neu co FER2013, script se tu anh xa:
- `angry`, `disgust`, `fear`, `sad` -> `0`
- `happy`, `surprise` -> `2`
- `neutral` -> `3`

## Cach train model moi

### Cai thu vien

```bash
pip install -r requirements.txt
```

### Train chi voi DAiSEE

```bash
python train_best_model.py --daisee-dir data/daisee_4class
```

### Train voi FER2013 + DAiSEE

```bash
python train_best_model.py --daisee-dir data/daisee_4class --fer-dir data/fer2013
```

### Vi du tuy chinh tham so

```bash
python train_best_model.py ^
  --daisee-dir data/daisee_4class ^
  --fer-dir data/fer2013 ^
  --img-size 260 ^
  --batch-size 32 ^
  --daisee-epochs-head 10 ^
  --daisee-epochs-finetune 14
```

## Ket qua sau khi train

Script se tao:
- `training_runs/best_model/checkpoints/`
- `training_runs/best_model/training_curves.png`
- `training_runs/best_model/confusion_matrix.png`
- `training_runs/best_model/training_summary.json`
- `training_runs/best_model/emotion_model_daisee_best.h5`

Va dong thoi cap nhat model inference tai:

```text
models/emotion_model_daisee.h5
```

## Chay realtime voi webcam

```bash
python main.py
```

## Notebook cu

Cac notebook trong `colab/` van duoc giu lai de tham khao hoac chay tung buoc tren Colab. Tuy nhien neu muc tieu la train ra model moi tot va lap lai on dinh, nen uu tien `train_best_model.py`.
