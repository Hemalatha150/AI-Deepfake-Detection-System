# AI-Powered Deepfake Detection System for Image and Video Authenticity Verification

**B.Tech Project Work — Department of Information Technology, S R K R Engineering College (A), Bhimavaram**

A deep-learning system that decides whether an uploaded **image or video** is
**Original** or a **Deepfake** (AI-generated or manipulated). It reports a
confidence score and explains its decision visually. It runs as a
**Streamlit** web application and also has a command-line interface.

---

## 1. Features

| Feature | Description |
|---|---|
| Image detection | Detects every face, classifies each one and marks it on the image |
| Video detection | Samples frames evenly, classifies each face and aggregates over time |
| Confidence score | Probability of manipulation, plus a calibrated confidence |
| Temporal analysis | Per-frame probability timeline, fake-frame ratio and temporal jitter (flicker) |
| Two detectors | Face model (close-up face analysis) + scene model (whole-image AI-generation detection) |
| Explainability | **Grad-CAM** heatmaps show which facial regions drove the decision |
| Forensics | **Error Level Analysis (ELA)** and **FFT frequency spectrum** visualisations |
| Robust face detection | **MTCNN** → **Dlib** (HOG) → OpenCV Haar fallback chain |
| Multiple CNNs | EfficientNet-B0 / Xception (transfer learning) or a custom CNN |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix and ROC curve (scikit-learn) |

## 2. System Architecture

```
            ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
 Upload ──► │ Image/Video  │ ──► │ Face Detection   │ ──► │ Face crop + margin   │
            │ (Streamlit)  │     │ MTCNN / Dlib     │     │ resize 224×224       │
            └──────────────┘     └──────────────────┘     └──────────┬───────────┘
                   │ (video: N evenly spaced frames)                  │
                   ▼                                                  ▼
            ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
            │ Verdict +    │ ◄── │ Aggregation      │ ◄── │ CNN classifier       │
            │ confidence,  │     │ image: max(face) │     │ EfficientNet-B0      │
            │ Grad-CAM,    │     │ video: mean(fr.) │     │ → sigmoid P(fake)    │
            │ ELA, FFT     │     │ + temporal stats │     └──────────────────────┘
            └──────────────┘     └──────────────────┘
```

**Model:** EfficientNet-B0 pretrained on ImageNet → Global Average Pooling →
Dropout → Dense(128, ReLU) → Dropout → Dense(1, sigmoid).
Training happens in two phases:
1. Train the head only, with the backbone frozen.
2. Fine-tune the top 30 % of the backbone at a 10× lower learning rate.

The decision threshold is then **calibrated** on the validation set by
maximising Youden's J statistic.

**Training augmentation:** horizontal flip, brightness/contrast/saturation
jitter, and random **JPEG re-compression (quality 60–100)**. The JPEG step
simulates the compression that social-media platforms apply.

## 3. Project Structure

```
deepfake/
├── app.py                 # Streamlit web application
├── train.py               # model training (two-phase transfer learning)
├── evaluate.py            # test-set metrics, confusion matrix, ROC curve
├── predict.py             # command-line prediction for images/videos
├── preprocess.py          # extract face crops from raw images/videos into a dataset
├── download_dataset.py    # download the 140k Real & Fake Faces dataset
├── config.py              # paths and hyper-parameters
├── requirements.txt
├── src/
│   ├── face_detector.py   # MTCNN / Dlib / Haar face detection and cropping
│   ├── model.py           # CNN architectures, freezing, save/load
│   ├── data.py            # dataset discovery + tf.data pipeline + augmentation
│   ├── predictor.py       # inference engine for images and videos
│   ├── video.py           # frame sampling
│   ├── gradcam.py         # Grad-CAM explainability
│   └── forensics.py       # Error Level Analysis, frequency spectrum
├── models/                # trained model (.keras) + metadata (.json)
├── reports/               # training curves, metrics, confusion matrix, ROC curve
├── tests/smoke_test.py    # end-to-end pipeline test on synthetic data
└── data/faces/{train,val,test}/{real,fake}/   # dataset (not committed)
```

## 4. Installation

Requires Python 3.10+ (tested on Python 3.13, Windows 11, CPU only).

```bash
pip install -r requirements.txt
```

## 5. Dataset

### Option A — 140k Real and Fake Faces (used for the included model)
70,000 real faces (Flickr-Faces-HQ) and 70,000 StyleGAN-generated faces.

```bash
python download_dataset.py                                # ~1.1 GB: 16.7k train, 10k val, 10k test
python download_dataset.py --train-shards 6 --eval-shards 2   # full dataset (~4 GB)
```

### Option B — your own images / videos (e.g. FaceForensics++, Celeb-DF, DFDC)
Arrange the raw files like this:

```
data/raw/real/...   (images and/or videos)
data/raw/fake/...
```

Then extract the face crops:

```bash
python preprocess.py --input data/raw --output data/faces --frames-per-video 15
```

The train/val/test split is made **per source file**, so frames from one
video never appear in both the training data and the test data. Datasets that
are already split (`train/val/test` folders) keep their split.

## 6. Training

```bash
python train.py                                   # EfficientNet-B0, 224 px
python train.py --arch xception
python train.py --arch cnn --img-size 128 --epochs 20   # custom CNN, fast on CPU
python train.py --limit-per-class 5000            # use a subset for a quicker run
```

Outputs:
- `models/deepfake_detector.keras` and `models/deepfake_detector.json` (metadata, including the calibrated threshold)
- `reports/training_curves.png`

## 7. Evaluation

```bash
python evaluate.py
```

This writes `reports/metrics.json`, `classification_report.txt`,
`confusion_matrix.png` and `roc_curve.png`.

## 8. Running the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 and upload an image (`jpg/png/bmp/webp`) or a
video (`mp4/avi/mov/mkv/webm`).

The sidebar lets you:
- adjust the decision threshold;
- set the number of video frames to sample;
- choose the face detector;
- toggle Grad-CAM and the forensic views.

Command line:

```bash
python predict.py photo.jpg video.mp4
python predict.py video.mp4 --frames 30 --json
```

## 9. Results

**Model:** EfficientNet-B0, 224×224 input, trained on CPU.

**Training (two rounds):**
1. 3 head epochs + 4 fine-tuning epochs on the 140k dataset (one shard).
2. 2 further fine-tuning epochs with **camera-robust augmentation** (sensor noise, blur,
   low resolution, JPEG 30–100, lighting changes), adding **6,000 real LFW photos**
   to the real class.

**Training data:** 22,667 images (14,334 real, 8,333 StyleGAN fake).

**Test data:** 11,339 held-out images — 5,000 StyleGAN fakes, 5,000 FFHQ real faces
and 1,339 LFW real photos.

**Decision threshold:** 0.845, calibrated on validation data.

| Metric | Score |
|---|---|
| Accuracy | **95.94 %** |
| Precision (Deepfake) | 95.55 % |
| Recall (Deepfake) | 95.24 % |
| F1-score | 95.39 % |
| ROC-AUC | **0.9931** |

**Robustness to real camera footage.** The first model wrongly flagged a genuine
webcam recording as a deepfake (84 % fake), because it had only ever seen sharp,
studio-quality real photos. After the robustness training:
- The same recording is classified as **Original** (P(fake) ≈ 0.15, against a threshold of 0.845).
- On real faces degraded to look like video (blur + low resolution + compression),
  false alarms fell from 20 % to about 10 %.
- About 95 % of fakes are still caught.

Plots are in `reports/`:
- `training_curves.png`
- `confusion_matrix.png`
- `roc_curve.png`

Inference on CPU takes about 1.3 s per image.

### Scene model (whole-image AI-generation detector)

A second EfficientNet-B0 classifies the **whole image**. It catches modern AI pictures that
the face model cannot judge:
- diffusion images (Midjourney, Stable Diffusion, ...);
- images whose faces are tiny;
- images that contain no face at all.

**Training data:** 17,270 images, prepared with `prepare_modern_data.py`:
- Tiny-GenImage: real photos vs Midjourney, SD 1.4/1.5, ADM, GLIDE, VQDM, Wukong and BigGAN images;
- DeepFakeFace: real IMDB-WIKI photos vs SD text2img, SD inpainting and InsightFace face swaps;
- FFHQ, StyleGAN and LFW images.

All images are centre-cropped to a square and re-encoded identically. This stops the model
from using file format or aspect ratio as a shortcut. At inference, three square crops of
the image are averaged.

**Performance:** validation ROC-AUC 0.864. The scene model runs at a **low-false-alarm
threshold (0.80)**, so it only overrides the face model when it is very confident.

| Test images at threshold 0.80 | Flagged as fake |
|---|---|
| Real photos of people (FFHQ / LFW / IMDB-WIKI) | 1.8 % / 0 % / 5.6 % |
| Real ImageNet photos | 12.0 % |
| Midjourney / SD 1.5 / Wukong | 56.9 % / 70.5 % / 76.6 % |
| SD text2img faces | 69.8 % |

The scene model is the weaker of the two detectors. Detecting AI generation from a whole
scene is a much harder, open research problem than close-up face analysis.

**Combining the models.** Scores from both models are calibrated so that each model's own
threshold maps to 0.5. An image is reported as a **Deepfake** if either model is confident.

## 10. Testing

```bash
python tests/smoke_test.py
```

This runs preprocessing, training, evaluation and image + video prediction on a
small synthetic dataset, to check that the whole pipeline works.

## 11. Limitations and Future Work

- The included model is trained on **StyleGAN-generated faces**. It detects
  fully synthetic faces well. For **face-swap / re-enactment video deepfakes**,
  retrain it on FaceForensics++, Celeb-DF or DFDC with `preprocess.py`.
- New generators that are not represented in the training data can evade
  detection. Retrain periodically on newer data.
- Future work:
  - temporal models (CNN + LSTM, or a video transformer) over consecutive frames;
  - audio-visual lip-sync analysis;
  - ensembles of several backbones.

## 12. Technologies

Python · TensorFlow / Keras · OpenCV · Streamlit · CNN (EfficientNet-B0, Xception) ·
MTCNN · Dlib · NumPy · Pandas · Scikit-learn · Matplotlib

## 13. References

1. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
2. F. Chollet, *Deep Learning with Python*, Manning Publications.
3. M. Tan, Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
4. F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017.
5. K. Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks (MTCNN)", IEEE SPL 2016.
6. A. Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019.
7. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
8. TensorFlow Documentation — https://www.tensorflow.org
9. OpenCV Documentation — https://docs.opencv.org
