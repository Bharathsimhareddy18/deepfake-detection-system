# Deepfake Image Classification with MobileNet Transfer Learning

## Overview

This project implements a deepfake image classifier using MobileNet and transfer learning in TensorFlow 2.17.0. It includes an exploratory data analysis (EDA) module for dataset validation and visualization, as well as a simple Flask application for web deployment of the trained model. The solution is optimized for CPU training with limited resources—ideal for experimenting on personal laptops or entry-level machines.

## Features

- **Transfer learning with MobileNet:** Fast and accurate model, compatible with TensorFlow 2.17 and Keras 3.0
- **Exploratory Data Analysis (EDA):** Automated scripts to check dataset structure, class distribution, and image properties; generates informative visualizations
- **Flask web interface:** User-friendly frontend to upload images and detect whether they are "Real" or "Fake"
- **Fully reproducible requirements:** All dependencies curated for CPU, deep learning, web deployment, and visualization
- **Class-balanced, multi-split dataset:** Out-of-the-box support for balanced real/fake classes across train, validation, and test splits

## Project Structure

```
├── train_mobilenet_transfer.py   # Model training with MobileNet transfer learning
├── eda_analysis.py              # Dataset exploration and visualization
├── app.py                       # Flask web application for model deployment
├── requirements.txt             # Python package dependencies
├── dataset_distribution.jpg     # Example dataset visualization
└── README.md                   # This file
```

### File Descriptions

- **`train_mobilenet_transfer.py`** — Model training using TensorFlow and MobileNet with custom data generators, augmentation, staged training, and automatic checkpointing
- **`eda_analysis.py`** — EDA tool for class balance, basic statistics, dataset visuals, and quick structure validation. Generates pie and bar charts
- **`app.py`** — Flask deployment file for loading the trained model and serving the prediction interface to users
- **`requirements.txt`** — All required packages and pinned versions for full reproducibility

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd <repo-directory>
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
Ensure your images are organized in the following directory structure:
```
Dataset/
├── train/
│   ├── real/
│   └── fake/
├── validation/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### 4. Run EDA & Visualize Dataset
```bash
python eda_analysis.py
```
This creates `dataset_distribution.png` showing the distribution of real and fake images across splits.

### 5. Train the Model
```bash
python train_mobilenet_transfer.py
```
The trained model will be saved as `models/deepfakemobilenetfinal.keras`.

### 6. Launch the Web Application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser to use the prediction interface.

## Key Technologies

| Technology      | Version | Purpose                           |
|-----------------|---------|----------------------------------|
| TensorFlow      | 2.17.0  | Deep learning backend            |
| MobileNet       | -       | Pretrained image model           |
| Flask           | 3.0.3   | Web API and UI                   |
| Matplotlib      | 3.9.4   | Plots/bar/pie charts             |
| Pillow          | 10.4.0  | Image loading/processing         |
| OpenCV          | 4.10.0  | Computer vision operations       |
| Seaborn         | 0.13.2  | Data visualization styling       |
| NumPy           | 1.24.3  | Numerical computations           |
| Scikit-learn    | 1.3.0   | Metrics and evaluation           |

## Dataset Distribution Example

The project works with balanced datasets. Here's an example distribution:

| Split      | Real   | Fake   | Total  |
|------------|--------|--------|--------|
| Train      | 70,001 | 70,001 | 140,002|
| Validation | 19,787 | 19,641 | 39,428 |
| Test       | 5,413  | 5,492  | 10,905 |
| **Total**  | **95,201** | **95,134** | **190,335** |

## Model Architecture

- **Base Model:** MobileNet (pretrained on ImageNet)
- **Input Size:** 224x224x3
- **Custom Layers:** 
  - Global Average Pooling
  - Batch Normalization
  - Dense Layer (128 units)
  - Dropout (0.5)
  - Output Layer (1 unit, sigmoid activation)

## Training Configuration

- **Batch Size:** 8 (optimized for limited memory)
- **Learning Rate:** 1e-4
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Web Application Features

- **Image Upload:** Support for common image formats (JPG, JPEG, PNG, BMP, TIFF)
- **Real-time Prediction:** Instant classification with confidence scores
- **Clean Interface:** Simple HTML interface for easy interaction
- **Model Loading:** Automatic loading of trained model on startup

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MobileNet architecture by Google
- TensorFlow team for the excellent deep learning framework
- Flask community for the lightweight web framework

## Contact

If you have any questions or suggestions, feel free to open an issue or contact the maintainers.

---

**Note:** This project is designed for educational and research purposes. Ensure you have proper rights to use any datasets and comply with relevant data protection regulations.
