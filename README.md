# 🍃 ELM Plant Disease Classification

Classify plant leaf diseases from images using an **Extreme Learning Machine (ELM)** — a fast, lightweight alternative to deep neural networks — served through a simple Flask web app.

Upload a leaf photo and get a real-time prediction of the disease class, without the training overhead of a full deep learning pipeline.

---

## ✨ Key Features

- **Extreme Learning Machine (ELM):** A single-hidden-layer feed-forward network trained via closed-form least-squares, making training near-instant compared to backprop-based CNNs.
- **Flask Web Interface:** Simple upload form — drop in an image, get a classification back in the browser.
- **Model Persistence:** Trained ELM weights are serialized and reloaded at inference time, so the app doesn't retrain on every request.
- **Robust File Handling:** Validates file type (`png`, `jpg`, `jpeg`, `gif`), enforces a max upload size, and uses `secure_filename` to sanitize uploads.
- **Notebook-Driven Training:** The full data prep, feature extraction, and ELM training workflow lives in a Jupyter notebook for transparency and easy experimentation.
- **Extensible:** Swap in a new dataset or add disease classes without changing the app's architecture.

---

## 📁 Project Structure

```
elm_leaf_disease_classification/
├── app.py                    # Flask app entry point (upload + inference routes)
├── elm_classificaion.ipynb   # Notebook: data prep, training, and evaluation of the ELM model
├── requirements.txt          # Python dependencies
├── models/
│   └── elm.py                 # ELM class, training pipeline, and run_from_model() inference helper
├── templates/
│   └── index.html              # Upload form + results page (Jinja2)
├── utils/                    # Helper functions
├── images/                   # Sample/reference leaf images
├── dataset/                  # Expected at runtime: TRAIN/, TEST/, train_labels.csv, test_labels.csv
└── README.md
```

> **Note:** `models/elm.py` expects a `dataset/` directory one level up from `models/` (i.e. at the project root) containing `TRAIN/` and `TEST/` image folders plus `train_labels.csv` and `test_labels.csv`. This dataset isn't included in the repo and needs to be added separately (see [Dataset](#-dataset) below).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/binary-beater/elm_leaf_disease_classification.git
   cd elm_leaf_disease_classification
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

```
numpy
scikit-learn
pandas
pillow
flask
```

### Running the App

```bash
python app.py
```

By default, Flask will start a local development server (debug mode enabled). Open your browser to:

```
http://127.0.0.1:5000
```

Upload a leaf image (`.png`, `.jpg`, `.jpeg`, or `.gif`, up to 16MB) and the app will return the predicted disease class.

> Uploaded files are temporarily stored in a local `tmp/` folder, which is created automatically on first run.

---

## 📊 Dataset

`models/elm.py` looks for the following layout at the project root (a sibling of `models/`):

```
dataset/
├── TRAIN/                # Training images
├── TEST/                 # Held-out test images
├── train_labels.csv      # Maps filenames -> class labels for TRAIN/
└── test_labels.csv       # Maps filenames -> class labels for TEST/
```

The label CSVs are expected to have a `filename` column and a `class` column. Each row's filename is looked up inside the corresponding image directory, resized to `128x128`, converted to RGB, and flattened into a normalized feature vector.

## 🧠 Model Training

Training logic lives in both [`elm_classificaion.ipynb`](./elm_classificaion.ipynb) and [`models/elm.py`](./models/elm.py). The `ELM` class in `elm.py` implements the core algorithm:

- **Input:** flattened, normalized `128x128x3` RGB images
- **Hidden layer:** `HIDDEN_SIZE = 4096` randomly initialized weights and biases, passed through a sigmoid activation
- **Output layer (`beta`):** solved in closed form using the Moore-Penrose pseudoinverse (`np.linalg.pinv`) of the hidden-layer activations — no gradient descent, which is why ELM training is so fast
- **Labels:** one-hot encoded via `sklearn`'s `OneHotEncoder`

`ELM.start_train()`:
1. Loads the first 1000 rows of `train_labels.csv` and builds feature/label arrays via `csv_to_gold`
2. Splits into an 80/20 train/validation split
3. If a saved model (`elm_model.npz`) already exists, it's loaded instead of retraining; otherwise a new ELM is trained and its `weights`, `bias`, and `beta` are saved to `elm_model.npz`
4. Reports accuracy on the held-out validation split

`ELM.start_final_test()` then evaluates the trained model against `test_labels.csv` (first 1000 rows) for a final unseen-data accuracy check.

To retrain or experiment with the model, open the notebook:

```bash
jupyter notebook elm_classificaion.ipynb
```

Or run the training routine directly:

```bash
python -c "from models.elm import ELM; elm, encoder = ELM.start_train(); ELM.start_final_test(elm, encoder)"
```

This produces `elm_model.npz`, which `app.py` (via `run_from_model`) loads at inference time.

---

## 🔧 How It Works

1. A user uploads an image through the upload form in `templates/index.html`.
2. `app.py` validates the file extension (`png`/`jpg`/`jpeg`/`gif`), sanitizes the filename with `secure_filename`, and saves it into the `tmp/` upload folder.
3. `app.py` calls `run_from_model(filepath)` from `models/elm.py`, which:
   - Opens the image, resizes it to `128x128`, converts to RGB, and normalizes pixel values to `[0, 1]`
   - Loads the trained weights, bias, and beta from `elm_model.npz`
   - Runs a forward pass (`sigmoid(x · weights + bias) · beta`) and takes the `argmax` over the output
   - Maps that index back to a class name using the same `OneHotEncoder` categories derived from `train_labels.csv`
4. `app.py` renders `index.html` again, this time passing `result_string` (the predicted class) and `uploaded_filename`, which are displayed in the results section of the page.

Flash messages (success/warning/error) are shown above the form for upload issues such as missing files or invalid file types.

---

## 🗺️ Roadmap / Ideas for Extension

- [ ] Add more plant species and disease categories
- [ ] Display prediction confidence scores
- [ ] Add a REST API endpoint (JSON in/out) alongside the HTML UI
- [ ] Containerize with Docker for easier deployment
- [ ] Add automated tests for preprocessing and inference

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, bug fixes, or new disease datasets.
