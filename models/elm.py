import numpy as np
from os import path
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# For directory agnoistic csv imports

BASE_DIR = path.dirname(path.abspath(__file__))
ROOT_DIR = path.join(BASE_DIR, "..")
DATASET = path.join(ROOT_DIR, "dataset")

test_csv = path.join(DATASET, "test_labels.csv")
train_csv = path.join(DATASET, "train_labels.csv")

test_dir = path.join(DATASET, "TEST")
train_dir = path.join(DATASET, "TRAIN")

HIDDEN_SIZE = 4096

def image_process_log(success, fails, invalid):
    print(f"Processed={success} Fail={fails} Invalid={invalid}", end="\r", flush=True)


############################################################
def csv_to_gold(df, dirname):
    x, y = [], []
    success, fail, invalid = 0, 0, 0

    for row in df.itertuples():
        image_path = path.join(dirname, row.filename)
        if not path.exists(image_path):
            invalid += 1
            image_process_log(success, fail, invalid)
            continue

        try:
            image = Image.open(image_path).resize((128, 128)).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
            x.append(array.ravel())
            y.append(row._4)

            success += 1
            image_process_log(success, fail, invalid)
        except:
            fail += 1
            image_process_log(success, fail, invalid)

    x = np.array(x)
    y = np.array(y)
    
    print("\nDONE") # Also prevents final line override

    return x, y

############################################################
def activation(x):
    return 1 / (1 + np.exp(-x))

############################################################
def predict(x, weights, bias, beta):
    H = activation(np.dot(x, weights) + bias.reshape(1, -1))
    return np.dot(H, beta)

############################################################
def run_from_model(filepath):
        if not path.exists("elm_model.npz"):
            print("[!] Model doesn't exist, what are you trying to do?")
            return
        
        image = Image.open(filepath).resize((128, 128)).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        x = np.array([array.ravel()])

        saved_model = np.load("elm_model.npz")
        weights = saved_model["weights"]
        bias = saved_model["bias"]
        beta = saved_model["beta"]

        predicted = np.argmax(predict(x, weights, bias, beta), axis=1)

        df = pd.read_csv(train_csv)[['class']]

        y = []

        for row in df.itertuples():
            y.append(row._1)
        
        y = np.array(y).reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit_transform(y)

        return encoder.categories_[0][predicted[0]]

class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.weights = np.random.randn(self.input_size, self.hidden_size)
        self.bias = np.random.randn(self.hidden_size)

        self.beta = None
    
    def train(self, x, y):
        H = activation(np.dot(x, self.weights) + self.bias.reshape(1, -1))
        self.beta = np.dot(np.linalg.pinv(H), y)


    @staticmethod
    def start_train():
        df = pd.read_csv(train_csv)[:1000]

        x, y = csv_to_gold(df, train_dir)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        y_encoded = encoder.fit_transform(y.reshape(-1, 1))

        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, train_size=0.8, random_state=3)

        elm = ELM(x_train.shape[1], HIDDEN_SIZE, y_train.shape[1])

        if path.exists("elm_model.npz"):
            saved_model = np.load("elm_model.npz")
            elm.weights = saved_model["weights"]
            elm.bias = saved_model["bias"]
            elm.beta = saved_model["beta"]
        else:
            elm.train(x_train, y_train)
            np.savez("elm_model.npz", bias=elm.bias, weights=elm.weights, beta=elm.beta)

        y_pred = predict(x_test, elm.weights, elm.bias, elm.beta)

        y_pred_max = np.argmax(y_pred, axis=1)
        y_test_max = np.argmax(y_test, axis=1)

        print("Accuracy on seen dataset: %.2f" % (accuracy_score(y_test_max, y_pred_max) * 100))

        return elm, encoder

    @staticmethod
    def start_final_test(elm, encoder):
        df = pd.read_csv(test_csv)[:1000]
        x_test, y_test = csv_to_gold(df, test_dir)

        y_encoded = encoder.transform(y_test.reshape(-1, 1))

        y_pred = predict(x_test, elm.weights, elm.bias, elm.beta)

        y_pred_max = np.argmax(y_pred, axis=1)
        y_test_max = np.argmax(y_encoded, axis=1)

        print("Accuracy on unseen dataset: %.2f" % (accuracy_score(y_test_max, y_pred_max) * 100))

        exit(0)

if __name__ == "__main__":
    # elm, encoder = ELM.start()
    # ELM.final_test(elm, encoder)

    run_from_model("/home/erucix/Programming/elm_plant_disease_classification/dataset/TRAIN/xgray-leaf-spot.jpg.pagespeed.ic.ltF8KfKtQ1.jpg")