import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

MODEL_PATH = "campus_placement_model"

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("Error loading model:", e)

MAPS = {
    "Gender": {"M": 1, "F": 0},
    "SSC Board": {"Central": 1, "Others": 0},
    "HSC Board": {"Central": 1, "Others": 0},
    "HSC Stream": {"Commerce": 1, "Science": 2, "Arts": 0},
    "Degree Trade": {"Sci&Tech": 1, "Comm&Mgmt": 2, "Others": 0},
    "Specialisation": {"Mkt&Fin": 1, "Mkt&HR": 0},
    "Work Experience": {"Yes": 1, "No": 0},
}

FEATURES = [
    ("Gender", ["M", "F"]),
    ("SSC Percentage", None),
    ("SSC Board", ["Central", "Others"]),
    ("HSC Percentage", None),
    ("HSC Board", ["Central", "Others"]),
    ("HSC Stream", ["Commerce", "Science", "Arts"]),
    ("Degree Percentage", None),
    ("Degree Trade", ["Sci&Tech", "Comm&Mgmt", "Others"]),
    ("Work Experience", ["Yes", "No"]),
    ("Etest Percentage", None),
    ("Specialisation", ["Mkt&Fin", "Mkt&HR"]),
    ("MBA Percentage", None),
]

class PlacementGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Campus Placement Tester")
        self.root.geometry("700x780")
        self.root.resizable(False, False)

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Campus Placement Prediction", font=("Segoe UI", 16, "bold")).pack(pady=(0, 12))
        ttk.Label(frame, text="Enter features and click Predict", font=("Segoe UI", 10)).pack(pady=(0, 8))

        self.entries = {}
        for name, choices in FEATURES:
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=name, width=18, anchor="w").pack(side="left")
            if choices is None:
                ent = ttk.Entry(row)
                ent.pack(side="left", fill="x", expand=True)
                self.entries[name] = ent
            else:
                cb = ttk.Combobox(row, values=choices, state="readonly")
                cb.current(0)
                cb.pack(side="left", fill="x", expand=True)
                self.entries[name] = cb

        btn = ttk.Button(frame, text="Predict", command=self.on_predict)
        btn.pack(pady=12)

        self.result_var = tk.StringVar(value="Model not loaded" if model is None else "Ready")
        ttk.Label(frame, textvariable=self.result_var, font=("Segoe UI", 11, "bold")).pack(pady=5)

    def on_predict(self):
        global model
        if model is None:
            messagebox.showerror("Error", "Model file not found. Train and save the model first.")
            return

        try:
            values = []
            for name, choices in FEATURES:
                val = self.entries[name].get().strip()
                if choices is None:
                    if val == "":
                        raise ValueError(f"{name} is required")
                    values.append(float(val))
                else:
                    values.append(MAPS[name][val])

            arr = np.array(values).reshape(1, -1)
            pred = model.predict(arr)[0]
            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(arr)[0]

            if pred == 1:
                text = "Prediction: Placed"
                if probs is not None:
                    text += f" (Confidence: {probs[1]*100:.2f}%)"
            else:
                text = "Prediction: Not Placed"
                if probs is not None:
                    text += f" (Confidence: {probs[0]*100:.2f}%)"
            self.result_var.set(text)
        except Exception as e:
            messagebox.showerror("Input error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PlacementGui(root)
    root.mainloop()
