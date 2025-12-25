import emlearn
import numpy as np
import polars as pl
import tl2cgen
import treelite
import xgboost
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

df_train_pos = pl.read_csv("sufficient.csv")
df_train_pos = pl.concat([df_train_pos, pl.read_csv("sufficient2.csv").drop("filename")])
df_train_pos = df_train_pos.with_columns(pl.lit(1).alias("sufficient"))

df_train_neg = pl.read_csv("insufficient.csv")
df_train_neg = pl.concat([df_train_neg, pl.read_csv("insufficient2.csv").drop("filename")])
df_train_neg = df_train_neg.with_columns(pl.lit(0).alias("sufficient"))

df_train = pl.concat([df_train_neg, df_train_pos]).unique()

df_val_pos = pl.read_csv("sufficient2_val.csv").drop("filename")
df_val_pos = df_val_pos.with_columns(pl.lit(1).alias("sufficient"))

df_val_neg = pl.read_csv("insufficient2_val.csv").drop("filename")
df_val_neg = df_val_neg.with_columns(pl.lit(0).alias("sufficient"))

df_val = pl.concat([df_val_neg, df_val_pos]).unique()

df_train = df_train.with_columns(pl.col("zoom_level").add(1.0).log())
df_val = df_val.with_columns(pl.col("zoom_level").mul(1.21).add(1.0).log())

X_train = df_train.drop("sufficient")
y_train = df_train["sufficient"]

X_val = df_val.drop("sufficient")
y_val = df_val["sufficient"]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

model = xgboost.XGBClassifier(n_estimators=3, max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_val)
pred_probs = model.predict_proba(X_val)[:,0]

print(metrics.accuracy_score(y_val, preds))
print(metrics.recall_score(y_val, preds))
print(metrics.roc_auc_score(y_val, pred_probs))
print(metrics.confusion_matrix(y_val, preds))

# cmodel = emlearn.convert(model, method="inline")
# cmodel.save(file="test.h", name="test")

tl_model = treelite.frontend.from_xgboost(model.get_booster())
tl2cgen.generate_c_code(tl_model, "./test", params={})

model = MLPClassifier(hidden_layer_sizes=(16, 16, 16, 16, 16,), 
                      activation='relu', 
                      solver='adam', 
                    #   max_iter=100,
                        learning_rate="adaptive",
                      early_stopping=True,
                      random_state=1)
model.fit(X_train, y_train)

preds = model.predict(X_val)
pred_probs = model.predict_proba(X_val)[:,0]

print(metrics.accuracy_score(y_val, preds))
print(metrics.recall_score(y_val, preds))
print(metrics.roc_auc_score(y_val, pred_probs))
print(metrics.confusion_matrix(y_val, preds))

with open("test.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write(f"inline constexpr int N_INPUTS = {model.n_features_in_};\n")
    f.write(f"inline constexpr int N_LAYERS = {model.n_layers_ - 2};\n")
    f.write(f"inline constexpr int LAYER_SIZES[] = {{{", ".join(str(l.shape[0]) for l in model.coefs_[1:])}}};\n")

    def write_array(name: str, array: np.ndarray, align: bool = True) -> None:
        if align:
            f.write("alignas(64) ")
        f.write(f"inline constexpr float {name}[] = {{")
        f.write(", ".join(map(str, array.astype(np.float32).flatten())))
        f.write("};\n")

    write_array("INPUT_MEANS", scaler.mean_, False)
    write_array("INPUT_STDS", scaler.scale_, False)
    write_array("W0", model.coefs_[0])
    write_array("B0", model.intercepts_[0])
    write_array("W1", model.coefs_[1])
    write_array("B1", model.intercepts_[1])
    write_array("W2", model.coefs_[2])
    write_array("B2", model.intercepts_[2])
    write_array("W3", model.coefs_[3])
    write_array("B3", model.intercepts_[3])
    write_array("W4", model.coefs_[4])
    write_array("B4", model.intercepts_[4])
    