import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# تحميل البيانات
df = pd.read_excel("Dummy_ICU_Lab_Data.xlsx")

label_encoders = {}
for col in ["Diagnosis", "Lab Test"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# تحويل إلى 1/0
df["Label"] = df["Label"].map({"Risk": 1, "No Risk": 0})

X = df[["Diagnosis", "Lab Test", "Result", "Previous Result"]]
y = df["Label"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)


# حفظ النموذج 
joblib.dump(model, "risk_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model and encoders saved successfully.")
