import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score




df = pd.read_csv("salary_train.csv")
df = df.dropna(subset=['salary'])

X = df.drop(columns=['ID', 'salary'])
y = df['salary']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Preprocessing
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
voting_model = VotingRegressor([('rf', rf_model), ('gb', gb_model)])

# Pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", voting_model)
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
print(f"R² Score: {r2:.4f}")


import pickle
with open("pbel_project.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)


# GUI
def predict_salary_gui():
    try:
        input_data = {
            'education_level': education_level.get(),
            'years_experience': float(years_experience.get()),
            'job_title': job_title.get(),
            'industry': industry.get(),
            'location': location.get(),
            'company_size': company_size.get(),
            'certifications': float(certifications.get()),
            'age': float(age.get()),
            'working_hours': float(working_hours.get()),
            'crucial_code': crucial_code.get()
        }

        new_employee = pd.DataFrame([input_data])
        prediction = model_pipeline.predict(new_employee)
        result.set(f"Predicted Salary: ₹{prediction[0]:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


root = tk.Tk()
root.title("Salary Prediction App")

fields = {
    "Education Level": "education_level",
    "Years of Experience": "years_experience",
    "Job Title": "job_title",
    "Industry": "industry",
    "Location": "location",
    "Company Size": "company_size",
    "Certifications": "certifications",
    "Age": "age",
    "Working Hours per Week": "working_hours",
    "Crucial Code": "crucial_code"
}

# Widgets
entries = {}
for i, (label_text, var_name) in enumerate(fields.items()):
    tk.Label(root, text=label_text).grid(row=i, column=0, sticky="e", padx=10, pady=5)
    entry = tk.Entry(root, width=40)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[var_name] = entry

education_level = entries["education_level"]
years_experience = entries["years_experience"]
job_title = entries["job_title"]
industry = entries["industry"]
location = entries["location"]
company_size = entries["company_size"]
certifications = entries["certifications"]
age = entries["age"]
working_hours = entries["working_hours"]
crucial_code = entries["crucial_code"]

result = tk.StringVar()
tk.Button(root, text="Predict Salary", command=predict_salary_gui).grid(row=len(fields), columnspan=2, pady=10)
tk.Label(root, textvariable=result, font=("Arial", 14), fg="green").grid(row=len(fields)+1, columnspan=2)

root.mainloop()
