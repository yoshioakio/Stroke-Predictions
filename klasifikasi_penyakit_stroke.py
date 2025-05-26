import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os

kaggle_dir = os.path.join(os.environ['USERPROFILE'], '.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
kaggle_credentials = {
    "username": "fajriharyanto",
    "key": "998acd734e359f906329500715f20f7e"
}

with open(kaggle_json_path, 'w') as f:
    json.dump(kaggle_credentials, f)
os.chmod(kaggle_json_path, 0o600) 

print(f'File kaggle.json berhasil dibuat di {kaggle_json_path}')

try:
    csv_path = os.path.join(os.path.dirname(__file__), "stroke_prediction_dataset.csv")
    stroke_df = pd.read_csv(csv_path) 
    print(stroke_df.head())
    print(stroke_df.info())
    print(stroke_df.describe())
except FileNotFoundError:
    print("File dataset stroke_prediction_dataset.csv tidak ditemukan. Pastikan path sudah benar.")

stroke_df.head()

stroke_df.info()

stroke_df.describe()

stroke_df.shape

stroke_df.head()

stroke_df.tail()


stroke_df.drop([
    'Patient ID',
    'Patient Name',
    'Hypertension',
    'Symptoms',
    'Marital Status',
    'Cholesterol Levels',
    'Blood Pressure Levels',
    'Stroke History'
], axis=1, inplace=True)

stroke_df.head()

stroke_df.isnull().sum()

stroke_df.duplicated().sum()

stroke_df.info()

sns.boxplot(x=stroke_df['Age'])

sns.boxplot(x=stroke_df['Heart Disease'])

sns.boxplot(x=stroke_df['Average Glucose Level'])

sns.boxplot(x=stroke_df['Body Mass Index (BMI)'])

sns.boxplot(x=stroke_df['Stress Levels'])

num_features = ['Age', 'Heart Disease', 'Average Glucose Level', 'Body Mass Index (BMI)', 'Stress Levels']

Q1 = stroke_df[['Age', 'Heart Disease', 'Average Glucose Level', 'Body Mass Index (BMI)', 'Stress Levels']].quantile(0.25)
Q3 = stroke_df[['Age', 'Heart Disease', 'Average Glucose Level', 'Body Mass Index (BMI)', 'Stress Levels']].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

stroke_df = stroke_df[
    (stroke_df['Age'] >= lower_bound['Age']) & (stroke_df['Age'] <= upper_bound['Age']) &
    (stroke_df['Heart Disease'] >= lower_bound['Heart Disease']) & (stroke_df['Heart Disease'] <= upper_bound['Heart Disease']) &
    (stroke_df['Average Glucose Level'] >= lower_bound['Average Glucose Level']) & (stroke_df['Average Glucose Level'] <= upper_bound['Average Glucose Level']) &
    (stroke_df['Body Mass Index (BMI)'] >= lower_bound['Body Mass Index (BMI)']) & (stroke_df['Body Mass Index (BMI)'] <= upper_bound['Body Mass Index (BMI)']) &
    (stroke_df['Stress Levels'] >= lower_bound['Stress Levels']) & (stroke_df['Stress Levels'] <= upper_bound['Stress Levels'])
]

plt.figure(figsize=(20, 8))
for i, column in enumerate(num_features, 1):
    plt.subplot(2, (len(num_features) + 1) // 2, i)
    sns.boxplot(data=stroke_df, y=column, color="green")
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.show()

stroke_df.head()

num_features = ['Age', 'Heart Disease', 'Average Glucose Level', 'Body Mass Index (BMI)', 'Stress Levels']
cat_features = ['Gender', 'Work Type', 'Residence Type', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Family History of Stroke', 'Dietary Habits', 'Diagnosis']


feature = cat_features[0]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[1]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[2]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[3]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[4]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[5]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[6]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[7]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = cat_features[8]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);


stroke_df.hist(bins=50, figsize=(15,7))
plt.show()

cat_features = stroke_df.select_dtypes(include='object').columns.to_list()

for col in cat_features:
  sns.catplot(x=col, y="Age", kind="bar", dodge=False, height=3, aspect=2, data=stroke_df, hue=col, palette="Set3", legend=False)
  plt.title("Rata-rata umur terhadap - {}".format(col))

plt.show()

sns.pairplot(stroke_df, diag_kind='kde', height=2.5)
plt.show()

correlation_matrix = stroke_df[num_features].corr().round(2)
plt.figure(figsize=(10, 8))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix untuk Fitur Numerik", size=20)
plt.show()

le = LabelEncoder()
stroke_df['Gender encode'] = le.fit_transform(stroke_df['Gender'])

stroke_df['Work Type encode'] = le.fit_transform(stroke_df['Work Type'])

stroke_df['Residence Type encode'] = le.fit_transform(stroke_df['Residence Type'])

stroke_df['Smoking Status encode'] = le.fit_transform(stroke_df['Smoking Status'])

stroke_df['Alcohol Intake encode'] = le.fit_transform(stroke_df['Alcohol Intake'])

stroke_df['Physical Activity encode'] = le.fit_transform(stroke_df['Physical Activity'])

stroke_df['Family History of Stroke encode'] = le.fit_transform(stroke_df['Family History of Stroke'])

stroke_df['Dietary Habits encode'] = le.fit_transform(stroke_df['Dietary Habits'])

stroke_df['Diagnosis encode'] = le.fit_transform(stroke_df['Diagnosis'])

stroke_df.info()


scaler = StandardScaler()

stroke_df[num_features] = scaler.fit_transform(stroke_df[num_features])

stroke_df.head()

encode_features = ['Gender encode', 'Work Type encode', 'Residence Type encode', 'Smoking Status encode', 'Alcohol Intake encode', 'Physical Activity encode', 'Family History of Stroke encode', 'Dietary Habits encode', 'Diagnosis encode']

all_features = (num_features + encode_features)

plt.figure(figsize=(30, 5))
for i, column in enumerate(cat_features, 1):
    plt.subplot(2, (len(cat_features) + 1) // 2, i)
    sns.boxplot(data=stroke_df, y=column, color="green")
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.show()

plt.figure(figsize=(30, 5))
for i, column in enumerate(all_features, 1):
    plt.subplot(2, (len(all_features) + 1) // 2, i)
    sns.boxplot(data=stroke_df, y=column, color="green")
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.show()

stroke_df.describe()


# Buat kolom baru: Smoking Status Encode (0 = sering merokok, 1 = tidak merokok)
def encode_smoking(smoking_val):
    return 0 if smoking_val < 0.5 else 1

stroke_df['Smoking Status encode'] = stroke_df['Smoking Status encode'].apply(encode_smoking)

# Hitung threshold dari data Diagnosis encode = 1 (sudah terkena stroke)
diagnosis_positive = stroke_df[stroke_df['Diagnosis encode'] == 1]

thresholds = {
    'Age': diagnosis_positive['Age'].mean(),
    'Heart Disease': diagnosis_positive['Heart Disease'].mean(),
    'Average Glucose Level': diagnosis_positive['Average Glucose Level'].mean(),
    'Family History of Stroke encode': diagnosis_positive['Family History of Stroke encode'].mean(),
    'Stress Levels': diagnosis_positive['Stress Levels'].mean(),
    'Smoking Status encode': diagnosis_positive['Smoking Status encode'].mean(),
}

def classify_stroke(row):
    score = 0

    # Komponen risiko (dibandingkan threshold penderita stroke)
    if row['Age'] >= thresholds['Age']:
        score += 1
    if row['Heart Disease'] >= thresholds['Heart Disease']:
        score += 1
    if row['Average Glucose Level'] >= thresholds['Average Glucose Level']:
        score += 1
    if row['Family History of Stroke encode'] >= thresholds['Family History of Stroke encode']:
        score += 1
    if row['Stress Levels'] >= thresholds['Stress Levels']:
        score += 1
    if row['Smoking Status encode'] <= thresholds['Smoking Status encode']:  # Sering merokok
        score += 1

    # Skor maksimum: 6, minimum: 0
    if row['Diagnosis encode'] == 1:
        # Sudah terkena stroke
        if score >= 5:
            return 3  # Stroke parah (faktor-faktor sangat tinggi)
        else:
            return 2  # Stroke ringan/sedang
    else:
        # Belum terkena stroke
        if score >= 4:
            return 1  # Berpotensi tinggi
        else:
            return 0  # Aman

stroke_df['Diagnosis Stroke'] = stroke_df.apply(classify_stroke, axis=1)


x = stroke_df[all_features].drop('Age', axis =1)
y = stroke_df['Diagnosis Stroke']

columns_to_scale = x.columns
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

print(f'Jumlah total sampel dalam seluruh dataset: {len(x)}')
print(f'Jumlah total sampel dalam dataset pelatihan (train): {len(X_train)}')
print(f'Jumlah total sampel dalam dataset pengujian (test): {len(X_test)}')

train_distribution = y_train.value_counts()
print("Distribusi Target pada Training Set:")
train_distribution


knn = KNeighborsClassifier(
    n_neighbors=30,
    weights='distance',
    metric='minkowski',
    p=1
)

knn.fit(X_train, y_train)

"""### Random Forest"""

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=123,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

"""### Boosting Algorithm"""

boost = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

boost.fit(X_train, y_train)


models = {
    'K-Nearest Neighbor': knn,
    'Random Forest': rf,
    'Boosting': boost
}

hasil = {}

for nama, model in models.items():
    pred = model.predict(X_test)
    hasil[nama] = [
        accuracy_score(y_test, pred) * 100,
        precision_score(y_test, pred, average='macro') * 100,
        recall_score(y_test, pred, average='macro') * 100,
        f1_score(y_test, pred, average='macro') * 100
    ]

df = pd.DataFrame(hasil, index=["Accuracy", "Precision", "Recall", "F1 Score"])
print("Ringkasan Hasil Metrik Evaluasi Lengkap:")
print(df.round(2))

print("\nAkurasi Training dan Test:")
for name, model in models.items():
  y_train_pred = model.predict(X_train)
  train_acc = accuracy_score(y_train, y_train_pred) * 100
  y_test_pred = model.predict(X_test)
  test_acc = accuracy_score(y_test, y_test_pred) * 100
  print(f"{name}:")
  print(f"  Akurasi Train: {train_acc:.2f}%")
  print(f"  Akurasi Test: {test_acc:.2f}%")

colors = ['skyblue', 'lightcoral', 'lightgreen']
ax = df.T.plot(kind='bar', figsize=(12, 7), color=colors)

plt.title('Perbandingan Metrik Evaluasi Antar Model')
plt.xlabel('Model')
plt.ylabel('Skor (%)')
plt.xticks(rotation=0)
plt.legend(title='Metrik', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

plt.show()