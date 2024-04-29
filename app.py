from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from flask import render_template
import matplotlib
from flask import render_template_string
from xgboost import plot_tree
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sklearn
import os
from io import BytesIO
import base64
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import render_template_string


app = Flask(__name__)

df = pd.DataFrame()
X_train, X_test, y_train, y_test = None, None, None, None
xgb_model = XGBClassifier()
xgb_accuracy, svm_accuracy, accuracy, fs_accuracy, ann_accuracy = None, None, None, None, None
precision, xgb_precision, svm_precision, fs_precision, ann_precision = None, None, None, None, None
recall,xgb_recall, svm_recall,fs_recall,ann_recall= None, None, None, None, None
f1,xgb_f1,svm_f1,fs_f1,ann_f1= None, None, None, None, None


label_encoders = {}
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global df
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        try:
            df = pd.read_csv(file)
            table_html = df.head().to_html()
            return render_template('index.html', table=table_html)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

label_encoders = {}
@app.route('/preprocess')
def preprocess():
    global df, label_encoders
    if df.empty:
        return render_template('index.html', message='DataFrame is empty. Upload a file first.')

    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = sklearn.preprocessing.LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    table_html = df.head().to_html()
    return render_template('index.html', table=table_html, message='Preprocessing completed.')

@app.route('/split')
def split():
    global df, X_train, X_test, y_train, y_test
    if df is not None and not df.empty:
        try:
            X = df[['Soil_color', 'pH', 'Rainfall', 'Temperature', 'Crop']]
            Y = df['Fertilizer']
            

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
            message = f'Split completed successfully. Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}'
            return render_template('index.html', message=message)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

    else:
        return render_template('index.html', message='Error: Data not loaded or empty. Please click "Show" first.')


@app.route('/random_forest')
def random_forest():
    global X_train, X_test, y_train, y_test, accuracy, precision,recall,f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')

    try:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        app.config['accuracy'] = accuracy

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'Random Forest Metrics:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        return render_template('random_forest_result.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')



@app.route('/xgboost')
def xgboost():
    global X_train, X_test, y_train, y_test, xgb_accuracy, xgb_precision, xgb_recall, xgb_f1

    
    try:
        # Define preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), ['Soil_color', 'pH', 'Rainfall', 'Temperature', 'Crop'])
                # Add more transformers if needed for categorical features
            ])

        # Create a pipeline
        pipe = make_pipeline(preprocessor, XGBClassifier())

        # Define the hyperparameters grid for tuning
        param_grid = {
            'xgbclassifier__n_estimators': [100, 200, 300],
            'xgbclassifier__learning_rate': [0.1, 0.01, 0.001],
        }

        # Perform grid search cross-validation
        grid_search = GridSearchCV(pipe, param_grid, cv=10, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Access the best parameters and best estimator
        best_estimator = grid_search.best_estimator_
        xgb_model = best_estimator

        y_pred = xgb_model.predict(X_test)

        xgb_accuracy = accuracy_score(y_test, y_pred)
        xgb_precision = precision_score(y_test, y_pred, average='weighted')
        xgb_recall = recall_score(y_test, y_pred, average='weighted')
        xgb_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'XGBoost Metrics:')
        print(f'Accuracy: {xgb_accuracy:.4f}')
        print(f'Precision: {xgb_precision:.4f}')
        print(f'Recall: {xgb_recall:.4f}')
        print(f'F1-Score: {xgb_f1:.4f}')

        return render_template('xgboost_result.html', xgb_accuracy=xgb_accuracy, xgb_precision=xgb_precision, xgb_recall=xgb_recall, xgb_f1=xgb_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')


@app.route('/feature_selection')
def feature_selection():
    global X_train, X_test, y_train, y_test,fs_accuracy,fs_precision,fs_recall,fs_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')

    try:
        k_best = SelectKBest(score_func=chi2, k=2)
        X_train_selected = k_best.fit_transform(X_train, y_train)
        X_test_selected = k_best.transform(X_test)
        clf = SVC(kernel='linear')
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)

        fs_accuracy = accuracy_score(y_test, y_pred)
        fs_precision = precision_score(y_test, y_pred, average='weighted')
        fs_recall = recall_score(y_test, y_pred, average='weighted')
        fs_f1 = f1_score(y_test, y_pred, average='weighted')


        print(f'Feature Selection Metrics:')
        print(f'Accuracy: {fs_accuracy:.4f}')
        print(f'Precision: {fs_precision:.4f}')
        print(f'Recall: {fs_recall:.4f}')
        print(f'F1-Score: {fs_f1:.4f}')

        return render_template('feature_selection_result.html', fs_accuracy=fs_accuracy, fs_precision=fs_precision, fs_recall=fs_recall, fs_f1=fs_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')


@app.route('/svm')
def svm():
    global X_train, X_test, y_train, y_test,svm_accuracy,svm_precision,svm_recall,svm_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')

    try:
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        svm_accuracy = accuracy_score(y_test, y_pred)
        
        svm_precision = precision_score(y_test, y_pred, average='weighted')
        svm_recall = recall_score(y_test, y_pred, average='weighted')
        svm_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'SVM Metrics:')
        print(f'Accuracy: {svm_accuracy:.4f}')
        print(f'Precision: {svm_precision:.4f}')
        
        print(f'Recall: {svm_recall:.4f}')
        print(f'F1-Score: {svm_f1:.4f}')
        return render_template('svm_result.html', svm_accuracy=svm_accuracy, svm_precision=svm_precision, svm_recall=svm_recall, svm_f1=svm_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')

@app.route('/ann')
def ann():
    global X_train, X_test, y_train, y_test,ann_accuracy,ann_precision,ann_recall,ann_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')


    try:
        ann_model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=10,  
            solver='adam',
            activation='relu',
            batch_size=80,
            random_state=42
        )

        ann_model.fit(X_train, y_train)

        y_pred = ann_model.predict(X_test)

        ann_accuracy = accuracy_score(y_test, y_pred)
        app.config['ann_accuracy'] = ann_accuracy
        ann_precision = precision_score(y_test, y_pred, average='weighted')
        ann_recall = recall_score(y_test, y_pred, average='weighted')
        ann_f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'ANN Metrics:')
        print(f'Accuracy: {ann_accuracy:.4f}')
        print(f'Precision: {ann_precision:.4f}')
        print(f'Recall: {ann_recall:.4f}')
        print(f'F1-Score: {ann_f1:.4f}')


        return render_template('ann_result.html', ann_accuracy=ann_accuracy, ann_precision=ann_precision, ann_recall=ann_recall, ann_f1=ann_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')

@app.route('/make_prediction')
def make_prediction():
    return render_template('make_prediction.html')

@app.route('/make_prediction_result', methods=['POST'])
def make_prediction_result():
    global xgb_model, X_train, y_train, label_encoders

    if request.method == 'POST':
        Soil_color = float(request.form['Soil_color'])
        pH = float(request.form['pH'])
        Rainfall = float(request.form['Rainfall'])
        Temperature = float(request.form['Temperature'])
        Crop = float(request.form['Crop'])

        input_values = [[Soil_color, pH, Rainfall, Temperature, Crop]]
        print("Input values:", input_values)

        if xgb_model is None:
            xgb_model = XGBClassifier()

        try:
            input_df = pd.DataFrame(input_values, columns=X_train.columns)
            xgb_model.predict(input_df)
        except (AttributeError, Exception, sklearn.exceptions.NotFittedError) as e:
            xgb_model.fit(X_train, y_train)

        prediction = xgb_model.predict(input_df)

        prediction_fertilizer_name = label_encoders['Fertilizer'].inverse_transform(prediction)

        print("Prediction:", prediction_fertilizer_name)

        return render_template('prediction_result.html', prediction=prediction_fertilizer_name[0])

    return render_template('make_prediction.html')


def generate_accuracy_bar_graph():
    import matplotlib.pyplot as plt
    
    categories = ['XGBooster', 'Random_Forest', 'ann',"svm","FS"]
    values = [xgb_accuracy*100, accuracy * 100, ann_accuracy*100,svm_accuracy*100,fs_accuracy*100]
    plt.bar(categories, values, color='blue')

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Graph Example')
    plt.show()
    


    categories = ['XGBooster', 'Random Forest', 'ANN',"svm","FS"]
    
    values = [xgb_accuracy*100, accuracy*100,ann_accuracy*100,svm_accuracy*100,fs_accuracy*100]

    fig, ax = plt.subplots()
    ax.plot(categories, values, color='blue')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Accuracy Comparison')

    graph_bytes = BytesIO()
    FigureCanvas(fig).print_png(graph_bytes)
    plt.close(fig)

    graph_encoded = base64.b64encode(graph_bytes.getvalue()).decode('utf-8')
    graph_html = f'<img src="data:image/png;base64,{graph_encoded}" alt="Accuracy Graph">'

    x1 = ['XGBooster', 'Random Forest', 'ANN',"svm","FS"]
    precision_values = [xgb_precision*100, precision*100, ann_precision*100,svm_precision*100,fs_precision*100]
    fig, ax = plt.subplots()
    ax.plot(x1, precision_values, '-.', marker='o')
    ax.set_xlabel('ML Algorithms')
    ax.set_ylabel('Precision Values')
    ax.set_title('Precision Values Comparison')

    precision_bytes = BytesIO()
    FigureCanvas(fig).print_png(precision_bytes)
    plt.close(fig)
    
    precision_encoded = base64.b64encode(precision_bytes.getvalue()).decode('utf-8')
    precision_html = f'<img src="data:image/png;base64,{precision_encoded}" alt="Precision Values Graph">'

    
    recall_values = [xgb_recall*100, recall*100, ann_recall*100,svm_recall*100,fs_recall*100]
    fig, ax = plt.subplots()
    ax.plot(x1, recall_values, '--', marker='o', color='green')
    ax.set_xlabel('ML Algorithms')
    ax.set_ylabel('Recall Values')
    ax.set_title('Recall Values Comparison')
    
    recall_bytes = BytesIO()
    FigureCanvas(fig).print_png(recall_bytes)
    plt.close(fig)
    
    recall_encoded = base64.b64encode(recall_bytes.getvalue()).decode('utf-8')
    recall_html = f'<img src="data:image/png;base64,{recall_encoded}" alt="Recall Values Graph">'

    f1_values = [xgb_f1*100, f1*100, ann_f1*100,svm_f1*100,fs_f1*100]
    fig, ax = plt.subplots()
    ax.plot(x1, f1_values, ':', marker='o', color='purple')
    ax.set_xlabel('ML Algorithms')
    ax.set_ylabel('F1-Score Values')
    ax.set_title('F1-Score Values Comparison')
    
    f1_bytes = BytesIO()
    FigureCanvas(fig).print_png(f1_bytes)
    plt.close(fig)
    
    f1_encoded = base64.b64encode(f1_bytes.getvalue()).decode('utf-8')
    f1_html = f'<img src="data:image/png;base64,{f1_encoded}" alt="F1-Score Values Graph">'
    return graph_html, precision_html, recall_html, f1_html

@app.route('/graph', methods=['POST'])
def generate_graph():
    graph_html, precision_plot_html, recall_plot_html, f1_plot_html = generate_accuracy_bar_graph()
    return render_template('index.html', graph=graph_html, precision_plot=precision_plot_html, recall_plot=recall_plot_html, f1_plot=f1_plot_html)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5110, debug=True)