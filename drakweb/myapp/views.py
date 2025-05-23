from django.shortcuts import render
from .models import userdetails
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib
# Create your views here.


def index(request):
    return render(request,'myapp/index.html')

def login(request):
    if request.method == "POST":
        username = request.POST.get('uname')
        password = request.POST.get('pwd')
        print(username, password)

        # Check if username and password match admin credentials
        if username == 'admin' and password == 'admin':
            request.session['uname']='admin'
            content = {
                'data1': 'admin'
            }
            return render(request, 'myapp/homepage.html', content)

        else:
            try:
                # Query the database for user details
                user = userdetails.objects.get(first_name=username, password=password)
                request.session['userid'] = user.id
                request.session['uname'] = user.first_name
                print(user.id)
                content={
                    'data1':user.first_name
                }
                return render(request, 'myapp/homepage.html',content)
            except userdetails.DoesNotExist:
                return render(request, 'myapp/login.html')
    return render(request,'myapp/login.html')

def register(request):
    if request.method == 'POST':
        first_name = request.POST['firstname']
        last_name = request.POST['lastname']
        emailid = request.POST['email']
        mobileno = request.POST['mobno']
        # username = request.POST['uname']
        password = request.POST['pwd']

        newuser = userdetails(first_name=first_name, last_name=last_name, emailid=emailid, password=password,phonenumber=mobileno)
        newuser.save()
        return render(request, "myapp/login.html", {})
    return render(request,'myapp/register.html')

def homepage(request):
    return render(request,'myapp/homepage.html')

def dataupload(request):
    # Load the dataset
    data = pd.read_csv('Top_10_Features_Darknet_With_Label.csv')

    # Split the dataset into features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    content={
        'data1':X_train.shape[0],
        'data2':X_test.shape[0],

    }
    return render(request,'myapp/dataupload.html',content)

def modeltraining(request):
    # Load the dataset
    data = pd.read_csv('Top_10_Features_Darknet_With_Label.csv')

    # Split the dataset into features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Random Forest model (Bagging)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the Random Forest model
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")
    res=accuracy_rf * 100
    content={
        'data':res
    }
    return render(request,'myapp/modeltraining.html',content)

def xgbst(request):
    # Load the dataset
    data = pd.read_csv('Top_10_Features_Darknet_With_Label.csv')

    # Split the dataset into features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train an XGBoost model (Boosting)
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate the XGBoost model
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    res = accuracy_xgb * 100
    content = {
        'data': res
    }
    return render(request,'myapp/xgbst.html',content)


def load_model(model_name):
    # Load the specified model and the scaler
    model = joblib.load(f'{model_name}.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def predict(model_name, features):
    model, scaler = load_model(model_name)

    # Ensure the features are in the correct shape
    features = np.array(features).reshape(1, -1)

    # Standardize the features
    features = scaler.transform(features)

    # Predict using the specified model
    prediction = model.predict(features)

    # Return the prediction
    return prediction[0]

def predictdata(request):
    # Example feature values for prediction
    # Replace these values with actual feature values
    if request.method=='POST':
        fwdbytes=int(request.POST['fwdbytes'])
        fwdmin=int(request.POST['fwdmin'])
        idlemax=int(request.POST['idlemax'])
        bwdmin=int(request.POST['bwdmin'])
        idlemean=int(request.POST['idlemean'])
        idlemin=int(request.POST['idlemin'])
        bwdbytes=int(request.POST['bwdbytes'])
        pktlenmin=int(request.POST['pktlenmin'])
        pktlenmax=int(request.POST['pktlenmax'])
        flowmin=int(request.POST['flowmin'])


        feature_names = [
            "FWD Init Win Bytes", "Fwd Seg Size Min", "Idle Max",
            "Bwd Packet Length Min", "Idle Mean", "Idle Min",
            "Bwd Init Win Bytes", "Packet Length Min",
            "Packet Length Max", "Flow IAT Min"
        ]
        # feature_values = [
        #     5000, 40, 1000000, 1, 500000, 0,
        #     3000, 20, 1500, 100
        # ]

        feature_values = [
                fwdbytes, fwdmin, idlemax, bwdmin, idlemean,idlemin,
                bwdbytes, pktlenmin, pktlenmax, flowmin
            ]
        print("Feature Names and Values:")
        for name, value in zip(feature_names, feature_values):
            print(f"{name}: {value}")

        model_name = 'rf_model'  # or 'xgb_model'

        prediction = predict(model_name, feature_values)
        print(f"\nThe predicted label using {model_name} is: {prediction}")
        res= "The predicted label using "+ model_name+ "is: "+ str(prediction)
        content={
            'data':res
        }

        return render(request, 'myapp/predictdata.html',content)
    return render(request,'myapp/predictdata.html')