# myapp/views.py

from django.shortcuts import render
from django.http import JsonResponse # Import JsonResponse for API proxy
from django.views.decorators.csrf import csrf_exempt # For simplicity in development, consider more robust CSRF handling for production
from django.conf import settings # Import settings to access GEMINI_API_KEY
import requests # Import requests library for making HTTP calls
import json # Import json for handling JSON data

from .models import userdetails
import pandas as pd
import os
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
# joblib is already imported above


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
    csv_path = os.path.join(settings.BASE_DIR, 'Top_10_Features_Darknet_With_Label.csv')
    data = pd.read_csv(csv_path)

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
    csv_path = os.path.join(settings.BASE_DIR, 'Top_10_Features_Darknet_With_Label.csv')
    data = pd.read_csv(csv_path)

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
    csv_path = os.path.join(settings.BASE_DIR, 'Top_10_Features_Darknet_With_Label.csv')
    data = pd.read_csv(csv_path)

    # Split the dataset into features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']

    # FIX: Corrected typo from train_train_test_split to train_test_split
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
    # Ensure 'scaler.pkl' and model_name.pkl are accessible
    # You might need to adjust paths if they are not in the same directory as views.py
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

@csrf_exempt # Use this decorator for simplicity in development to allow POST requests without CSRF token
def gemini_proxy_api(request):
    """
    Proxies requests from the frontend to the Google Gemini API.
    This helps bypass CORS issues and keeps the API key secure on the server.
    """
    if request.method == 'POST':
        try:
            # Parse the incoming JSON data from the frontend
            data = json.loads(request.body)
            prompt = data.get('prompt')
            response_schema = data.get('responseSchema')

            if not prompt or not response_schema:
                return JsonResponse({'error': 'Missing prompt or responseSchema in request body'}, status=400)

            # Construct the payload for the Gemini API
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": response_schema
                }
            }

            # Get API key from Django settings
            gemini_api_key = settings.GEMINI_API_KEY
            gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"

            # Make the request to the actual Gemini API
            headers = {'Content-Type': 'application/json'}
            gemini_response = requests.post(gemini_api_url, headers=headers, data=json.dumps(payload))
            gemini_response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            # Return the Gemini API's response directly to the frontend
            return JsonResponse(gemini_response.json())

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
        except requests.exceptions.RequestException as e:
            # Log the full error for debugging on the server side
            print(f"Error calling Gemini API: {e}")
            return JsonResponse({'error': f'Failed to connect to Gemini API: {e}'}, status=500)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return JsonResponse({'error': f'An unexpected error occurred: {e}'}, status=500)
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


def predictdata(request):
    # This view is for your server-side machine learning model prediction.
    # The client-side JavaScript in predictdata.html makes its own API calls.
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