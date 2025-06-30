from django.http import JsonResponse
from django.shortcuts import render
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import os
import pickle
import tensorflow as tf 
import numpy as np

# Model setup
os.environ["GOOGLE_API_KEY"] = 'your api key'
# heart_model_path = 'models/heart.pkl'
# cancer_model_path = 'models/cancer.pkl'
# diabetics='models/diabetics_model.h5'
# Load models
# try:
#     with open(heart_model_path, 'rb') as f:
#         heart_model = pickle.load(f)
# except:
#     heart_model = None

# try:
#     with open(cancer_model_path, 'rb') as f:
#         cancer_model = pickle.load(f)
# except:
#     cancer_model = None

# try:
#     with open(diabetics, 'rb') as f:
#         diabetics = tf.keras.models.load_model(f)
# except:
#     diabetics = None

# Load AI agent components
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
prompt = hub.pull('hwchase17/react')
search = DuckDuckGoSearchRun()
agent = create_react_agent(llm=model, tools=[search], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[search], verbose=True, handle_parsing_errors=True)

def home(request):
    return render(request, 'dashboard.html')

def agent(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        user_prompt = request.POST.get("prompt", "").strip().lower()
        try:
            if any(phrase in user_prompt for phrase in ["who are you",'who is your creator','who is your developer','who buid you',"what are you", "introduce yourself", "your identity",'identify yourself']):
                return JsonResponse({
                    "response": "I am an AI agent developed by Muhammad Dawood for personal use."
                })
            elif user_prompt =='are google developed':
                return JsonResponse({
                    "response": "No, I am an AI agent developed by Muhammad Dawood for personal use."
                })
            elif user_prompt =='are ai agent':
                return JsonResponse({
                    "response": "Yes, I am an AI agent developed by Muhammad Dawood for personal use."
                })
            result = executor.invoke({"input": user_prompt})
            return JsonResponse({"response": result["output"]})
        except Exception as e:
            return JsonResponse({"response": f"Error: {str(e)}"})
    return render(request, 'agent.html')

# def heart_disease_prediction(request):
#     if request.method == 'POST':
#         try:
#             # Get form data with default values
#             age = float(request.POST.get('age', 0))
#             sex = float(request.POST.get('sex', 0))
#             cp = float(request.POST.get('cp', 0))
#             trestbps = float(request.POST.get('trestbps', 0))
#             chol = float(request.POST.get('chol', 0))
#             fbs = float(request.POST.get('fbs', 0))
#             restecg = float(request.POST.get('restecg', 0))
#             thalach = float(request.POST.get('thalach', 0))
#             exang = float(request.POST.get('exang', 0))
#             oldpeak = float(request.POST.get('oldpeak', 0))
#             slope = float(request.POST.get('slope', 0))
#             ca = float(request.POST.get('ca', 0))
#             thal = float(request.POST.get('thal', 0))
            
#             # Validate that required fields are not empty
#             required_fields = {
#                 'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
#                 'chol': chol, 'fbs': fbs, 'restecg': restecg,
#                 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
#                 'slope': slope, 'ca': ca, 'thal': thal
#             }
            
#             for field_name, value in required_fields.items():
#                 if value is None:
#                     return render(request, 'heart.html', {
#                         'error': f"Please provide a value for {field_name.replace('_', ' ')}"
#                     })

#             # Create input array
#             input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
#                                   thalach, exang, oldpeak, slope, ca, thal]])
            
#             # Make prediction
#             prediction = heart_model.predict(input_data)
#             probability = heart_model.predict_proba(input_data)
            
#             # Prepare result
#             result = "Positive" if prediction[0] == 1 else "Negative"
#             confidence = round(max(probability[0]) * 100, 2)
            
#             return render(request, 'heart_result.html', {
#                 'result': result,
#                 'confidence': confidence,
#                 'input_data': {
#                     'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
#                     'chol': chol, 'fbs': fbs, 'restecg': restecg,
#                     'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
#                     'slope': slope, 'ca': ca, 'thal': thal
#                 }
#             })
#         except Exception as e:
#             return render(request, 'heart.html', {'error': f"An error occurred: {str(e)}"})
#     return render(request, 'heart.html')

# def cancer_prediction(request):
#     if request.method == 'POST':
#         try:
#             # Get form data with default values
#             radius_mean = float(request.POST.get('radius_mean', 0))
#             texture_mean = float(request.POST.get('texture_mean', 0))
#             perimeter_mean = float(request.POST.get('perimeter_mean', 0))
#             area_mean = float(request.POST.get('area_mean', 0))
#             smoothness_mean = float(request.POST.get('smoothness_mean', 0))
            
#             # Validate that required fields are not empty
#             required_fields = {
#                 'radius_mean': radius_mean,
#                 'texture_mean': texture_mean,
#                 'perimeter_mean': perimeter_mean,
#                 'area_mean': area_mean,
#                 'smoothness_mean': smoothness_mean
#             }
            
#             for field_name, value in required_fields.items():
#                 if value is None:
#                     return render(request, 'cancer.html', {
#                         'error': f"Please provide a value for {field_name.replace('_', ' ')}"
#                     })

#             # Create input array
#             input_data = np.array([[radius_mean, texture_mean, perimeter_mean, 
#                                   area_mean, smoothness_mean]])
            
#             # Make prediction
#             prediction = cancer_model.predict(input_data)
#             probability = cancer_model.predict_proba(input_data)
            
#             # Prepare result
#             result = "Malignant" if prediction[0] == 1 else "Benign"
#             confidence = round(max(probability[0]) * 100, 2)
            
#             return render(request, 'cancer_result.html', {
#                 'result': result,
#                 'confidence': confidence,
#                 'input_data': {
#                     'radius_mean': radius_mean,
#                     'texture_mean': texture_mean,
#                     'perimeter_mean': perimeter_mean,
#                     'area_mean': area_mean,
#                     'smoothness_mean': smoothness_mean
#                 }
#             })
#         except Exception as e:
#             return render(request, 'cancer.html', {'error': f"An error occurred: {str(e)}"})
#     return render(request, 'cancer.html')

# def diabetes_prediction(request):
#     if request.method == 'POST':
#         try:
#             # Get form data with default values
#             pregnancies = float(request.POST.get('pregnancies', 0))
#             glucose = float(request.POST.get('glucose', 0))
#             blood_pressure = float(request.POST.get('blood_pressure', 0))
#             skin_thickness = float(request.POST.get('skin_thickness', 0))
#             insulin = float(request.POST.get('insulin', 0))
#             bmi = float(request.POST.get('bmi', 0))
#             diabetes_pedigree = float(request.POST.get('diabetes_pedigree', 0))
#             age = float(request.POST.get('age', 0))
            
#             # Validate that required fields are not empty
#             required_fields = {
#                 'pregnancies': pregnancies,
#                 'glucose': glucose,
#                 'blood_pressure': blood_pressure,
#                 'skin_thickness': skin_thickness,
#                 'insulin': insulin,
#                 'bmi': bmi,
#                 'diabetes_pedigree': diabetes_pedigree,
#                 'age': age
#             }
            
#             for field_name, value in required_fields.items():
#                 if value is None:
#                     return render(request, 'diabetes.html', {
#                         'error': f"Please provide a value for {field_name.replace('_', ' ')}"
#                     })

#             # Create input array and normalize if your model expects normalized data
#             input_data = np.array([[pregnancies, glucose, blood_pressure, 
#                                   skin_thickness, insulin, bmi, 
#                                   diabetes_pedigree, age]])
            
#             # Make prediction
#             prediction_prob = diabetics.predict(input_data)
#             prediction = (prediction_prob > 0.5).astype(int)
            
#             # Prepare result
#             result = "Positive" if prediction[0] == 1 else "Negative"
#             confidence = round(float(prediction_prob[0][0]) * 100, 2)
            
#             return render(request, 'diabetes_result.html', {
#                 'result': result,
#                 'confidence': confidence,
#                 'input_data': {
#                     'pregnancies': pregnancies,
#                     'glucose': glucose,
#                     'blood_pressure': blood_pressure,
#                     'skin_thickness': skin_thickness,
#                     'insulin': insulin,
#                     'bmi': bmi,
#                     'diabetes_pedigree': diabetes_pedigree,
#                     'age': age
#                 }
#             })
#         except Exception as e:
#             return render(request, 'diabetes.html', {'error': f"An error occurred: {str(e)}"})
#     return render(request, 'diabetes.html')
