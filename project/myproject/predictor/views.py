from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
import pickle
import io
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import pandas as pd
import pickle
import io

# Load the trained model
with open('predictor/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

expected_features = model.feature_names_in_.tolist()

class PredictCSVView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES.get('file', None)

        if not file_obj:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Read the uploaded file into a DataFrame
            csv_data = file_obj.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))

            # Preprocess data
            df = df.drop(columns=['label', 'action_no'], errors='ignore')

            # Identify non-numeric columns
            non_numeric_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Apply the same encoding used during training
            df = pd.get_dummies(df, columns=non_numeric_columns)

            # Align the DataFrame with the training features
            df = df.reindex(columns=expected_features, fill_value=0)

            # Make predictions
            predictions = model.predict(df)
            predictions = predictions.tolist()

            return Response({'predictions': predictions}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def upload_and_predict(request):
    prediction = None
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded file
            uploaded_file = request.FILES['file']
            csv_data = uploaded_file.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))

            # Preprocess data as needed
            df = df.drop(columns=['label', 'action_no'], errors='ignore')

            # Ensure the DataFrame has the same columns as the model expects
            expected_features = model.feature_names_in_.tolist()
            df = df.reindex(columns=expected_features, fill_value=0)

            # Make predictions
            predictions = model.predict(df)

            prediction = predictions.tolist()
    else:
        form = UploadFileForm()

    return render(request, 'predictor/upload_and_predict.html', {'form': form, 'prediction': prediction})

