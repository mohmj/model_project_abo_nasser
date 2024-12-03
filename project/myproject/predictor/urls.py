from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_predict, name='upload_and_predict'),
    path('api/predict/', views.PredictCSVView.as_view(), name='api_predict_csv'),
]

