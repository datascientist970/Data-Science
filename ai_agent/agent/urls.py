from django.urls import path
from . import views
urlpatterns = [
    path('',views.home,name='home'),
    path('agent',views.agent,name='agent'),
    # path('cancer',views.cancer_prediction,name='cancer'),
    # path('heart/', views.heart_disease_prediction, name='heart'),
    # path('diabetes/', views.diabetes_prediction, name='diabetes'),
]
