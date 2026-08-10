from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.history, name='history'),
    path('history/<int:pk>/', views.history_detail, name='history_detail'),
    path('about/', views.about, name='about'),
]
