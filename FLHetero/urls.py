from django.urls import path
from . import views

app_name = 'FLHetero'
urlpatterns = [
    path('', views.hello),
    path('initialize/', views.initialize),
    path('identify/', views.identify),
    path('customize/', views.customize),
]
