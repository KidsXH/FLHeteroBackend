from django.urls import path
from . import views

app_name = 'FLHetero'
urlpatterns = [
    path('', views.hello),
    path('datasets/', views.datasets),
    path('client/', views.client),
    path('weights/', views.weights),
    path('sampling/', views.sampling),
    path('pca/', views.pca),
    path('labels/', views.labels),
    path('cluster/', views.cluster),
    path('cpca/', views.cpca),
    path('annotation/', views.annotation),
    path('annotationList/', views.annotation_list),
]
