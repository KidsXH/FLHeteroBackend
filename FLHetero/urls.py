from django.urls import path
from . import views

app_name = 'FLHetero'
urlpatterns = [
    path('', views.hello),
    path('datasets/', views.datasets),
    path('client/', views.client),
    path('weights/', views.weights),
    path('sampling/', views.sampling),
    path('cpca/all/', views.cpca_all),
    path('cluster/', views.cluster),
    path('cpca/cluster/', views.cpca_cluster),
    path('annotation/', views.annotation),
    path('annotationList/', views.annotation_list),
]
