from django.urls import path
from . import views
 
urlpatterns = [
path('', views.home),
path('newpage/',  views.new_page,  name="my_function")
]