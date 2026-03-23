from django.urls import path
from .views import pm_view

urlpatterns = [
	path('', pm_view, name='pm_view'),
]