from django.conf.urls import url

from . import views

urlpatterns = [
	
	url(r'^get_prediction/$', views.get_prediction, name='get_prediction'),
    url(r'^$', views.homepage, name='homepage'),
]
