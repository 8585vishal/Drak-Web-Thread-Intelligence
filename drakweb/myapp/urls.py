# drakweb/urls.py

"""
URL configuration for drakweb project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include # Ensure 'include' is imported
from django.conf import settings # Import settings
from django.conf.urls.static import static # Import static function

# Assuming 'views' here refers to your myapp's views due to your project structure.
# A more standard Django approach would be:
# from myapp import views
# or
# path('', include('myapp.urls')),
# but given your existing structure, this might be how it's set up.
from myapp import views # Explicitly import views from myapp

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', views.index, name='index'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('homepage', views.homepage, name='homepage'),
    path('dataupload', views.dataupload, name='dataupload'),
    path('modeltraining', views.modeltraining, name='modeltraining'),
    path('predictdata', views.predictdata, name='predictdata'),
    path('xgbst', views.xgbst, name='xgbst'),

    # New URL for the Gemini API proxy
    path('api/gemini-proxy/', views.gemini_proxy_api, name='gemini_proxy_api'),
]

# Serve static files during development (IMPORTANT for {% load static %})
# This block is essential for Django to serve static files like CSS/JS in development.
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # If you have media files uploaded by users, you might also need:
    # urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)