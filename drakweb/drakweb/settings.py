# drakweb/settings.py

from pathlib import Path
import os # Import os module for path operations

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key-here' # KEEP THIS SECRET IN PRODUCTION
DEBUG = True # Set to False in production
ALLOWED_HOSTS = [] # Add your domain names here in production, e.g., ['yourdomain.com', 'www.yourdomain.com']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',  # Your application name
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'drakweb.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # You can add project-wide template directories here if needed
        'APP_DIRS': True,  # This is crucial for Django to find templates inside each app's 'templates' folder
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'drakweb.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(BASE_DIR / 'db.sqlite3'),
    }
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files configuration
STATIC_URL = '/static/'
# This defines where Django will look for static files when you run `collectstatic`
# and helps the development server locate them.
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Add your Gemini API Key here (for development purposes)
# For production, consider using environment variables:
# GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'default-fallback-key-if-not-set')
GEMINI_API_KEY = 'AIzaSyBGyYz72RjipMSLkq5KoHupM02-X5ZPeVY' # Your actual key

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'