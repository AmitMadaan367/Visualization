"""visual URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from visualapp import views
from django.urls import path


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index, name='index'),
    path('userdash/',views.dashboard, name='userdash'),
    path('updateboard/',views.updateboard, name='updateboard'),
    path('signup/',views.signup_page, name='signup'),
    path('signin/',views.login_user, name='signin'),
    path('logout/',views.user_logout, name='user_logout'),
]
