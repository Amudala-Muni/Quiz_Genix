"""
URL Configuration for Quiz Application
"""
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Landing page (Home)
    path('', views.landing_page, name='landing'),
    path('home/', views.landing_page, name='home'),
    
    # AI Service Test
    path('test-ai/', views.test_ai_service, name='test_ai'),
    
    # Static pages
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    
    # Authentication
    path('login/', views.user_login, name='login'),
    path('user/login/', views.user_login, name='user_login'),
    path('user/logout/', views.user_logout, name='user_logout'),
    path('logout/', auth_views.LogoutView.as_view(next_page='landing'), name='logout'),
    path('register/', views.register, name='register'),
    
    # Main views
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('pdf/delete/<int:pdf_id>/', views.delete_pdf, name='delete_pdf'),
    
    # Quiz management - Topic based
    path('quiz/create/topic/', views.create_quiz_from_topic, name='create_quiz_topic'),
    
    # Quiz management - PDF based
    path('quiz/create/<int:pdf_id>/', views.create_quiz, name='create_quiz'),
    path('quiz/<int:quiz_id>/', views.take_quiz, name='take_quiz'),
    path('quiz/<int:quiz_id>/submit/', views.submit_quiz, name='submit_quiz'),
    path('quiz/<int:quiz_id>/result/', views.quiz_result, name='quiz_result'),
    path('quiz/<int:quiz_id>/detail/', views.quiz_detail, name='quiz_detail'),
    path('quiz/delete/<int:quiz_id>/', views.delete_quiz, name='delete_quiz'),
    
    # AJAX endpoints
    path('ajax/generate-quiz/', views.generate_quiz_ajax, name='generate_quiz_ajax'),
    
    # Admin views
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin_logout/', views.admin_logout, name='admin_logout'),
    
    # Custom approve/reject user URLs (simple pattern)
    path('approve_user/<int:user_id>/', views.approve_user, name='approve_user'),
    path('reject_user/<int:user_id>/', views.reject_user, name='reject_user'),
    
    # Legacy admin URLs (for backward compatibility)
    path('admin/user/<int:user_id>/approve/', views.approve_user, name='approve_user_legacy'),
    path('admin/user/<int:user_id>/reject/', views.reject_user, name='reject_user_legacy'),
    path('admin/user/<int:user_id>/toggle/', views.toggle_user_status, name='toggle_user_status'),
]
