"""
Django models for Quiz Generation System
"""
from django.db import models
from django.contrib.auth.models import User
import os


class UserProfile(models.Model):
    """Extended user profile with additional fields"""
    STATUS_CHOICES = [
        ('Waiting', 'Waiting'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    name = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Waiting')
    profile_image = models.ImageField(
        upload_to='profile_images/',
        default='profile_images/default.png',
        blank=True,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.status}"
    
    @property
    def has_custom_image(self):
        """Check if user has uploaded a custom profile image"""
        return bool(self.profile_image and self.profile_image.name != 'profile_images/default.png')


class PDFDocument(models.Model):
    """Model for storing uploaded PDF documents"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='pdfs')
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    text_content = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.title
    
    def filename(self):
        return os.path.basename(self.file.name)
    
    class Meta:
        ordering = ['-uploaded_at']


class Quiz(models.Model):
    """Model for storing generated quizzes"""
    DIFFICULTY_CHOICES = [
        ('Easy', 'Easy'),
        ('Medium', 'Medium'),
        ('Hard', 'Hard'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='quizzes')
    pdf = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='quizzes')
    subject = models.CharField(max_length=255)
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES)
    number_of_questions = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_completed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.subject} - {self.difficulty} - {self.number_of_questions} Questions"
    
    class Meta:
        ordering = ['-created_at']


class Question(models.Model):
    """Model for storing quiz questions"""
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    option_a = models.CharField(max_length=500)
    option_b = models.CharField(max_length=500)
    option_c = models.CharField(max_length=500)
    option_d = models.CharField(max_length=500)
    correct_answer = models.CharField(max_length=1, choices=[
        ('A', 'A'),
        ('B', 'B'),
        ('C', 'C'),
        ('D', 'D'),
    ])
    explanation = models.TextField(blank=True, null=True)
    order = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Q{self.order}: {self.question_text[:50]}..."
    
    class Meta:
        ordering = ['order']


class UserAnswer(models.Model):
    """Model for storing user answers to quiz questions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='answers')
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='user_answers')
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='user_answers')
    selected_answer = models.CharField(max_length=1, choices=[
        ('A', 'A'),
        ('B', 'B'),
        ('C', 'C'),
        ('D', 'D'),
    ])
    is_correct = models.BooleanField()
    answered_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - Q{self.question.order}: {self.selected_answer}"
    
    class Meta:
        unique_together = ['user', 'question']
        ordering = ['-answered_at']


class Result(models.Model):
    """Model for storing quiz results with AI-generated feedback"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='results')
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='results')
    score = models.IntegerField()
    total_questions = models.IntegerField()
    correct_answers = models.IntegerField()
    wrong_answers = models.IntegerField()
    percentage = models.FloatField()
    
    # AI-generated feedback
    strength_analysis = models.TextField(blank=True, null=True)
    weakness_analysis = models.TextField(blank=True, null=True)
    suggestions = models.TextField(blank=True, null=True)
    
    completed_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.quiz.subject} - {self.percentage}%"
    
    class Meta:
        ordering = ['-completed_at']
