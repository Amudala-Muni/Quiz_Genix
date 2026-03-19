"""
Django Forms for Quiz Generation System
"""
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile


class UserRegistrationForm(forms.ModelForm):
    """
    User registration form with profile image upload support.
    Validates:
    - File extension (.jpg, .jpeg, .png only)
    - File size (max 2MB)
    """
    name = forms.CharField(
        max_length=255,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name'
        })
    )
    username = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Choose a username'
        })
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email'
        })
    )
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Create a password'
        })
    )
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })
    )
    profile_image = forms.ImageField(
        required=False,
        allow_empty_file=True,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/jpg,image/png'
        })
    )

    class Meta:
        model = User
        fields = ['username', 'email']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError('Username already exists')
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Email already exists')
        return email

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get('password1')
        password2 = cleaned_data.get('password2')

        if password1 and password2:
            if password1 != password2:
                raise forms.ValidationError('Passwords do not match')

        return cleaned_data

    def clean_profile_image(self):
        """
        Validate profile image:
        - Only allow .jpg, .jpeg, .png extensions
        - Maximum file size: 2MB
        """
        profile_image = self.cleaned_data.get('profile_image')

        if profile_image and profile_image != 'default.png':
            # Validate file extension
            allowed_extensions = ['.jpg', '.jpeg', '.png']
            import os
            ext = os.path.splitext(profile_image.name)[1].lower()

            if ext not in allowed_extensions:
                raise forms.ValidationError(
                    'Invalid file format. Only .jpg, .jpeg, and .png files are allowed.'
                )

            # Validate file size (2MB = 2 * 1024 * 1024 bytes)
            max_size = 2 * 1024 * 1024
            if profile_image.size > max_size:
                raise forms.ValidationError(
                    'File size exceeds 2MB limit. Please upload a smaller image.'
                )

        return profile_image


class UserProfileForm(forms.ModelForm):
    """
    Form for updating user profile information including profile image.
    """
    class Meta:
        model = UserProfile
        fields = ['name', 'profile_image']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your full name'
            }),
            'profile_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/jpeg,image/jpg,image/png'
            })
        }

    def clean_profile_image(self):
        """
        Validate profile image:
        - Only allow .jpg, .jpeg, .png extensions
        - Maximum file size: 2MB
        """
        profile_image = self.cleaned_data.get('profile_image')

        if profile_image:
            # Validate file extension
            allowed_extensions = ['.jpg', '.jpeg', '.png']
            import os
            ext = os.path.splitext(profile_image.name)[1].lower()

            if ext not in allowed_extensions:
                raise forms.ValidationError(
                    'Invalid file format. Only .jpg, .jpeg, and .png files are allowed.'
                )

            # Validate file size (2MB = 2 * 1024 * 1024 bytes)
            max_size = 2 * 1024 * 1024
            if profile_image.size > max_size:
                raise forms.ValidationError(
                    'File size exceeds 2MB limit. Please upload a smaller image.'
                )

        return profile_image
