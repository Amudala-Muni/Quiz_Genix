"""
Django signals for automatic UserProfile creation
"""
from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import UserProfile


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    Automatically create UserProfile when a new User is created.
    This ensures that every user has a profile object.
    """
    if created:
        UserProfile.objects.get_or_create(
            user=instance,
            defaults={
                'name': instance.get_full_name() or instance.username,
                'status': 'Approved'
            }
        )


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """
    Ensure UserProfile is saved whenever User is saved.
    """
    try:
        instance.profile.save()
    except UserProfile.DoesNotExist:
        # Create profile if it doesn't exist
        UserProfile.objects.get_or_create(
            user=instance,
            defaults={
                'name': instance.get_full_name() or instance.username,
                'status': 'Approved'
            }
        )
