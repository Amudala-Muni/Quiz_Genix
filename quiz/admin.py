"""
Django Admin Configuration for Quiz Application
"""
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import PDFDocument, Quiz, Question, UserAnswer


class UserAdmin(BaseUserAdmin):
    """Custom User Admin with is_active toggle"""
    list_display = ('username', 'email', 'is_active', 'is_staff', 'date_joined')
    list_filter = ('is_active', 'is_staff', 'is_superuser', 'date_joined')
    actions = ['activate_users', 'deactivate_users']
    
    # Override add_fieldsets to include is_active in user creation
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2', 'is_active'),
        }),
    )
    
    def activate_users(self, request, queryset):
        """Activate selected users"""
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} user(s) activated successfully.')
    activate_users.short_description = 'Activate selected users'
    
    def deactivate_users(self, request, queryset):
        """Deactivate selected users"""
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} user(s) deactivated successfully.')
    deactivate_users.short_description = 'Deactivate selected users'


# Re-register User with custom admin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)


@admin.register(PDFDocument)
class PDFDocumentAdmin(admin.ModelAdmin):
    """Admin for PDF Documents"""
    list_display = ('title', 'user', 'uploaded_at')
    list_filter = ('uploaded_at', 'user')
    search_fields = ('title', 'user__username')
    readonly_fields = ('uploaded_at',)


@admin.register(Quiz)
class QuizAdmin(admin.ModelAdmin):
    """Admin for Quizzes"""
    list_display = ('subject', 'user', 'difficulty', 'number_of_questions', 'is_completed', 'created_at')
    list_filter = ('difficulty', 'is_completed', 'created_at')
    search_fields = ('subject', 'user__username')
    readonly_fields = ('created_at',)


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    """Admin for Questions"""
    list_display = ('quiz', 'order', 'question_text', 'correct_answer')
    list_filter = ('quiz',)
    search_fields = ('question_text',)


@admin.register(UserAnswer)
class UserAnswerAdmin(admin.ModelAdmin):
    """Admin for User Answers"""
    list_display = ('user', 'quiz', 'question', 'selected_answer', 'is_correct', 'answered_at')
    list_filter = ('is_correct', 'answered_at')
    search_fields = ('user__username',)
    readonly_fields = ('answered_at',)
