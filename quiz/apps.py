from django.apps import AppConfig


class QuizConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'quiz'
    verbose_name = 'Quiz Generator'

    def ready(self):
        # Import signals to register them
        import quiz.signals
