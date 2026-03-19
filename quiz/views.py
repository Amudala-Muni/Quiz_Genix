"""
Django Views for Quiz Generation System
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.contrib.auth import authenticate, login, logout
from django.db.models.signals import post_save
from django.dispatch import receiver
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .models import PDFDocument, Quiz, Question, UserAnswer, Result, UserProfile
from .pdf_utils import extract_text_from_pdf
from .ai_services import (
    get_llm,
    build_quiz_prompt,
    parse_questions,
    build_feedback_prompt,
    parse_feedback,
    APIKeyError, 
    test_ai_connection
)
import logging

logger = logging.getLogger(__name__)

# Thread pool for running async AI calls - isolated from Django's request threads
_ai_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quiz_ai_")


def shutdown_ai_executor():
    """
    Gracefully shutdown the AI thread pool executor.
    Call this during Django application cleanup.
    """
    global _ai_executor
    if _ai_executor is not None:
        _ai_executor.shutdown(wait=True)
        _ai_executor = None


def _run_ai_async(coro):
    """
    Run an async coroutine in an isolated thread with its own event loop.
    This avoids event loop issues in Django's request processing threads.
    
    Args:
        coro: An async coroutine to execute.
    
    Returns:
        The result of the coroutine execution.
    """
    def _run_in_new_loop():
        # Create a completely new event loop for this thread
        # This ensures we don't interfere with any existing loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    future = _ai_executor.submit(_run_in_new_loop)
    return future.result()


def generate_quiz_questions_safe(pdf_text: str, subject: str, difficulty: str, num_questions: int):
    """
    Thread-safe wrapper for generating quiz questions.
    Safely runs the async AI call in an isolated thread.
    
    Args:
        pdf_text: Text content from the PDF or topic description
        subject: Subject/topic for the quiz
        difficulty: Difficulty level (Easy, Medium, Hard)
        num_questions: Number of questions to generate
    
    Returns:
        List of question dictionaries
    """
    async def _generate():
        llm = get_llm()
        prompt = build_quiz_prompt(pdf_text, subject, difficulty, num_questions)
        response = await llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        return parse_questions(response_text)
    
    return _run_ai_async(_generate())


def generate_performance_feedback_safe(subject: str, total: int, correct: int, 
                                       wrong: int, user_answers: list, difficulty: str):
    """
    Thread-safe wrapper for generating performance feedback.
    Safely runs the async AI call in an isolated thread.
    
    Args:
        subject: Subject of the quiz
        total: Total number of questions
        correct: Number of correct answers
        wrong: Number of wrong answers
        user_answers: List of user answer dictionaries
        difficulty: Difficulty level of the quiz
    
    Returns:
        Dictionary with feedback analysis
    """
    async def _generate():
        llm = get_llm()
        prompt = build_feedback_prompt(subject, total, correct, wrong, user_answers, difficulty)
        response = await llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        return parse_feedback(response_text)
    
    return _run_ai_async(_generate())


# Signal to create UserProfile when User is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)


def landing_page(request):
    """Modern AI-themed landing page for QuizGenix"""
    if request.user.is_authenticated:
        if request.user.is_staff or request.user.is_superuser:
            return redirect('admin_dashboard')
        return redirect('dashboard')
    return render(request, 'quiz/home.html')


def test_ai_service(request):
    """
    Test view to verify AI service configuration.
    Access this at /test-ai/ to check if Gemini API is working.
    """
    result = test_ai_connection()
    
    if request.headers.get('Accept') == 'application/json' or request.GET.get('format') == 'json':
        return JsonResponse(result)
    
    if result['configured']:
        messages.success(request, result['message'])
    else:
        messages.error(request, result['message'])
    
    return redirect('landing_page')


def about(request):
    """About page view"""
    return render(request, 'quiz/pages/about.html')


def contact(request):
    """Contact page view with form handling"""
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        
        # Validate form data
        if name and email and message:
            # Here you would typically send an email or save to database
            # For now, we'll just show a success message
            messages.success(request, f'Thank you, {name}! Your message has been sent successfully. We will get back to you at {email} soon.')
            return redirect('contact')
        else:
            messages.error(request, 'Please fill in all fields.')
    
    return render(request, 'quiz/pages/contact.html')


def user_login(request):
    """
    Custom login view for normal users only.
    - Allows ONLY normal users (is_staff=False AND is_superuser=False)
    - Rejects admin users with error message
    """
    if request.user.is_authenticated:
        if request.user.is_staff or request.user.is_superuser:
            return redirect('admin_dashboard')
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if not username or not password:
            messages.error(request, 'Please provide both username and password')
            return redirect('user_login')
        
        # Check if user exists first
        try:
            user = User.objects.get(username=username)
            
            # Check if user is admin - reject admins from user login
            if user.is_staff or user.is_superuser:
                messages.error(request, 'Admins must use Admin Login page.')
                return redirect('user_login')
            
            # Check if user is active
            if not user.is_active:
                messages.error(request, 'Your account is not active. Please contact admin.')
                return redirect('user_login')
                
        except User.DoesNotExist:
            # User doesn't exist - will fail authentication anyway
            pass
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Double-check user is not admin (defense in depth)
            if user.is_staff or user.is_superuser:
                messages.error(request, 'Admins must use Admin Login page.')
                return redirect('user_login')
            
            # Login successful - redirect to user dashboard
            login(request, user)
            next_url = request.GET.get('next')
            return redirect(next_url if next_url else 'dashboard')
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('user_login')
    
    return render(request, 'quiz/auth/login.html')


def admin_login(request):
    """
    Admin login view - strictly for admin users only.
    - Allows ONLY admin users (is_staff=True OR is_superuser=True)
    - Rejects normal users with error message
    """
    if request.user.is_authenticated:
        if request.user.is_staff or request.user.is_superuser:
            return redirect('admin_dashboard')
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if not username or not password:
            messages.error(request, 'Please provide both username and password')
            return redirect('admin_login')
        
        # Check if user exists first
        try:
            user = User.objects.get(username=username)
            
            # Check if user is active
            if not user.is_active:
                messages.error(request, 'Your account is not active. Please contact admin.')
                return redirect('admin_login')
            
            # Check if user is NOT admin - reject normal users
            if not user.is_staff and not user.is_superuser:
                messages.error(request, 'Access restricted to administrators.')
                return redirect('admin_login')
                
        except User.DoesNotExist:
            # User doesn't exist - will fail authentication anyway
            pass
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Verify user is admin (defense in depth)
            if not user.is_staff and not user.is_superuser:
                messages.error(request, 'Access restricted to administrators.')
                return redirect('admin_login')
            
            # Login successful - redirect to admin dashboard
            login(request, user)
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('admin_login')
    
    return render(request, 'quiz/admin_panel/admin_login.html')


def admin_logout(request):
    """Admin logout view - logs out and redirects to admin login"""
    # Store the user's admin status before logging out
    was_admin = request.user.is_staff or request.user.is_superuser if request.user.is_authenticated else False
    
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    
    if was_admin:
        return redirect('admin_login')
    return redirect('user_login')


def user_logout(request):
    """User logout view - logs out and redirects to user login"""
    # Store the user's admin status before logging out
    was_admin = request.user.is_staff or request.user.is_superuser if request.user.is_authenticated else False
    
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    
    if was_admin:
        return redirect('admin_login')
    return redirect('user_login')


def register(request):
    """User registration view with name field and optional profile image"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        username = request.POST.get('username')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        profile_image = request.FILES.get('profile_image')
        
        # Validate inputs
        if not all([name, email, username, password1, password2]):
            messages.error(request, 'All fields are required')
            return redirect('register')
        
        if password1 != password2:
            messages.error(request, 'Passwords do not match')
            return redirect('register')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return redirect('register')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists')
            return redirect('register')
        
        # Validate profile image if uploaded
        if profile_image:
            # Validate file extension
            allowed_extensions = ['.jpg', '.jpeg', '.png']
            import os
            ext = os.path.splitext(profile_image.name)[1].lower()
            
            if ext not in allowed_extensions:
                messages.error(request, 'Invalid file format. Only .jpg, .jpeg, and .png files are allowed.')
                return redirect('register')
            
            # Validate file size (2MB = 2 * 1024 * 1024 bytes)
            max_size = 2 * 1024 * 1024
            if profile_image.size > max_size:
                messages.error(request, 'File size exceeds 2MB limit. Please upload a smaller image.')
                return redirect('register')
        
        # Create user (active by default - auto-approved on registration)
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            is_active=True  # User is active immediately after registration
        )
        
        # Create or update user profile with name, status, and profile image
        profile, created = UserProfile.objects.get_or_create(user=user)
        profile.name = name
        profile.status = 'Approved'  # Auto-approve on registration
        
        # Save profile image if uploaded
        if profile_image:
            profile.profile_image = profile_image
        
        profile.save()
        
        messages.success(request, 'Registration Successful! You can now log in with your credentials.')
        return redirect('user_login')
    
    return render(request, 'quiz/auth/register.html')


@login_required
def dashboard(request):
    """User dashboard showing PDFs and quizzes"""
    # Redirect admin users to admin dashboard
    if request.user.is_staff or request.user.is_superuser:
        return redirect('admin_dashboard')
    
    user_pdfs = PDFDocument.objects.filter(user=request.user)
    user_quizzes = Quiz.objects.filter(user=request.user)
    user_results = Result.objects.filter(user=request.user)[:5]
    
    context = {
        'pdfs': user_pdfs,
        'quizzes': user_quizzes,
        'results': user_results,
    }
    return render(request, 'quiz/dashboard.html', context)


@login_required
def upload_pdf(request):
    """Handle PDF upload"""
    # Block admin users from uploading PDFs
    if request.user.is_staff or request.user.is_superuser:
        return HttpResponseForbidden("Admin users cannot upload PDFs.")
    
    if request.method == 'POST':
        if 'pdf_file' not in request.FILES:
            messages.error(request, 'No file uploaded')
            return redirect('dashboard')
        
        pdf_file = request.FILES['pdf_file']
        title = request.POST.get('title', pdf_file.name)
        
        # Validate file type
        if not pdf_file.name.lower().endswith('.pdf'):
            messages.error(request, 'Only PDF files are allowed')
            return redirect('dashboard')
        
        # Save PDF document
        pdf_doc = PDFDocument.objects.create(
            user=request.user,
            title=title,
            file=pdf_file
        )
        
        # Extract text from PDF
        try:
            pdf_doc.file.seek(0)  # Reset file pointer
            text_content = extract_text_from_pdf(pdf_doc.file)
            pdf_doc.text_content = text_content
            pdf_doc.save()
            messages.success(request, 'PDF uploaded and processed successfully')
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            messages.warning(request, 'PDF uploaded but text extraction failed')
        
        return redirect('dashboard')
    
    return redirect('dashboard')


@login_required
def create_quiz_from_topic(request):
    """Create quiz from topic without requiring PDF"""
    # Block admin users from creating quizzes
    if request.user.is_staff or request.user.is_superuser:
        return HttpResponseForbidden("Admin users cannot create quizzes.")
    
    if request.method == 'POST':
        topic = request.POST.get('topic')
        difficulty = request.POST.get('difficulty')
        num_questions = int(request.POST.get('num_questions', 10))
        
        if not topic:
            messages.error(request, 'Please provide a topic')
            return redirect('dashboard')
        
        # Validate inputs
        if difficulty not in ['Easy', 'Medium', 'Hard']:
            messages.error(request, 'Invalid difficulty level')
            return redirect('dashboard')
        
        if num_questions < 1 or num_questions > 50:
            messages.error(request, 'Number of questions must be between 1 and 50')
            return redirect('dashboard')
        
        # Create a placeholder PDF document for topic-based quizzes
        pdf_doc = PDFDocument.objects.create(
            user=request.user,
            title=f"Topic: {topic}",
            file=None,
            text_content=f"Topic: {topic}"
        )
        
        # Create quiz object
        quiz = Quiz.objects.create(
            user=request.user,
            pdf=pdf_doc,
            subject=topic,
            difficulty=difficulty,
            number_of_questions=num_questions
        )
        
        # =========================================
        # Prompt Structuring Algorithm
        # =========================================
        
        # =========================================
        # LLM-Based Question Generation Algorithm
        # =========================================
        
        # Generate questions using AI (without PDF content)
        try:
            questions_data = generate_quiz_questions_safe(
                pdf_text=f"Generate questions about: {topic}",
                subject=topic,
                difficulty=difficulty,
                num_questions=num_questions
            )
            
            # =========================================
            # JSON Parsing Algorithm
            # =========================================
            
            # Save questions to database
            for idx, q_data in enumerate(questions_data, 1):
                Question.objects.create(
                    quiz=quiz,
                    question_text=q_data.get('question', ''),
                    option_a=q_data.get('options', {}).get('A', ''),
                    option_b=q_data.get('options', {}).get('B', ''),
                    option_c=q_data.get('options', {}).get('C', ''),
                    option_d=q_data.get('options', {}).get('D', ''),
                    correct_answer=q_data.get('correct_answer', 'A'),
                    explanation=q_data.get('explanation', ''),
                    order=idx
                )
            
            quiz.is_completed = True
            quiz.save()
            messages.success(request, f'Quiz generated successfully with {len(questions_data)} questions!')
            
        except APIKeyError as e:
            logger.error(f"API Key Error: {e}")
            messages.error(request, 'AI service not configured. Please contact administrator.')
            quiz.delete()
            return redirect('dashboard')
        except Exception as e:
            logger.error(f"Quiz generation error: {e}")
            messages.error(request, f'Failed to generate quiz: {str(e)}')
            quiz.delete()
            return redirect('dashboard')
        
        return redirect('take_quiz', quiz_id=quiz.id)
    
    return redirect('dashboard')


@login_required
def create_quiz(request, pdf_id):
    """Create quiz from uploaded PDF"""
    # Block admin users from creating quizzes
    if request.user.is_staff or request.user.is_superuser:
        return HttpResponseForbidden("Admin users cannot create quizzes.")
    
    pdf_doc = get_object_or_404(PDFDocument, id=pdf_id, user=request.user)
    
    if request.method == 'POST':
        subject = request.POST.get('subject')
        difficulty = request.POST.get('difficulty')
        num_questions = int(request.POST.get('num_questions', 10))
        
        if not subject:
            messages.error(request, 'Please provide a subject name')
            return redirect('dashboard')
        
        # Validate inputs
        if difficulty not in ['Easy', 'Medium', 'Hard']:
            messages.error(request, 'Invalid difficulty level')
            return redirect('dashboard')
        
        if num_questions < 1 or num_questions > 50:
            messages.error(request, 'Number of questions must be between 1 and 50')
            return redirect('dashboard')
        
        # Check if PDF has text content
        if not pdf_doc.text_content:
            messages.error(request, 'PDF has no extractable text content')
            return redirect('dashboard')
        
        # Create quiz object
        quiz = Quiz.objects.create(
            user=request.user,
            pdf=pdf_doc,
            subject=subject,
            difficulty=difficulty,
            number_of_questions=num_questions
        )
        
        # =========================================
        # Prompt Structuring Algorithm
        # =========================================
        
        # =========================================
        # LLM-Based Question Generation Algorithm
        # =========================================
        
        # Generate questions using AI
        try:
            questions_data = generate_quiz_questions_safe(
                pdf_text=pdf_doc.text_content,
                subject=subject,
                difficulty=difficulty,
                num_questions=num_questions
            )
            
            # =========================================
            # JSON Parsing Algorithm
            # =========================================
            
            # Save questions to database
            for idx, q_data in enumerate(questions_data, 1):
                Question.objects.create(
                    quiz=quiz,
                    question_text=q_data.get('question', ''),
                    option_a=q_data.get('options', {}).get('A', ''),
                    option_b=q_data.get('options', {}).get('B', ''),
                    option_c=q_data.get('options', {}).get('C', ''),
                    option_d=q_data.get('options', {}).get('D', ''),
                    correct_answer=q_data.get('correct_answer', 'A'),
                    explanation=q_data.get('explanation', ''),
                    order=idx
                )
            
            quiz.is_completed = True
            quiz.save()
            messages.success(request, f'Quiz generated successfully with {len(questions_data)} questions!')
            
        except APIKeyError as e:
            logger.error(f"API Key Error: {e}")
            messages.error(request, 'AI service not configured. Please contact administrator.')
            quiz.delete()
            return redirect('dashboard')
        except Exception as e:
            logger.error(f"Quiz generation error: {e}")
            messages.error(request, f'Failed to generate quiz: {str(e)}')
            quiz.delete()
            return redirect('dashboard')
        
        return redirect('take_quiz', quiz_id=quiz.id)
    
    context = {
        'pdf': pdf_doc,
    }
    return render(request, 'quiz/create_quiz.html', context)


@login_required
def take_quiz(request, quiz_id):
    """Display quiz for taking"""
    quiz = get_object_or_404(Quiz, id=quiz_id, user=request.user)
    questions = quiz.questions.all()
    
    # Calculate total quiz time: 60 seconds per question
    total_questions = questions.count()
    total_time = total_questions * 60  # Total time in seconds
    
    context = {
        'quiz': quiz,
        'questions': questions,
        'total_time': total_time,
        'total_questions': total_questions,
    }
    return render(request, 'quiz/take_quiz.html', context)


@login_required
def submit_quiz(request, quiz_id):
    """Submit quiz answers and calculate score with AI feedback"""
    quiz = get_object_or_404(Quiz, id=quiz_id, user=request.user)
    questions = quiz.questions.all()
    
    if request.method == 'POST':
        # =========================================
        # Score Calculation Algorithm
        # =========================================
        
        # Use quiz.questions.count() to get total questions
        total_questions = quiz.questions.count()
        correct_count = 0
        
        # Collect user answers for feedback
        user_answers = []
        
        for question in questions:
            answer_key = f'question_{question.id}'
            selected_answer = request.POST.get(answer_key)
            
            if selected_answer:
                is_correct = selected_answer == question.correct_answer
                if is_correct:
                    correct_count += 1
                
                user_answers.append({
                    'question': question.question_text,
                    'selected': selected_answer,
                    'correct': question.correct_answer,
                    'is_correct': is_correct
                })
                
                # Save user answer
                UserAnswer.objects.update_or_create(
                    user=request.user,
                    quiz=quiz,
                    question=question,
                    defaults={'selected_answer': selected_answer, 'is_correct': is_correct}
                )
            else:
                # Unanswered question - mark as incorrect
                user_answers.append({
                    'question': question.question_text,
                    'selected': None,
                    'correct': question.correct_answer,
                    'is_correct': False
                })
                
                # Save user answer as incorrect (unanswered)
                UserAnswer.objects.update_or_create(
                    user=request.user,
                    quiz=quiz,
                    question=question,
                    defaults={'selected_answer': '', 'is_correct': False}
                )
        
        # Calculate wrong answers: total questions - correct answers
        # This ensures unanswered questions are counted as wrong
        wrong_count = total_questions - correct_count
        
        # Score is the same as correct_count (number of correct answers)
        score = correct_count
        
        # Calculate percentage: (correct_answers / total_questions) * 100
        percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        # =========================================
        # Weak Area Detection Algorithm
        # =========================================
        
        # Generate AI feedback
        try:
            feedback = generate_performance_feedback_safe(
                subject=quiz.subject,
                total_questions=total_questions,
                correct_count=correct_count,
                wrong_count=wrong_count,
                user_answers=user_answers,
                difficulty=quiz.difficulty
            )
        except APIKeyError as e:
            logger.error(f"API Key Error: {e}")
            feedback = {
                'strength_analysis': 'AI service not configured. Please contact administrator for detailed feedback.',
                'weakness_analysis': 'AI service not configured. Please contact administrator for detailed feedback.',
                'suggestions': 'AI service not configured. Please contact administrator for detailed feedback.'
            }
        except Exception as e:
            logger.error(f"AI feedback generation error: {e}")
            feedback = {
                'strength_analysis': 'Keep practicing to improve your understanding.',
                'weakness_analysis': 'Review the topics you got wrong.',
                'suggestions': 'Consider retaking the quiz after studying.'
            }
        
        # Save result
        result = Result.objects.create(
            user=request.user,
            quiz=quiz,
            score=score,
            total_questions=total_questions,
            correct_answers=correct_count,
            wrong_answers=wrong_count,
            percentage=round(percentage, 2),
            strength_analysis=feedback.get('strength_analysis', ''),
            weakness_analysis=feedback.get('weakness_analysis', ''),
            suggestions=feedback.get('suggestions', '')
        )
        
        # Store result ID in session for PRG pattern
        request.session['last_result_id'] = result.id
        
        # Redirect to result page (POST-Redirect-GET pattern)
        return redirect('quiz_result', quiz_id=quiz.id)
    
    return redirect('take_quiz', quiz_id=quiz.id)


@login_required
def quiz_result(request, quiz_id):
    """Display quiz result page (GET request after POST-Redirect-GET)"""
    quiz = get_object_or_404(Quiz, id=quiz_id, user=request.user)
    
    # Try to get result from session (PRG pattern)
    result_id = request.session.get('last_result_id')
    
    if result_id:
        result = Result.objects.filter(id=result_id, user=request.user, quiz=quiz).first()
        # Clear the session after retrieving
        del request.session['last_result_id']
    else:
        # Fallback: get the most recent result for this quiz
        result = Result.objects.filter(user=request.user, quiz=quiz).first()
    
    if not result:
        messages.error(request, 'No result found for this quiz.')
        return redirect('dashboard')
    
    # Get detailed question analysis
    questions = quiz.questions.all().order_by('order')
    user_answers = UserAnswer.objects.filter(user=request.user, quiz=quiz)
    
    # Create a dictionary mapping question_id to user answer
    answer_dict = {ua.question_id: ua for ua in user_answers}
    
    # =========================================
    # Topic-wise Accuracy Analysis Algorithm
    # =========================================
    
    # Build detailed analysis list
    detailed_results = []
    for question in questions:
        user_answer = answer_dict.get(question.id)
        selected_answer = user_answer.selected_answer if user_answer else None
        is_correct = user_answer.is_correct if user_answer else False
        
        # Get the selected and correct answer text
        option_mapping = {
            'A': question.option_a,
            'B': question.option_b,
            'C': question.option_c,
            'D': question.option_d,
        }
        
        selected_text = option_mapping.get(selected_answer, 'No answer') if selected_answer else 'No answer'
        correct_text = option_mapping.get(question.correct_answer, '')
        
        detailed_results.append({
            'question_number': question.order,
            'question_text': question.question_text,
            'selected_answer': selected_answer,
            'selected_text': selected_text,
            'correct_answer': question.correct_answer,
            'correct_text': correct_text,
            'is_correct': is_correct,
            'explanation': question.explanation,
            'options': option_mapping,
        })
    
    context = {
        'quiz': quiz,
        'score': result.total_questions,  # Use total_questions for "Total Questions" display
        'total': result.total_questions,
        'correct': result.correct_answers,
        'wrong': result.wrong_answers,
        'percentage': result.percentage,
        'result': result,
        'detailed_results': detailed_results,
    }
    return render(request, 'quiz/quiz_result.html', context)


@login_required
def quiz_detail(request, quiz_id):
    """View quiz details and results"""
    # Fetch quiz by id only - no user restriction
    # This allows admin to view ANY quiz
    quiz = get_object_or_404(Quiz, id=quiz_id)
    questions = quiz.questions.all()
    
    # For answers and results:
    # - Admins can see all answers/results for the quiz
    # - Regular users see only their own answers/results
    if request.user.is_staff or request.user.is_superuser:
        answers = UserAnswer.objects.filter(quiz=quiz)
        result = Result.objects.filter(quiz=quiz)
    else:
        answers = UserAnswer.objects.filter(user=request.user, quiz=quiz)
        result = Result.objects.filter(user=request.user, quiz=quiz).first()
    
    context = {
        'quiz': quiz,
        'questions': questions,
        'answers': answers,
        'result': result,
    }
    return render(request, 'quiz/quiz_detail.html', context)


@login_required
def delete_pdf(request, pdf_id):
    """Delete uploaded PDF"""
    pdf = get_object_or_404(PDFDocument, id=pdf_id, user=request.user)
    pdf.delete()
    messages.success(request, 'PDF deleted successfully')
    return redirect('dashboard')


@login_required
def delete_quiz(request, quiz_id):
    """Delete quiz"""
    quiz = get_object_or_404(Quiz, id=quiz_id, user=request.user)
    quiz.delete()
    messages.success(request, 'Quiz deleted successfully')
    return redirect('dashboard')


# AJAX endpoints for real-time feedback

@login_required
def generate_quiz_ajax(request):
    """AJAX endpoint for generating quiz"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
    pdf_id = request.POST.get('pdf_id')
    subject = request.POST.get('subject')
    difficulty = request.POST.get('difficulty')
    num_questions = int(request.POST.get('num_questions', 10))
    
    # Validate inputs
    if not all([pdf_id, subject, difficulty]):
        return JsonResponse({'error': 'Missing required fields'}, status=400)
    
    pdf_doc = get_object_or_404(PDFDocument, id=pdf_id, user=request.user)
    
    try:
        questions_data = generate_quiz_questions_safe(
            pdf_text=pdf_doc.text_content,
            subject=subject,
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return JsonResponse({
            'success': True,
            'questions': questions_data,
            'count': len(questions_data)
        })
        
    except Exception as e:
        logger.error(f"AJAX quiz generation error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# Admin Views (Staff only)

@login_required
def admin_dashboard(request):
    """Admin dashboard showing all users and quizzes"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('dashboard')
    
    users = User.objects.all().order_by('-date_joined')
    all_quizzes = Quiz.objects.all().order_by('-created_at')
    
    # Get user profiles
    user_profiles = {}
    for user in users:
        try:
            profile = user.profile
            user_profiles[user.id] = profile.status
        except UserProfile.DoesNotExist:
            user_profiles[user.id] = 'Waiting'
    
    context = {
        'users': users,
        'quizzes': all_quizzes,
        'user_profiles': user_profiles,
        'total_users': users.count(),
        'total_quizzes': all_quizzes.count(),
    }
    return render(request, 'quiz/admin_panel/admin_dashboard.html', context)


@login_required
def approve_user(request, user_id):
    """Approve user registration (admin only)"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('dashboard')
    
    target_user = get_object_or_404(User, id=user_id)
    
    # Prevent admin from modifying themselves
    if target_user == request.user:
        messages.error(request, 'You cannot modify your own account.')
        return redirect('admin_dashboard')
    
    # Update user profile status
    try:
        profile = target_user.profile
        profile.status = 'Approved'
        profile.save()
    except UserProfile.DoesNotExist:
        UserProfile.objects.create(user=target_user, status='Approved')
    
    # Activate the user
    target_user.is_active = True
    target_user.save()
    
    messages.success(request, f'User {target_user.username} has been approved.')
    return redirect('admin_dashboard')


@login_required
def reject_user(request, user_id):
    """Reject user registration (admin only)"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('dashboard')
    
    target_user = get_object_or_404(User, id=user_id)
    
    # Prevent admin from modifying themselves
    if target_user == request.user:
        messages.error(request, 'You cannot modify your own account.')
        return redirect('admin_dashboard')
    
    # Update user profile status
    try:
        profile = target_user.profile
        profile.status = 'Rejected'
        profile.save()
    except UserProfile.DoesNotExist:
        UserProfile.objects.create(user=target_user, status='Rejected')
    
    # Deactivate the user
    target_user.is_active = True
    target_user.save()
    
    messages.warning(request, f'User {target_user.username} has been rejected.')
    return redirect('admin_dashboard')


@login_required
def toggle_user_status(request, user_id):
    """Toggle user activation status (admin only) - Legacy function"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('dashboard')
    
    target_user = get_object_or_404(User, id=user_id)
    
    # Prevent admin from deactivating themselves
    if target_user == request.user:
        messages.error(request, 'You cannot deactivate your own account.')
        return redirect('admin_dashboard')
    
    target_user.is_active = not target_user.is_active
    target_user.save()
    
    status = 'activated' if target_user.is_active else 'deactivated'
    messages.success(request, f'User {target_user.username} has been {status}.')
    return redirect('admin_dashboard')
