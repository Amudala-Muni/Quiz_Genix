"""
AI Services for QuizGenix
LangChain + Google Gemini Integration
Synchronous version (Safe for Django)
Fixed for event loop issues in Django threads
"""

import os
import json
import re
import logging
import asyncio
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from django.conf import settings

logger = logging.getLogger(__name__)


class APIKeyError(Exception):
    pass


# Thread-local storage for event loops
_thread_local = threading.local()

# Global executor for running async code - use daemon threads for cleaner shutdown
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai_async_")


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create an event loop for the current thread.
    This solves the 'no current event loop' error in Django threads.
    
    Returns:
        A new or existing event loop for the current thread.
    """
    try:
        # Try to get the existing event loop for this thread
        loop = asyncio.get_event_loop()
        # Check if the loop is closed or not running
        if loop.is_closed():
            # Create a new loop if the old one was closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        # No event loop exists for this thread - create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _run_async_in_thread(coro):
    """
    Run an async coroutine in a separate thread with its own event loop.
    This ensures compatibility with Django's synchronous views.
    
    Args:
        coro: An async coroutine to execute.
    
    Returns:
        The result of the coroutine execution.
    """
    def _run():
        # Always create a fresh event loop for this thread execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Always close the loop to prevent resource leaks
            loop.close()
    
    future = _executor.submit(_run)
    return future.result()


def get_llm():
    """
    Get configured Google Gemini LLM instance.
    """
    api_key = getattr(settings, "GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise APIKeyError("Google API key not configured")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )


def build_quiz_prompt(pdf_text: str, subject: str, difficulty: str, num_questions: int) -> str:
    """
    Build prompt for quiz question generation.
    """
    return f"""
Generate {num_questions} MCQ questions.

Subject: {subject}
Difficulty: {difficulty}

Content:
{pdf_text[:4000]}

Return JSON format:

[
{{
"question":"Question text",
"options":{{
"A":"Option A",
"B":"Option B",
"C":"Option C",
"D":"Option D"
}},
"correct_answer":"A",
"explanation":"Explanation"
}}
]
"""


def parse_questions(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse AI response into structured question data.
    """
    try:
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response_text)

        if json_match:
            questions = json.loads(json_match.group())
        else:
            questions = json.loads(response_text)

        # Validate and normalize structure
        normalized = []
        for q in questions:
            normalized.append({
                "question": q.get("question", ""),
                "options": q.get("options", {}),
                "correct_answer": q.get("correct_answer", "A"),
                "explanation": q.get("explanation", "")
            })

        return normalized

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise Exception("AI response parsing failed - invalid JSON format")
    except Exception as e:
        logger.error(f"Question parsing error: {e}")
        raise Exception("AI response parsing failed")


def _llm_invoke_sync(llm, prompt: str) -> str:
    """
    Safely invoke LLM in a way that works with Django.
    Uses thread-based event loop execution.
    """
    async def _async_invoke():
        # Use ainvoke for async-safe invocation
        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    
    return _run_async_in_thread(_async_invoke())


def generate_quiz_questions_sync(pdf_text: str, subject: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
    """
    Generate quiz questions from PDF content.
    Fully synchronous - safe for Django views.
    """
    llm = get_llm()
    prompt = build_quiz_prompt(pdf_text, subject, difficulty, num_questions)

    try:
        # Use async-safe invocation in separate thread
        response_text = _llm_invoke_sync(llm, prompt)
        return parse_questions(response_text)

    except Exception as e:
        logger.error(f"AI quiz generation error: {e}")
        raise Exception(f"AI quiz generation failed: {str(e)}")


# For backward compatibility - map async-looking names to sync versions
generate_quiz_questions = generate_quiz_questions_sync


def build_feedback_prompt(subject: str, total: int, correct: int, wrong: int, 
                          user_answers: List[Dict[str, Any]], difficulty: str) -> str:
    """
    Build prompt for performance feedback analysis.
    """
    percentage = (correct / total) * 100 if total else 0

    summary = ""
    for i, a in enumerate(user_answers[:10], 1):
        status = "Correct" if a.get("is_correct", False) else "Wrong"
        question_text = a.get("question", "")[:80]
        summary += f"{i}. {question_text} - {status}\n"

    return f"""
Analyze quiz performance and provide actionable feedback.

Subject: {subject}
Difficulty: {difficulty}

Total Questions: {total}
Correct Answers: {correct}
Wrong Answers: {wrong}
Percentage: {percentage:.1f}%

User Answers:
{summary}

Return JSON format with three keys:

{{
"strength_analysis":"What the user did well - specific topics/concepts mastered",
"weakness_analysis":"What areas need improvement - specific topics/concepts to review",
"suggestions":"Actionable study recommendations for improvement"
}}

Provide detailed, specific feedback based on the actual performance.
"""


def parse_feedback(response_text: str) -> Dict[str, str]:
    """
    Parse AI feedback response into structured data.
    """
    try:
        # Try to find JSON object in response
        json_match = re.search(r'\{{[\s\S]*\}}', response_text)

        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response_text)

        return {
            "strength_analysis": data.get("strength_analysis", "Good overall performance."),
            "weakness_analysis": data.get("weakness_analysis", "Some areas need improvement."),
            "suggestions": data.get("suggestions", "Continue practicing to strengthen understanding.")
        }

    except Exception as e:
        logger.warning(f"Feedback parsing warning: {e}")
        # Return default feedback if parsing fails
        return {
            "strength_analysis": "Good attempt at the quiz.",
            "weakness_analysis": "Some questions were challenging.",
            "suggestions": "Review the material and try again for better results."
        }


def generate_performance_feedback_sync(subject: str, total: int, correct: int, 
                                        wrong: int, user_answers: List[Dict[str, Any]], 
                                        difficulty: str) -> Dict[str, str]:
    """
    Generate performance feedback from quiz results.
    Fully synchronous - safe for Django views.
    """
    llm = get_llm()
    prompt = build_feedback_prompt(subject, total, correct, wrong, user_answers, difficulty)

    try:
        # Use async-safe invocation in separate thread
        response_text = _llm_invoke_sync(llm, prompt)
        return parse_feedback(response_text)

    except Exception as e:
        logger.error(f"AI feedback generation error: {e}")
        raise Exception(f"AI feedback generation failed: {str(e)}")


# For backward compatibility - map async-looking names to sync versions
generate_performance_feedback = generate_performance_feedback_sync


def test_ai_connection() -> Dict[str, Any]:
    """
    Test AI connection and configuration.
    Returns status information.
    """
    try:
        llm = get_llm()
        
        # Use async-safe invocation
        response = _llm_invoke_sync(llm, "Say 'AI connection successful' if you can read this.")
        
        return {
            "status": "success",
            "message": "AI connected successfully",
            "configured": True,
            "response": response[:100] if response else ""
        }

    except APIKeyError as e:
        return {
            "status": "error",
            "message": f"API Key Error: {str(e)}",
            "configured": False
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "configured": True  # API key exists but connection failed
        }


def shutdown_executor():
    """
    Gracefully shutdown the thread pool executor.
    Call this during Django application cleanup.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
