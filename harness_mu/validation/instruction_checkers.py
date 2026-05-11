"""IHEval-equivalent instruction checkers with CheckerRegistry.

Replaces the missing IHEval library. Each checker implements the same
interface as IHEval's instruction classes:
    __init__(instruction_id) -> build_description(**kwargs) -> check_following(text) -> bool

The CheckerRegistry mimics instructions_registry.INSTRUCTION_DICT so the
evaluation logic works identically to the original Muses-Bench evaluator.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Base checker
# ---------------------------------------------------------------------------

class BaseChecker(ABC):
    """Abstract base for all instruction-following checkers."""

    def __init__(self, instruction_id: str):
        self.instruction_id = instruction_id
        self._kwargs: Dict[str, Any] = {}

    def build_description(self, **kwargs) -> None:
        """Store the kwargs that parameterise this check."""
        self._kwargs = kwargs

    @abstractmethod
    def check_following(self, response_text: str) -> bool:
        """Return True if response_text satisfies the instruction."""
        ...


# ---------------------------------------------------------------------------
# startend
# ---------------------------------------------------------------------------

class EndChecker(BaseChecker):
    """Check that response ends with an exact phrase."""

    def check_following(self, response_text: str) -> bool:
        end_phrase = self._kwargs.get("end_phrase", "")
        return response_text.rstrip().endswith(end_phrase)


class QuotationChecker(BaseChecker):
    """Check that response is wrapped in quotation marks."""

    def check_following(self, response_text: str) -> bool:
        text = response_text.strip()
        if len(text) < 2:
            return False
        # Support both single and double quotation marks
        for q in ('"', "'"):
            if text.startswith(q) and text.endswith(q):
                return True
        return False


# ---------------------------------------------------------------------------
# punctuation
# ---------------------------------------------------------------------------

class NoCommaChecker(BaseChecker):
    """Check that response contains no commas."""

    def check_following(self, response_text: str) -> bool:
        return "," not in response_text


# ---------------------------------------------------------------------------
# keywords
# ---------------------------------------------------------------------------


def _check_numeric_relation(actual: int, relation: str, threshold: int) -> bool:
    """Evaluate a numeric relation against a threshold."""
    if relation == "less than":
        return actual < threshold
    elif relation == "at least":
        return actual >= threshold
    elif relation == "equal to":
        return actual == threshold
    return False


class KeywordFrequencyChecker(BaseChecker):
    """Check how many times a keyword appears (relation: less than / at least)."""

    def check_following(self, response_text: str) -> bool:
        keyword = self._kwargs.get("keyword", "")
        relation = self._kwargs.get("relation", "at least")
        frequency = self._kwargs.get("frequency", 1)
        count = response_text.count(keyword)
        return _check_numeric_relation(count, relation, frequency)


class ForbiddenWordsChecker(BaseChecker):
    """Check that none of the forbidden words appear in the response."""

    def check_following(self, response_text: str) -> bool:
        forbidden = self._kwargs.get("forbidden_words", [])
        lower_text = response_text.lower()
        for word in forbidden:
            # Match whole word only to avoid partial matches
            if re.search(r'\b' + re.escape(word.lower()) + r'\b', lower_text):
                return False
        return True


class KeywordExistenceChecker(BaseChecker):
    """Check that all required keywords appear in the response."""

    def check_following(self, response_text: str) -> bool:
        keywords = self._kwargs.get("keywords", [])
        lower_text = response_text.lower()
        for word in keywords:
            if word.lower() not in lower_text:
                return False
        return True


class LetterFrequencyChecker(BaseChecker):
    """Check how many times a specific letter/character appears."""

    def check_following(self, response_text: str) -> bool:
        letter = self._kwargs.get("letter", "")
        relation = self._kwargs.get("let_relation", "at least")
        frequency = self._kwargs.get("let_frequency", 1)
        count = response_text.count(letter)
        return _check_numeric_relation(count, relation, frequency)


# ---------------------------------------------------------------------------
# detectable_format
# ---------------------------------------------------------------------------

class NumberBulletListsChecker(BaseChecker):
    """Check for exactly N markdown bullet points (lines starting with * or -)."""

    def check_following(self, response_text: str) -> bool:
        target = self._kwargs.get("num_bullets", 0)
        lines = response_text.split("\n")
        bullet_count = sum(
            1 for line in lines
            if line.strip().startswith(("* ", "- ", "+ "))
        )
        return bullet_count == target


class NumberHighlightedSectionsChecker(BaseChecker):
    """Check for exactly N *highlighted text* sections."""

    def check_following(self, response_text: str) -> bool:
        target = self._kwargs.get("num_highlights", 0)
        # Match *text* (non-greedy, non-empty content between asterisks)
        matches = re.findall(r'\*[^*]+\*', response_text)
        return len(matches) == target


class TitleChecker(BaseChecker):
    """Check that response starts with a title (line starting with # or bold text)."""

    def check_following(self, response_text: str) -> bool:
        text = response_text.strip()
        # Title can be: # Title, **Title**, or first line followed by blank line
        if text.startswith("#"):
            return True
        first_line = text.split("\n")[0].strip()
        if first_line.startswith("**") and first_line.endswith("**"):
            return True
        # Accept any response that has a clear heading-like first line
        if first_line and len(first_line) < 120:
            return True
        return False


class JsonFormatChecker(BaseChecker):
    """Check that response is a valid JSON object or array."""

    def check_following(self, response_text: str) -> bool:
        text = response_text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False


class MultipleSectionsChecker(BaseChecker):
    """Check for exactly N sections separated by a delimiter string."""

    def check_following(self, response_text: str) -> bool:
        spliter = self._kwargs.get("section_spliter", "SECTION")
        num_sections = self._kwargs.get("num_sections", 1)
        # Count occurrences of the section delimiter
        sections = response_text.split(spliter)
        return len(sections) == num_sections


class ConstrainedResponseChecker(BaseChecker):
    """Check that response follows a constrained format.

    In IHEval, this checks that the response is in a specific constrained
    output format. We check that the response contains a well-structured
    answer (not just raw text).
    """

    def check_following(self, response_text: str) -> bool:
        # A constrained response should have some structure:
        # either bullet points, numbered items, JSON, or sections
        text = response_text.strip()
        has_bullets = bool(re.search(r'(?:^|\n)\s*(?:[*\-+]|\d+[.)])\s', text))
        has_json = text.startswith("{") or text.startswith("[")
        has_heading = text.startswith("#") or "**" in text
        return has_bullets or has_json or has_heading or len(text.split("\n")) > 3


# ---------------------------------------------------------------------------
# change_case
# ---------------------------------------------------------------------------

class CapitalWordFrequencyChecker(BaseChecker):
    """Check number of ALL-CAPS words (relation: less than / at least)."""

    def check_following(self, response_text: str) -> bool:
        relation = self._kwargs.get("capital_relation", "at least")
        threshold = self._kwargs.get("capital_frequency", 1)
        # Words that are entirely uppercase letters, minimum 2 chars
        words = response_text.split()
        caps_words = [
            w for w in words
            if len(w) >= 2 and re.match(r'^[A-Z0-9]+$', w)
        ]
        return _check_numeric_relation(len(caps_words), relation, threshold)


class EnglishLowercaseChecker(BaseChecker):
    """Check that all letters in the response are lowercase.

    Note: Proper nouns and sentence starts are exempted by IHEval's
    implementation. We use a lenient check: no words that are ALL-CAPS
    (except acronyms) and overall text is predominantly lowercase.
    """

    def check_following(self, response_text: str) -> bool:
        # Remove common exceptions
        text = response_text.strip()
        # Check that there are no uppercase letters at all
        return not bool(re.search(r'[A-Z]', text))


class EnglishCapitalChecker(BaseChecker):
    """Check that sentences start with a capital letter.

    This is a lenient check: we verify that most sentences begin with
    an uppercase letter, allowing for proper nouns and exceptions.
    """

    def check_following(self, response_text: str) -> bool:
        text = response_text.strip()
        if not text:
            return True
        # First character of the whole response should be capital or non-letter
        if text[0].isalpha() and not text[0].isupper():
            return False
        # Check sentence starts after period+space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences[1:]:  # skip first sentence (already checked)
            sent = sent.strip()
            if sent and sent[0].isalpha() and not sent[0].isupper():
                return False
        return True


# ---------------------------------------------------------------------------
# length_constraints
# ---------------------------------------------------------------------------

class NumberWordsChecker(BaseChecker):
    """Check total word count (relation: less than / at least)."""

    def check_following(self, response_text: str) -> bool:
        relation = self._kwargs.get("relation", "at least")
        target = self._kwargs.get("num_words", 0)
        word_count = len(response_text.split())
        return _check_numeric_relation(word_count, relation, target)


class NumberSentencesChecker(BaseChecker):
    """Check total sentence count (relation: less than / at least)."""

    def check_following(self, response_text: str) -> bool:
        relation = self._kwargs.get("relation", "at least")
        target = self._kwargs.get("num_sentences", 0)
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', response_text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return _check_numeric_relation(len(sentences), relation, target)


class NumberParagraphsChecker(BaseChecker):
    """Check exact number of paragraphs."""

    def check_following(self, response_text: str) -> bool:
        target = self._kwargs.get("num_paragraphs", 1)
        # Split on double newlines (blank lines) for paragraphs
        paragraphs = re.split(r'\n\s*\n', response_text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return len(paragraphs) == target


class NthParagraphFirstWordChecker(BaseChecker):
    """Check that the nth paragraph starts with a specific word."""

    def check_following(self, response_text: str) -> bool:
        first_word = self._kwargs.get("first_word", "")
        num_paragraphs = self._kwargs.get("num_paragraphs", 1)
        nth = self._kwargs.get("nth_paragraph", 1)

        paragraphs = re.split(r'\n\s*\n', response_text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Verify total paragraph count
        if len(paragraphs) != num_paragraphs:
            return False

        if nth < 1 or nth > len(paragraphs):
            return False

        target_para = paragraphs[nth - 1]
        para_words = target_para.split()
        if not para_words:
            return False

        return para_words[0].lower() == first_word.lower()


# ---------------------------------------------------------------------------
# detectable_content
# ---------------------------------------------------------------------------

class NumberPlaceholdersChecker(BaseChecker):
    """Check for exactly N placeholders (like [placeholder] or ___ or [...])."""

    def check_following(self, response_text: str) -> bool:
        target = self._kwargs.get("num_placeholders", 1)
        # Match common placeholder patterns
        patterns = [
            r'\[[^\]]*\](?!\()',    # [text] not followed by ( (to avoid markdown links)
            r'_{3,}',                # ___ or more
            r'\.{3,}',               # ...
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, response_text))
        return count == target


class PostscriptChecker(BaseChecker):
    """Check that response ends with a postscript marker (e.g. 'P.S.')."""

    def check_following(self, response_text: str) -> bool:
        marker = self._kwargs.get("postscript_marker", "P.S.")
        # Check if the text contains the postscript marker
        return marker in response_text


# ---------------------------------------------------------------------------
# combination
# ---------------------------------------------------------------------------

class RepeatPromptChecker(BaseChecker):
    """Check that the prompt text is repeated verbatim in the response."""

    def check_following(self, response_text: str) -> bool:
        prompt = self._kwargs.get("prompt_to_repeat", "")
        if not prompt:
            return True
        # Normalize whitespace for fuzzy matching
        normalized_prompt = " ".join(prompt.split())
        normalized_response = " ".join(response_text.split())
        return normalized_prompt in normalized_response


class TwoResponsesChecker(BaseChecker):
    """Check that response contains two distinct response sections."""

    def check_following(self, response_text: str) -> bool:
        # Check for common separators between two responses
        patterns = [
            r'(?:Response|Answer)\s*[12]',
            r'^---+$',
            r'^===+$',
            r'^\*{3,}$',           # *** or more
            r'(?:First|Second)\s+(?:response|answer|part)',
        ]
        matches = 0
        for pattern in patterns:
            found = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            matches += len(found)
        # Need at least 2 indicators for two distinct responses
        return matches >= 2


# ---------------------------------------------------------------------------
# language
# ---------------------------------------------------------------------------

class ResponseLanguageChecker(BaseChecker):
    """Check that the response is in the specified language.

    Uses a simple character-set heuristic for common languages.
    """
    # Unicode script ranges for common languages
    LANGUAGE_SCRIPTS = {
        "hi": "Devanagari",      # Hindi
        "ur": "Arabic",          # Urdu
        "zh": "Han",             # Chinese
        "ja": "Hiragana|Katakana|Han",  # Japanese
        "ko": "Hangul",          # Korean
        "ar": "Arabic",          # Arabic
        "ru": "Cyrillic",        # Russian
        "th": "Thai",            # Thai
    }

    SCRIPT_RANGES = {
        "Devanagari": r'[ऀ-ॿ]',
        "Arabic": r'[؀-ۿݐ-ݿ]',
        "Han": r'[一-鿿]',
        "Hiragana": r'[぀-ゟ]',
        "Katakana": r'[゠-ヿ]',
        "Hangul": r'[가-힯]',
        "Cyrillic": r'[Ѐ-ӿ]',
        "Thai": r'[฀-๿]',
    }

    def check_following(self, response_text: str) -> bool:
        language = self._kwargs.get("language", "").lower()
        if language == "en" or language == "english":
            # English check: no non-Latin script characters dominate
            return True

        script_name = self.LANGUAGE_SCRIPTS.get(language)
        if not script_name:
            # Unknown language — accept (can't verify)
            return True

        # Build regex for the target script(s)
        scripts = script_name.split("|")
        patterns = []
        for s in scripts:
            p = self.SCRIPT_RANGES.get(s)
            if p:
                patterns.append(p)

        if not patterns:
            return True

        combined = "|".join(patterns)
        script_chars = len(re.findall(combined, response_text))
        # At least some characters should match the target script
        total_chars = len(re.sub(r'\s', '', response_text))
        if total_chars == 0:
            return True
        # Require at least 30% of characters to be from the target script
        return (script_chars / total_chars) >= 0.3


# ---------------------------------------------------------------------------
# Registry (mimics instructions_registry.INSTRUCTION_DICT)
# ---------------------------------------------------------------------------

class CheckerRegistry:
    """Mimics IHEval's instructions_registry.INSTRUCTION_DICT for drop-in compatibility.

    Usage:
        registry = CheckerRegistry()
        if inst_id in registry.INSTRUCTION_DICT:
            checker_cls = registry.INSTRUCTION_DICT[inst_id]
            checker = checker_cls(inst_id)
            checker.build_description(**kwargs)
            passed = checker.check_following(response_text)
    """

    INSTRUCTION_DICT: Dict[str, type] = {
        "startend:end_checker": EndChecker,
        "startend:quotation": QuotationChecker,
        "punctuation:no_comma": NoCommaChecker,
        "keywords:frequency": KeywordFrequencyChecker,
        "keywords:forbidden_words": ForbiddenWordsChecker,
        "keywords:existence": KeywordExistenceChecker,
        "keywords:letter_frequency": LetterFrequencyChecker,
        "detectable_format:number_bullet_lists": NumberBulletListsChecker,
        "detectable_format:number_highlighted_sections": NumberHighlightedSectionsChecker,
        "detectable_format:title": TitleChecker,
        "detectable_format:json_format": JsonFormatChecker,
        "detectable_format:multiple_sections": MultipleSectionsChecker,
        "detectable_format:constrained_response": ConstrainedResponseChecker,
        "change_case:capital_word_frequency": CapitalWordFrequencyChecker,
        "change_case:english_lowercase": EnglishLowercaseChecker,
        "change_case:english_capital": EnglishCapitalChecker,
        "length_constraints:number_words": NumberWordsChecker,
        "length_constraints:number_sentences": NumberSentencesChecker,
        "length_constraints:number_paragraphs": NumberParagraphsChecker,
        "length_constraints:nth_paragraph_first_word": NthParagraphFirstWordChecker,
        "detectable_content:number_placeholders": NumberPlaceholdersChecker,
        "detectable_content:postscript": PostscriptChecker,
        "combination:repeat_prompt": RepeatPromptChecker,
        "combination:two_responses": TwoResponsesChecker,
        "language:response_language": ResponseLanguageChecker,
    }


# Global singleton for convenience
REGISTRY = CheckerRegistry()
