import google.generativeai as genai
import json
from typing import Dict, List, Any
import logging
import os
from pydantic import BaseModel
import time
import re

logger = logging.getLogger(__name__)

class QueryStructure(BaseModel):
    """Structured representation of a parsed query"""
    intent: str
    entities: Dict[str, Any]
    keywords: List[str]
    domain: str
    complexity: str

class LLMProcessor:
    """Optimized LLM processor with better prompting and error handling"""
    
    def __init__(self, model: str = "gemini-1.5-flash", max_tokens: int = 800):
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            logger.info(f"Gemini client initialized with model: {model}")
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY not set. Using fallback processing.")
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Fast query parsing with enhanced fallback
        """
        # For speed, use enhanced fallback primarily
        return self._enhanced_fallback_parse_query(query)
    
    def generate_answer(self, question: str, context: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate answer with optimized prompting and robust fallbacks
        """
        if not retrieved_chunks:
            return self._no_context_answer(question)
        
        # Prepare context more efficiently
        context_text = self._prepare_optimized_context(retrieved_chunks)
        
        # Try LLM first with timeout protection
        try:
            if self.client:
                return self._generate_with_gemini(question, context_text, retrieved_chunks)
            else:
                return self._enhanced_fallback_answer(question, retrieved_chunks)
                
        except Exception as e:
            logger.warning(f"LLM generation failed: {str(e)}")
            return self._enhanced_fallback_answer(question, retrieved_chunks)
    
    def _generate_with_gemini(self, question: str, context_text: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Optimized Gemini generation with better prompting"""
        
        # More concise and effective prompt
        prompt = f"""Based on the document excerpts below, answer the question accurately and concisely.

QUESTION: {question}

DOCUMENT EXCERPTS:
{context_text}

INSTRUCTIONS:
- Answer directly and specifically
- Quote specific clauses when possible
- If information is insufficient, state what's missing
- Be precise about numbers, periods, and conditions

ANSWER:"""

        try:
            start_time = time.time()
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.1,  # Lower temperature for more consistent answers
                    top_p=0.8,
                ),
            )
            
            processing_time = time.time() - start_time
            answer_text = response.text.strip()
            
            # Post-process the answer
            cleaned_answer = self._clean_llm_answer(answer_text)
            
            return {
                "answer": cleaned_answer,
                "reasoning": self._extract_reasoning_fast(cleaned_answer),
                "confidence": self._assess_confidence_fast(retrieved_chunks, cleaned_answer),
                "supporting_chunks": [chunk['id'] for chunk in retrieved_chunks[:2]],
                "token_usage": self._estimate_tokens(prompt + answer_text),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return self._enhanced_fallback_answer(question, retrieved_chunks)
    
    def _prepare_optimized_context(self, chunks: List[Dict]) -> str:
        """Prepare context more efficiently with length limits"""
        context_parts = []
        total_length = 0
        max_context_length = 2000  # Limit context to prevent token overflow
        
        for i, chunk in enumerate(chunks[:3]):  # Only top 3 chunks for speed
            chunk_text = chunk['text']
            
            # Truncate very long chunks
            if len(chunk_text) > 600:
                chunk_text = chunk_text[:600] + "..."
            
            if total_length + len(chunk_text) > max_context_length:
                break
                
            context_parts.append(f"[Excerpt {i+1}] {chunk_text}")
            total_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def _clean_llm_answer(self, answer_text: str) -> str:
        """Clean and format LLM answer"""
        # Remove common LLM prefixes
        answer_text = re.sub(r'^(Answer:|ANSWER:|Response:)\s*', '', answer_text)
        
        # Remove excessive whitespace
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()
        
        # Ensure it ends with proper punctuation
        if answer_text and not answer_text.endswith(('.', '!', '?')):
            answer_text += '.'
        
        return answer_text
    
    def _extract_reasoning_fast(self, answer_text: str) -> str:
        """Fast reasoning extraction"""
        # Look for reasoning indicators
        reasoning_patterns = [
            r'because[^.]*\.',
            r'according to[^.]*\.',
            r'based on[^.]*\.',
            r'as stated[^.]*\.',
            r'the policy[^.]*\.'
        ]
        
        reasoning_parts = []
        for pattern in reasoning_patterns:
            matches = re.finditer(pattern, answer_text, re.IGNORECASE)
            for match in matches:
                reasoning_parts.append(match.group(0))
        
        return ' '.join(reasoning_parts[:2]) if reasoning_parts else "Direct answer from document analysis."
    
    def _assess_confidence_fast(self, chunks: List[Dict], answer: str) -> str:
        """Fast confidence assessment"""
        if not chunks:
            return "low"
        
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        # Check if answer contains specific numbers or definitive statements
        has_specifics = bool(re.search(r'\d+|specific|exactly|precisely', answer, re.IGNORECASE))
        
        if avg_score > 0.8 and has_specifics:
            return "high"
        elif avg_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _enhanced_fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback query parsing with better pattern recognition"""
        import re
        
        entities = {}
        keywords = []
        
        # Enhanced pattern matching
        patterns = {
            'waiting_period': r'waiting period|wait.*period',
            'grace_period': r'grace period',
            'coverage': r'cover|coverage|covered',
            'surgery': r'surgery|operation|surgical',
            'maternity': r'maternity|pregnancy|childbirth',
            'pre_existing': r'pre.?existing|PED',
            'claim': r'claim|discount|NCD',
            'hospital': r'hospital|medical facility',
            'treatment': r'treatment|therapy|AYUSH',
            'room_rent': r'room rent|ICU|accommodation'
        }
        
        query_lower = query.lower()
        intent = "general_inquiry"
        domain = "insurance"
        
        # Determine intent based on patterns
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, query_lower):
                intent = pattern_name
                entities[pattern_name] = True
                keywords.extend(re.findall(r'\b\w+\b', pattern))
                break
        
        # Extract numbers and time periods
        numbers = re.findall(r'\d+', query)
        time_units = re.findall(r'(month|year|day)s?', query_lower)
        
        if numbers:
            entities['numbers'] = numbers
        if time_units:
            entities['time_units'] = time_units
        
        # Extract key terms
        key_terms = re.findall(r'\b[A-Z][a-z]+\b|\b(?:policy|insurance|medical|health)\b', query)
        keywords.extend([term.lower() for term in key_terms])
        
        return {
            "intent": intent,
            "entities": entities,
            "keywords": list(set(keywords + query.lower().split())),
            "domain": domain,
            "complexity": "medium"
        }
    
    def _enhanced_fallback_answer(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Enhanced fallback answer generation with better extraction"""
        if not chunks:
            return self._no_context_answer(question)
        
        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Try advanced rule-based extraction
        answer = self._advanced_rule_extraction(question, sorted_chunks)
        
        confidence = "medium" if sorted_chunks[0].get('relevance_score', 0) > 0.7 else "low"
        
        return {
            "answer": answer,
            "reasoning": f"Answer extracted using advanced pattern matching from {len(chunks)} document sections.",
            "confidence": confidence,
            "supporting_chunks": [chunk['id'] for chunk in sorted_chunks[:2]],
            "token_usage": 0,
            "processing_time": 0.1
        }
    
    def _advanced_rule_extraction(self, question: str, chunks: List[Dict]) -> str:
        """Advanced rule-based answer extraction"""
        question_lower = question.lower()
        
        # Combine text from top chunks
        combined_text = " ".join([chunk['text'] for chunk in chunks[:2]])
        
        # Enhanced extraction patterns
        extraction_rules = {
            'waiting period': [
                r'waiting period[^.]*?(\d+)[^.]*?(month|year|day)s?[^.]*\.',
                r'(\d+)[^.]*?(month|year|day)s?[^.]*?waiting period[^.]*\.'
            ],
            'grace period': [
                r'grace period[^.]*?(\d+)[^.]*?(day|month)s?[^.]*\.',
                r'(\d+)[^.]*?(day|month)s?[^.]*?grace period[^.]*\.'
            ],
            'coverage': [
                r'(cover|covered)[^.]*?\.',
                r'(eligible|entitled)[^.]*?\.'
            ],
            'percentage': [
                r'(\d+)%[^.]*\.',
                r'(\d+)\s*percent[^.]*\.'
            ],
            'define': [
                r'defined as[^.]*\.',
                r'means[^.]*\.',
                r'refers to[^.]*\.'
            ]
        }
        
        # Find best matching rule
        best_answer = ""
        for rule_type, patterns in extraction_rules.items():
            if rule_type in question_lower:
                for pattern in patterns:
                    matches = re.finditer(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        # Find the full sentence containing this match
                        sentences = re.split(r'[.!?]+', combined_text)
                        for sentence in sentences:
                            if match.group(0).lower() in sentence.lower():
                                candidate = sentence.strip()
                                if len(candidate) > len(best_answer):
                                    best_answer = candidate
                                break
        
        # If no specific pattern matched, use keyword-based extraction
        if not best_answer:
            best_answer = self._keyword_based_extraction(question, combined_text)
        
        # Ensure proper ending
        if best_answer and not best_answer.endswith('.'):
            best_answer += '.'
        
        return best_answer or "The requested information could not be found in the available document sections."
    
    def _keyword_based_extraction(self, question: str, text: str) -> str:
        """Extract answer based on keyword matching"""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        sentences = re.split(r'[.!?]+', text)
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
                
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            common_words = question_words.intersection(sentence_words)
            score = len(common_words) / len(question_words) if question_words else 0
            
            scored_sentences.append((score, sentence.strip()))
        
        # Return the sentence with highest keyword overlap
        if scored_sentences:
            scored_sentences.sort(reverse=True)
            return scored_sentences[0][1]
        
        return ""
    
    def _no_context_answer(self, question: str) -> Dict[str, Any]:
        """Answer when no context is available"""
        return {
            "answer": "I could not find relevant information in the document to answer this question. Please ensure the document contains the requested information.",
            "reasoning": "No relevant document sections were retrieved for this question.",
            "confidence": "low",
            "supporting_chunks": [],
            "token_usage": 0,
            "processing_time": 0.01
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text) // 4