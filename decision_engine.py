from typing import Dict, List, Any
import logging
from llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    AI-powered decision engine that combines semantic search results 
    with LLM reasoning for explainable decisions
    """
    
    def __init__(self, llm_processor: LLMProcessor):
        self.llm_processor = llm_processor
        logger.info("Initialized Decision Engine")
    
    def generate_answer(self, question: str, parsed_query: Dict[str, Any], 
                       context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive answer with explainable reasoning
        """
        try:
            # Step 1: Analyze question intent and context quality
            analysis = self._analyze_context_quality(question, context_chunks)
            
            # Step 2: Try LLM first, but have robust fallbacks
            llm_result = None
            try:
                llm_result = self.llm_processor.generate_answer(
                    question=question, 
                    context="", 
                    retrieved_chunks=context_chunks
                )
            except Exception as e:
                logger.warning(f"LLM processing failed, using enhanced fallback: {str(e)}")
                llm_result = self._enhanced_fallback_answer(question, context_chunks)
            
            # Step 3: Post-process and enhance the answer
            enhanced_answer = self._enhance_answer(
                question=question,
                llm_result=llm_result,
                parsed_query=parsed_query,
                context_chunks=context_chunks,
                analysis=analysis
            )
            
            return enhanced_answer
            
        except Exception as e:
            logger.error(f"Error in decision engine: {str(e)}")
            return self._enhanced_fallback_answer(question, context_chunks)
    
    def _analyze_context_quality(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality and relevance of retrieved context
        """
        if not chunks:
            return {
                "quality": "poor",
                "relevance": 0.0,
                "coverage": "incomplete",
                "missing_info": ["No relevant document sections found"]
            }
        
        # Calculate average relevance score
        avg_relevance = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        # Analyze coverage
        total_length = sum(len(chunk['text']) for chunk in chunks)
        
        # Determine quality rating
        if avg_relevance > 0.8 and total_length > 1000:
            quality = "excellent"
        elif avg_relevance > 0.6 and total_length > 500:
            quality = "good"
        elif avg_relevance > 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "relevance": avg_relevance,
            "coverage": "complete" if total_length > 800 else "partial",
            "chunk_count": len(chunks),
            "total_context_length": total_length
        }
    
    def _enhanced_fallback_answer(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced fallback answer when LLM processing fails - uses rule-based logic
        """
        if not chunks:
            return {
                "answer": "I could not find relevant information in the document to answer your question.",
                "reasoning": "No relevant chunks retrieved from the document.",
                "confidence": "low",
                "supporting_chunks": [],
                "token_usage": 0
            }
        
        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        best_chunk = sorted_chunks[0]
        
        # Try to extract answer using pattern matching and rules
        answer = self._extract_answer_with_rules(question, best_chunk['text'])
        
        if not answer:
            # Fallback to chunk summary
            answer = f"Based on the document: {best_chunk['text'][:300]}..."
        
        return {
            "answer": answer,
            "reasoning": f"Answer extracted from document section with {best_chunk.get('relevance_score', 0):.1%} relevance match using rule-based processing.",
            "confidence": self._assess_confidence_from_score(best_chunk.get('relevance_score', 0)),
            "supporting_chunks": [chunk['id'] for chunk in sorted_chunks[:3]],
            "token_usage": 0
        }
    
    def _extract_answer_with_rules(self, question: str, text: str) -> str:
        """
        Extract answers using rule-based pattern matching
        """
        import re
        
        question_lower = question.lower()
        text_lower = text.lower()
        
        # Common question patterns and extraction rules
        patterns = {
            'waiting period': r'waiting period[^\d]*(\d+)\s*(month|year|day)s?',
            'grace period': r'grace period[^\d]*(\d+)\s*(day|month)s?',
            'coverage': r'(cover|coverage)[^.]*[.!]',
            'percentage': r'(\d+)%',
            'amount': r'(\d+(?:,\d{3})*)\s*(rupee|dollar|\$|₹)',
            'definition': r'defined as[^.]*[.]',
            'benefit': r'benefit[^.]*[.]',
            'condition': r'condition[^.]*[.]'
        }
        
        # Check which pattern matches the question
        for pattern_name, pattern in patterns.items():
            if pattern_name in question_lower:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Find the sentence containing this match
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if match.group(0).lower() in sentence.lower():
                            return sentence.strip() + "."
        
        # If no specific pattern matches, look for direct question keywords
        question_keywords = re.findall(r'\b\w+\b', question_lower)
        best_sentence = ""
        max_keyword_matches = 0
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_matches = sum(1 for keyword in question_keywords if keyword in sentence_lower)
            if keyword_matches > max_keyword_matches:
                max_keyword_matches = keyword_matches
                best_sentence = sentence.strip()
        
        return best_sentence + "." if best_sentence else ""
    
    def _assess_confidence_from_score(self, relevance_score: float) -> str:
        """Assess confidence from relevance score"""
        if relevance_score > 0.8:
            return "high"
        elif relevance_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _enhance_answer(self, question: str, llm_result: Dict[str, Any], 
                       parsed_query: Dict[str, Any], context_chunks: List[Dict[str, Any]],
                       analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the LLM-generated answer with additional structured information
        """
        # Extract key information from the answer
        answer_text = llm_result.get("answer", "")
        
        # Add structured metadata
        enhanced_result = {
            "answer": answer_text,
            "metadata": {
                "question_intent": parsed_query.get("intent", "unknown"),
                "domain": parsed_query.get("domain", "unknown"),
                "context_quality": analysis["quality"],
                "confidence_level": llm_result.get("confidence", "medium"),
                "processing_stats": {
                    "chunks_analyzed": len(context_chunks),
                    "avg_relevance": analysis["relevance"],
                    "token_usage": llm_result.get("token_usage", 0)
                }
            },
            "supporting_evidence": self._extract_supporting_evidence(context_chunks),
            "reasoning_chain": self._build_reasoning_chain(question, context_chunks, answer_text),
            "limitations": self._identify_limitations(analysis, parsed_query)
        }
        
        return enhanced_result
    
    def _extract_supporting_evidence(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and structure supporting evidence from context chunks
        """
        evidence = []
        
        for i, chunk in enumerate(chunks[:3]):  # Top 3 most relevant
            evidence.append({
                "chunk_id": chunk.get("id", i),
                "relevance_score": chunk.get("relevance_score", 0),
                "excerpt": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                "length": len(chunk["text"]),
                "rank": i + 1
            })
        
        return evidence
    
    def _build_reasoning_chain(self, question: str, chunks: List[Dict[str, Any]], 
                              answer: str) -> List[Dict[str, str]]:
        """
        Build a step-by-step reasoning chain for explainability
        """
        reasoning_steps = []
        
        # Step 1: Question analysis
        reasoning_steps.append({
            "step": "Question Analysis",
            "description": f"Analyzed the question: '{question[:100]}...' to understand the information requirement."
        })
        
        # Step 2: Context retrieval
        if chunks:
            reasoning_steps.append({
                "step": "Context Retrieval", 
                "description": f"Retrieved {len(chunks)} relevant document sections using semantic search."
            })
        
        # Step 3: Evidence evaluation
        if chunks:
            best_chunk = max(chunks, key=lambda x: x.get('relevance_score', 0))
            reasoning_steps.append({
                "step": "Evidence Evaluation",
                "description": f"Found most relevant information with {best_chunk.get('relevance_score', 0):.2%} confidence match."
            })
        
        # Step 4: Answer synthesis
        reasoning_steps.append({
            "step": "Answer Synthesis",
            "description": "Combined evidence from multiple document sections to provide comprehensive answer."
        })
        
        return reasoning_steps
    
    def _identify_limitations(self, analysis: Dict[str, Any], 
                            parsed_query: Dict[str, Any]) -> List[str]:
        """
        Identify potential limitations in the answer
        """
        limitations = []
        
        if analysis["quality"] == "poor":
            limitations.append("Limited relevant information found in the document")
        
        if analysis["relevance"] < 0.5:
            limitations.append("Low confidence in document relevance to the question")
        
        if analysis.get("chunk_count", 0) < 2:
            limitations.append("Answer based on limited document sections")
        
        if parsed_query.get("complexity") == "complex" and analysis["quality"] != "excellent":
            limitations.append("Complex question may require additional context not available in the document")
        
        return limitations