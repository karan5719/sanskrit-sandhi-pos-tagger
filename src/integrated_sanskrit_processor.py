"""
Integrated Sanskrit NLP Pipeline
Combines Hybrid BiLSTM Sandhi Splitter with CRF POS Tagger
For comprehensive Sanskrit text analysis and prediction
"""

import os
import sys
import pickle
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict, Counter

# Add paths
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'data'))
sys.path.insert(0, current_dir)

try:
    from tokenizer import SanskritTokenizer
    from hybrid_sandhi_splitter import HybridSandhiSplitter
    from crf_pos_tagger import CRFPOSTagger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the correct directories")
    sys.exit(1)


class IntegratedSanskritProcessor:
    """
    Integrated Sanskrit NLP pipeline combining:
    1. Hybrid BiLSTM Sandhi Splitter
    2. CRF POS Tagger
    3. Comprehensive text analysis
    """
    
    def __init__(self, 
                 bilstm_threshold: float = 0.7,
                 pos_model_path: str = None,
                 use_bilstm: bool = True):
        """
        Initialize the integrated processor.
        
        Args:
            bilstm_threshold: Threshold for BiLSTM sandhi splitting
            pos_model_path: Path to CRF POS model
            use_bilstm: Whether to use BiLSTM model
        """
        self.bilstm_threshold = bilstm_threshold
        self.use_bilstm = use_bilstm
        
        # Initialize components
        print("🔧 Initializing Integrated Sanskrit Processor...")
        
        # 1. Tokenizer
        self.tokenizer = SanskritTokenizer()
        print("✅ Tokenizer loaded")
        
        # 2. Hybrid Sandhi Splitter
        self.sandhi_splitter = HybridSandhiSplitter(
            use_bilstm=use_bilstm, 
            bilstm_threshold=bilstm_threshold
        )
        print(f"✅ Hybrid Sandhi Splitter loaded (BiLSTM: {use_bilstm}, Threshold: {bilstm_threshold})")
        
        # 3. CRF POS Tagger (load directly)
        self.crf_model = None
        if pos_model_path:
            self.crf_model = CRFPOSTagger(pos_model_path)
        else:
            # Default model path
            default_path = os.path.join(project_root, 'models', 'enhanced_comprehensive_model.pkl')
            if os.path.exists(default_path):
                self.crf_model = CRFPOSTagger(default_path)
        
        if self.crf_model and self.crf_model.is_trained:
            print("✅ CRF POS Tagger loaded")
        else:
            print("⚠️  CRF POS Tagger not available - using basic fallback")
        
        print("🎯 Integrated Processor Ready!")
    
    def _load_crf_model(self, model_path: str):
        """Load CRF POS model."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"✅ CRF model loaded from {model_path}")
            return model_data
        except Exception as e:
            print(f"❌ Error loading CRF model: {e}")
            return None
    
    def _is_sanskrit_text(self, text: str) -> Tuple[bool, float]:
        """
        Check if text contains Sanskrit characters.
        
        Args:
            text: Input text to check
            
        Returns:
            Tuple of (is_sanskrit, confidence_score)
        """
        if not text or not text.strip():
            return False, 0.0
        
        # Count Sanskrit characters
        sanskrit_chars = 0
        total_chars = 0
        
        # Sanskrit character ranges (Devanagari)
        sanskrit_ranges = [
            (0x0900, 0x097F),  # Devanagari
            (0x0951, 0x0954),  # Devanagari extended
            (0x0960, 0x0963),  # Devanagari additional
        ]
        
        # Sanskrit punctuation and symbols
        sanskrit_symbols = {'।', '॥', 'ऽ', 'ॐ', 'ं', 'ः'}
        
        for char in text.strip():
            if char.isspace() or char in '.,;:!?\'"-()[]{}':
                continue
                
            total_chars += 1
            char_code = ord(char)
            
            # Check if character is Sanskrit
            is_sanskrit = False
            for start, end in sanskrit_ranges:
                if start <= char_code <= end:
                    is_sanskrit = True
                    break
            
            if char in sanskrit_symbols:
                is_sanskrit = True
            
            if is_sanskrit:
                sanskrit_chars += 1
        
        # Calculate confidence
        if total_chars == 0:
            return False, 0.0
        
        sanskrit_ratio = sanskrit_chars / total_chars
        
        # Consider Sanskrit if at least 70% characters are Sanskrit
        is_sanskrit = sanskrit_ratio >= 0.7
        
        return is_sanskrit, sanskrit_ratio
    
    def process_text(self, text: str, 
                     split_sandhi: bool = True,
                     tag_pos: bool = True,
                     analyze_morphology: bool = True) -> Dict[str, Any]:
        """
        Process Sanskrit text comprehensively.
        
        Args:
            text: Input Sanskrit text
            split_sandhi: Whether to split sandhi compounds
            tag_pos: Whether to tag POS
            analyze_morphology: Whether to analyze morphology
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'original_text': text,
            'tokens': [],
            'sandhi_analysis': {},
            'pos_analysis': {},
            'morphology_analysis': {},
            'confidence_scores': {},
            'processing_steps': [],
            'language_check': {}
        }
        
        print(f"📝 Processing: '{text}'")
        
        # Step 0: Language Validation
        is_sanskrit, sanskrit_ratio = self._is_sanskrit_text(text)
        results['language_check'] = {
            'is_sanskrit': is_sanskrit,
            'sanskrit_ratio': sanskrit_ratio,
            'message': 'Valid Sanskrit text' if is_sanskrit else 'Please enter Sanskrit text only'
        }
        
        if not is_sanskrit:
            print(f"  ❌ Not Sanskrit text: {sanskrit_ratio:.2%} Sanskrit characters")
            results['error'] = 'Please enter Sanskrit text only. This system is designed for Sanskrit language analysis.'
            return results
        
        print(f"  ✅ Sanskrit text validated: {sanskrit_ratio:.2%} Sanskrit characters")
        
        # Step 1: Tokenization
        try:
            tokens = self.tokenizer.tokenize(text)
            results['tokens'] = tokens
            results['processing_steps'].append('tokenization')
            print(f"  1️⃣ Tokenized: {tokens}")
        except Exception as e:
            print(f"  ❌ Tokenization error: {e}")
            results['error'] = str(e)
            return results
        
        # Step 2: Sandhi Splitting (if requested)
        if split_sandhi:
            sandhi_results = self._analyze_sandhi(tokens)
            results['sandhi_analysis'] = sandhi_results
            results['processing_steps'].append('sandhi_splitting')
            
            # Update tokens with split results
            if sandhi_results.get('split_tokens'):
                results['tokens'] = sandhi_results['split_tokens']
                tokens = sandhi_results['split_tokens']
                print(f"  2️⃣ Sandhi Split: {tokens}")
        
        # Step 3: POS Tagging (if requested)
        if tag_pos and tokens:
            pos_results = self._tag_pos(tokens)
            results['pos_analysis'] = pos_results
            results['processing_steps'].append('pos_tagging')
            print(f"  3️⃣ POS Tagged: {len(pos_results.get('tagged_tokens', []))} tokens")
        
        # Step 4: Morphology Analysis (if requested)
        if analyze_morphology and tokens:
            morph_results = self._analyze_morphology(tokens, results.get('pos_analysis', {}))
            results['morphology_analysis'] = morph_results
            results['processing_steps'].append('morphology_analysis')
            print(f"  4️⃣ Morphology: {len(morph_results.get('word_analysis', []))} words analyzed")
        
        # Step 5: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(results)
        results['overall_confidence'] = overall_confidence
        
        print(f"  ✅ Complete! Overall confidence: {overall_confidence*100:.1f}%")
        return results
    
    def _analyze_sandhi(self, tokens: List[str]) -> Dict[str, Any]:
        """Analyze and split sandhi compounds."""
        sandhi_results = {
            'original_tokens': tokens,
            'split_tokens': [],
            'sandhi_operations': [],
            'confidence_scores': {},
            'methods_used': {}
        }
        
        split_tokens = []
        
        for token in tokens:
            if token in ['।', '॥', '.', ',', ';', ':', '!', '?']:
                # Punctuation - keep as is
                split_tokens.append(token)
                sandhi_results['methods_used'][token] = 'punctuation'
                continue
            
            # Try to split the token
            method, splits, confidence = self.sandhi_splitter.analyze_word(token)
            
            if splits and len(splits) > 1 and method != 'no_split':
                # Token was split
                split_tokens.extend(splits)
                sandhi_results['sandhi_operations'].append({
                    'original': token,
                    'split': splits,
                    'method': method,
                    'confidence': confidence
                })
                sandhi_results['confidence_scores'][token] = confidence
                sandhi_results['methods_used'][token] = method
            else:
                # Token was not split - still add operation for method tracking
                split_tokens.append(token)
                sandhi_results['sandhi_operations'].append({
                    'original': token,
                    'split': [token],  # No split, just the original token
                    'method': method if method != 'no_split' else 'NONE',
                    'confidence': confidence
                })
                sandhi_results['methods_used'][token] = 'no_split'
                sandhi_results['confidence_scores'][token] = 1.0
        
        sandhi_results['split_tokens'] = split_tokens
        return sandhi_results
    
    def _tag_pos(self, tokens: List[str]) -> Dict[str, Any]:
        """Tag POS for tokens."""
        pos_results = {
            'tagged_tokens': [],
            'pos_distribution': {},
            'confidence_scores': {},
            'unknown_words': []
        }
        
        try:
            # Join tokens for POS tagger
            text_for_pos = ' '.join([t for t in tokens if t not in ['।', '॥', '.', ',', ';', ':', '!', '?']])
            
            if text_for_pos.strip():
                # Try CRF model if available
                if self.crf_model:
                    tagged_sentence = self._tag_with_crf(text_for_pos)
                else:
                    # Use basic fallback POS tagging
                    tagged_sentence = self._basic_pos_tag(text_for_pos)
                
                # Parse tagged tokens
                if tagged_sentence:
                    for token_pos in tagged_sentence:
                        if isinstance(token_pos, tuple) and len(token_pos) == 2:
                            word, pos = token_pos
                            pos_results['tagged_tokens'].append((word, pos))
                            
                            # Track POS distribution
                            pos_results['pos_distribution'][pos] = pos_results['pos_distribution'].get(pos, 0) + 1
                            
                            # Track unknown words
                            if pos == 'UNKNOWN' or pos.startswith('unk'):
                                pos_results['unknown_words'].append(word)
                
                # Add punctuation back
                for token in tokens:
                    if token in ['।', '॥', '.', ',', ';', ':', '!', '?']:
                        pos_results['tagged_tokens'].append((token, 'PUNCT'))
                        pos_results['pos_distribution']['PUNCT'] = pos_results['pos_distribution'].get('PUNCT', 0) + 1
            
        except Exception as e:
            print(f"    ❌ POS tagging error: {e}")
            pos_results['error'] = str(e)
        
        return pos_results
    
    def _tag_with_crf(self, text: str) -> List[Tuple[str, str]]:
        """Tag using CRF model."""
        try:
            return self.crf_model.tag_text(text)
        except Exception as e:
            print(f"CRF tagging error: {e}")
            return self._basic_pos_tag(text)
    
    def _basic_pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """Basic fallback POS tagging."""
        words = text.split()
        tagged = []
        
        for word in words:
            pos = self._guess_pos(word)
            tagged.append((word, pos))
        
        return tagged
    
    def _guess_pos(self, word: str) -> str:
        """Basic POS guessing based on word patterns."""
        if not word:
            return 'UNKNOWN'
        
        # Handle visarga sandhi transformations
        if word.endswith('ो'):
            # Check if it's a visarga sandhi form (ः + vowel → ओ)
            base_word = word[:-1] + 'ः'  # Convert ओ to ः
            if base_word.endswith('ः'):
                return 'NOUN'  # Masculine nominative singular with visarga sandhi
            elif word in ['रामो', 'गो', 'पुरुषो', 'देवो', 'नरो', 'पुत्रो', 'शिष्यो', 'राजो', 'गुरो']:
                return 'NOUN'  # Common visarga sandhi forms
        elif word.endswith('े'):
            # Check if it's a visarga sandhi form (ः + ए → ए)
            if word in ['रामे', 'गे', 'देवे', 'नरे', 'पुत्रे', 'वने']:
                return 'NOUN'  # Locative singular with visarga sandhi
        elif word.endswith('ै'):
            # Check if it's a visarga sandhi form (ः + ऐ → ऐ)
            if word in ['रामै', 'गै', 'देवै']:
                return 'NOUN'  # Dative/ablative singular with visarga sandhi
        elif word.endswith('ौ'):
            # Check if it's a visarga sandhi form (ः + औ → औ)
            if word in ['रामौ', 'गौ', 'देवौ', 'नरौ', 'पुत्रौ']:
                return 'NOUN'  # Dual nominative/accusative with visarga sandhi
        elif word.endswith('ा'):
            # Check if it's a visarga sandhi form (ः + आ → आ)
            if word in ['रामा', 'गा', 'देवा', 'नरा', 'पुत्रा']:
                return 'NOUN'  # Feminine nominative singular with visarga sandhi
            else:
                return 'NOUN'  # General feminine ending
        elif word.endswith('ी'):
            # Check if it's a visarga sandhi form (ः + ई → ई)
            if word in ['रामी', 'देवी', 'नरी', 'पुत्री']:
                return 'NOUN'  # Feminine nominative/accusative plural with visarga sandhi
            else:
                return 'NOUN'  # General feminine ending
        
        # Check for common patterns
        if word.endswith('ः'):
            return 'NOUN'  # Masculine nominative singular
        elif word.endswith('म्'):
            return 'NOUN'  # Neuter ending
        elif word.endswith('ति') or word.endswith('न्ति'):
            return 'VERB'  # Verb endings
        elif word.endswith('स्') or word.endswith('श्च'):
            return 'VERB'  # Verb endings
        elif word in ['अस्ति', 'गच्छति', 'करोति', 'वदति', 'पश्यति']:
            return 'VERB'  # Common verbs to prevent splitting
        elif word in ['च', 'वा', 'अपि', 'तु', 'हि', 'उत', 'अथ']:
            return 'CONJ'
        elif word in ['एव', 'एवम्', 'इति', 'न']:
            return 'PART'
        elif len(word) == 1 and word in ['ऽ', 'ः', 'ं']:
            return 'PART'
        elif '्' in word and len(word) > 2:
            return 'NOUN'  # Likely a compound
        else:
            return 'UNKNOWN'
    
    def _analyze_morphology(self, tokens: List[str], pos_analysis: Dict) -> Dict[str, Any]:
        """Analyze morphology of tokens."""
        morph_results = {
            'word_analysis': [],
            'morphological_features': {},
            'compound_analysis': {}
        }
        
        # Create POS mapping
        pos_dict = {word: pos for word, pos in pos_analysis.get('tagged_tokens', [])}
        
        for token in tokens:
            if token in ['।', '॥', '.', ',', ';', ':', '!', '?']:
                continue
            
            word_analysis = {
                'word': token,
                'pos': pos_dict.get(token, 'UNKNOWN'),
                'length': len(token),
                'has_virama': '्' in token,
                'has_visarga': token.endswith('ः'),
                'has_anusvara': 'ं' in token,
                'has_avagraha': 'ऽ' in token,
                'is_compound': False,
                'components': []
            }
            
            # Check if it was split from sandhi
            sandhi_ops = pos_analysis.get('sandhi_analysis', {}).get('sandhi_operations', [])
            for op in sandhi_ops:
                if token in op['split']:
                    word_analysis['is_compound'] = True
                    word_analysis['components'] = op['split']
                    break
            
            # Get word properties from tokenizer
            try:
                word_props = self.tokenizer.get_word_properties(token)
                word_analysis.update(word_props)
            except:
                pass
            
            morph_results['word_analysis'].append(word_analysis)
        
        return morph_results
    
    def _calculate_overall_confidence(self, results: Dict) -> float:
        """Calculate overall confidence score."""
        confidences = []
        
        # Sandhi confidence
        sandhi_conf = results.get('sandhi_analysis', {}).get('confidence_scores', {})
        if sandhi_conf:
            avg_sandhi_conf = sum(sandhi_conf.values()) / len(sandhi_conf)
            confidences.append(avg_sandhi_conf)
        
        # POS confidence (simplified)
        pos_results = results.get('pos_analysis', {})
        if pos_results.get('tagged_tokens'):
            # Basic confidence based on unknown words ratio
            total_words = len([t for t, p in pos_results['tagged_tokens'] if p != 'PUNCT'])
            unknown_words = len(pos_results.get('unknown_words', []))
            pos_conf = 1.0 - (unknown_words / max(total_words, 1))
            confidences.append(pos_conf)
        
        # Overall average
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.5  # Default confidence
    
    def predict_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Main prediction method for user input.
        
        Args:
            user_input: Sanskrit text from user
            
        Returns:
            Comprehensive prediction results
        """
        print(f"🎯 PREDICTING USER INPUT: '{user_input}'")
        print("=" * 60)
        
        # Process the input
        results = self.process_text(
            text=user_input,
            split_sandhi=True,
            tag_pos=True,
            analyze_morphology=True
        )
        
        # Format results for user
        user_results = self._format_user_results(results)
        
        return user_results
    
    def _format_user_results(self, results: Dict) -> Dict[str, Any]:
        """Format results for user display."""
        formatted = {
            'input': results['original_text'],
            'summary': {
                'tokens_count': len(results['tokens']),
                'splits_performed': len(results.get('sandhi_analysis', {}).get('sandhi_operations', [])),
                'pos_tags_found': list(results.get('pos_analysis', {}).get('pos_distribution', {}).keys()),
                'overall_confidence': results.get('overall_confidence', 0) * 100
            },
            'detailed_analysis': {
                'tokens_with_pos': results.get('pos_analysis', {}).get('tagged_tokens', []),
                'sandhi_splits': results.get('sandhi_analysis', {}).get('sandhi_operations', []),
                'morphology': results.get('morphology_analysis', {}).get('word_analysis', [])
            },
            'confidence_breakdown': {
                'sandhi_confidence': results.get('sandhi_analysis', {}).get('confidence_scores', {}),
                'processing_steps': results.get('processing_steps', [])
            }
        }
        
        return formatted


# --- Demo and Testing ---
def demo_integrated_processor():
    """Demonstrate the integrated processor."""
    print("🎯 INTEGRATED SANSKRIT PROCESSOR DEMO")
    print("=" * 60)
    
    # Initialize processor
    processor = IntegratedSanskritProcessor(
        bilstm_threshold=0.7,
        use_bilstm=True
    )
    
    # Test examples
    test_inputs = [
        "रामो गच्छति",
        "नादानुस्बारयोः स्वर्येतेऽस्मात्परावेव",
        "महात्मा विष्णुरुच्यते",
        "एकनीचोऽतिप्रयत्नो दृढः",
        "तथापि चोच्चः स्याद्भक्तेर्नीचो"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n🔍 TEST {i}: {user_input}")
        print("-" * 40)
        
        results = processor.predict_user_input(user_input)
        
        # Display summary
        summary = results['summary']
        print(f"📊 Summary:")
        print(f"  Tokens: {summary['tokens_count']}")
        print(f"  Sandhi splits: {summary['splits_performed']}")
        print(f"  POS tags: {', '.join(summary['pos_tags_found'])}")
        print(f"  Confidence: {summary['overall_confidence']:.1f}%")
        
        # Display detailed results
        if results['detailed_analysis']['sandhi_splits']:
            print(f"  Sandhi operations:")
            for op in results['detailed_analysis']['sandhi_splits']:
                print(f"    {op['original']} → {op['split']} ({op['method']}, {op['confidence']*100:.1f}%)")
        
        if results['detailed_analysis']['tokens_with_pos']:
            print(f"  POS tags:")
            for word, pos in results['detailed_analysis']['tokens_with_pos']:
                print(f"    {word}: {pos}")


if __name__ == "__main__":
    demo_integrated_processor()
