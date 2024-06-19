import numpy as np
import pandas as pd
import random
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from googletrans import Translator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, DistilBertTokenizer, DistilBertForSequenceClassification

class QuestionGenerator:
    def __init__(self):
        self.questions = [
            "다른 사람들과의 상호작용을 통해 에너지를 얻는 편인가요, 아니면 혼자 시간을 보내는 것이 더 좋나요?",
            "계획을 세우고 따르는 것을 좋아하나요, 아니면 즉흥적으로 행동하는 것을 더 선호하나요?",
            "사실과 세부사항에 집중하는 편인가요, 아니면 큰 그림을 보는 것을 더 좋아하나요?",
            "논리와 이성을 바탕으로 결정을 내리는 편인가요, 아니면 감정과 가치를 중시하나요?",
            "사교적인 모임에서 쉽게 말을 거는 편인가요, 아니면 조용히 있는 것을 더 좋아하나요?",
            "다른 사람들에게 동정심을 잘 느끼는 편인가요, 아니면 객관적인 판단을 더 중시하나요?",
            "일정과 계획을 엄격히 지키는 것을 좋아하나요, 아니면 자유롭고 유연한 일정을 선호하나요?",
            "새로운 아이디어와 가능성에 대해 생각하는 것을 좋아하나요, 아니면 현실적이고 실용적인 접근을 더 중시하나요?",
            "문제를 해결할 때 감정보다 논리를 우선시하는 편인가요, 아니면 사람들의 감정을 고려하는 편인가요?",
            "상황에 따라 즉흥적으로 대처하는 것을 좋아하나요, 아니면 사전에 계획하고 준비하는 것을 더 좋아하나요?"
        ]
    
    def make_question(self, idx: int):
        """길이가 idx인 질문 목록을 만듭니다."""
        return random.sample(self.questions, idx)

class MBTIDemo:
    def __init__(self):
        self.demos = {
            'ISTJ': "별명은 '청렴결백한 논리주의자'입니다.\n ISTJ 특징은 청렴결백하면서도 실용적인 논리력과 헌신적으로 임무를 수행하는 성격으로 묘사되기도 하는 이들은, 가정 내에서뿐 아니라 법률 회사나 법 규제 기관 혹은 군대와 같이 전통이나 질서를 중시하는 조직에서 핵심 구성원 역할을 합니다. 이 유형의 사람은 자신이 맡은 바 책임을 다하며 그들이 하는 일에 큰 자부심을 가지고 있습니다.",
            'ISTP': "별명은 '만능 재주꾼'입니다.\n ISTP 특징은 타인을 잘 도우며 그들의 경험을 다른 이들과 공유하는 것을 좋아하기도 하며 특히나 그들이 아끼는 사람일수록 더욱 그러합니다. 냉철한 이성주의적 성향과 왕성한 호기심을 가진 만능재주꾼형 사람은 직접 손으로 만지고 눈으로 보면서 주변 세상을 탐색하는 것을 좋아합니다.",
            'ISFJ': "별명은 '용감한 수호자'입니다.\n ISFJ 특징은 책임감과 인내력 또한 매우 강하며, 본인의 친한 친구나 가족에게 애정이 가득합니다. 이들은 언제나 진솔하려 노력하고 가볍지 않기 때문에 관계를 맺기에 가장 믿음직스러운 유형입니다.",
            'ISFP': "별명은 '호기심 많은 예술가'입니다.\n ISFP 특징은 말없이 다정하고 온화하며 사람들에게 친절하고 상대방을 잘 알게 될 때까지 내면의 모습을 잘 드러내지 않는 경향이 있습니다. 이들은 창의적이고 감성적인 성향으로 예술적인 표현을 중요시합니다.",
            'INFJ': "별명은 '선의의 옹호자'입니다.\n INFJ 특징은 통찰력과 직관력이 뛰어나며 사람들의 동기를 잘 파악하고 이해하는 능력이 있습니다. 이들은 깊은 공감 능력을 바탕으로 타인을 돕고자 하며, 사회적 가치와 이상을 추구하는 경향이 있습니다.",
            'INFP': "별명은 '중재자'입니다.\n INFP 특징은 이상주의적이며, 깊은 내면의 가치를 중요시합니다. 이들은 창의적이고 독창적인 사고를 바탕으로 예술적 표현을 좋아하며, 타인의 감정을 잘 이해하고 공감하는 능력이 뛰어납니다.",
            'INTJ': "별명은 '전략가'입니다.\n INTJ 특징은 분석적이고 논리적인 사고를 바탕으로 문제를 해결하는 능력이 뛰어나며, 장기적인 목표를 설정하고 계획을 세우는 데 탁월합니다. 이들은 독립적이고 자기 주도적인 성향이 강합니다.",
            'INTP': "별명은 '논리적인 사색가'입니다.\n INTP 특징은 창의적이고 논리적인 사고를 바탕으로 문제를 해결하는 능력이 뛰어나며, 복잡한 아이디어와 개념을 이해하고 분석하는 데 능숙합니다. 이들은 지식과 진리를 추구하는 경향이 있습니다.",
            'ESTP': "별명은 '모험을 즐기는 사업가'입니다.\n ESTP 특징은 현실적이고 실용적인 접근을 중시하며, 상황에 따라 유연하게 대처하는 능력이 뛰어납니다. 이들은 도전적이고 활동적인 성향으로 새로운 경험을 즐깁니다.",
            'ESTJ': "별명은 '엄격한 관리자'입니다.\n ESTJ 특징은 조직적이고 계획적인 성향이 강하며, 책임감과 결단력이 뛰어납니다. 이들은 실용적이고 현실적인 접근을 중시하며, 목표 달성을 위해 체계적으로 일을 수행합니다.",
            'ESFP': "별명은 '자유로운 영혼'입니다.\n ESFP 특징은 외향적이고 사교적인 성향이 강하며, 사람들과의 상호작용을 통해 에너지를 얻습니다. 이들은 창의적이고 감성적인 성향으로 예술적 표현을 중요시합니다.",
            'ESFJ': "별명은 '친절한 사회 운동가'입니다.\n ESFJ 특징은 책임감과 인내력이 강하며, 타인을 돕고자 하는 마음이 큽니다. 이들은 사회적 가치와 이상을 추구하며, 사람들과의 관계를 중요시합니다.",
            'ENTP': "별명은 '논쟁을 즐기는 변론가'입니다.\n ENTP 특징은 창의적이고 독창적인 사고를 바탕으로 새로운 아이디어를 탐구하는 것을 즐깁니다. 이들은 논리적이고 분석적인 능력이 뛰어나며, 다양한 관점에서 문제를 바라보는 능력이 뛰어납니다.",
            'ENTJ': "별명은 '대담한 지도자'입니다.\n ENTJ 특징은 목표 지향적이며, 강력한 리더십을 발휘합니다. 이들은 전략적 사고와 계획을 통해 복잡한 문제를 해결하는 데 능숙합니다.",
            'ENFP': "별명은 '재기 발랄한 활동가'입니다.\n ENFP 특징은 창의적이고 독창적인 사고를 바탕으로 새로운 아이디어를 탐구하는 것을 즐깁니다. 이들은 열정적이고 사교적인 성향이 강하며, 사람들과의 관계를 중요시합니다.",
            'ENFJ': "별명은 '정의로운 사회 운동가'입니다.\n ENFJ 특징은 책임감과 인내력이 강하며, 타인을 돕고자 하는 마음이 큽니다. 이들은 사회적 가치와 이상을 추구하며, 사람들과의 관계를 중요시합니다."
        }

    def print_demo(self, type: str) -> str:
        """인자로 주어진 MBTI 유형의 데모를 출력합니다."""
        return self.demos.get(type, "Invalid MBTI type")

class MBTIConverter:
    def __init__(self):
        self.mbti_list = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']

    def idx_to_mbti(self, idx: int) -> str:
        """Label Encoder의 inverse_transform 함수와 같은 작업을 수행합니다."""
        if 0 <= idx < len(self.mbti_list):
            return self.mbti_list[idx]
        else:
            return "Invalid index"

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """URL과 비알파벳 문자를 제거하고, 소문자로 변환한 후 단어로 분리, nltk 도구를 이용해 전처리합니다."""
        text = re.sub(r'[^\\w\\s]', '', text)
        words = text.split()
        cleaned_text = [self.lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in self.stop_words]
        return ' '.join(cleaned_text)

class TranslatorService:
    def __init__(self):
        self.translator = Translator()

    def translate_text(self, text: str, src: str = 'en', dest: str = 'ko') -> str:
        """Google Translate API를 사용하여 텍스트를 번역합니다."""
        translated = self.translator.translate(text, src=src, dest=dest)
        return translated.text


class ModelSaver:
    def save_models(self):
        mbart_model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=16)

        mbart_model.save_pretrained("mbart_model")
        mbart_tokenizer.save_pretrained("mbart_tokenizer")
        distilbert_tokenizer.save_pretrained("distilbert_tokenizer")
        distilbert_model.save_pretrained("distilbert_model")

if __name__ == '__main__':
    question_generator = QuestionGenerator()
    mbti_demo = MBTIDemo()
    mbti_converter = MBTIConverter()
    text_preprocessor = TextPreprocessor()
    translator_service = TranslatorService()
    model_saver = ModelSaver()

