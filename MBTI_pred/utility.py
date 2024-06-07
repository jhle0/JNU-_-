import numpy as np
import pandas as pd
import typing
import random 
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BertTokenizer, BertForSequenceClassification

def make_question(idx:int):
    '''
    Make question list. Length of List is idx(arg).
    '''
    Questions = ['Q1', 
                 'Q2',
                 'Q3',
                 'Q4',
                 'Q5',
                 'Q6',
                 'Q7',
                 'Q8',
                 'Q9',
                 'Q10',
                 ]
    
    return random.sample(Questions, idx)

def print_demo(type:str) -> str:
    '''
    Output type(arg)'s MBTI demo.
    '''
    demos = {'ISTJ': '''별명은 "청렴결백한 논리주의자"입니다. \n ISTJ 특징은 청렴결백하면서도 실용적인 논리력과 헌신적으로 임무를 수행하는 
성격으로 묘사되기도 하는 이들은, 가정 내에서뿐 아니라 법률 회사나 법 규제 기관 혹은 군대와 같이 전통이나 질서를 중시하는 
조직에서 핵심 구성원 역할을 합니다. 이 유형의 사람은 자신이 맡은 바 책임을 다하며 그들이 하는 일에 큰 자부심을 가지고 
있습니다.''',
    'ISTP' : '''별명은 "만능 재주꾼"입니다. \n ISTP 특징은 타인을 잘 도우며 그들의 경험을 다른 이들과 공유하는 것을 좋아하기도 하며
특히나 그들이 아끼는 사람일수록 더욱 그러합니다. 냉철한 이성주의적 성향과 왕성한 호기심을 가진 만능재주꾼형 사람은 직접 
손으로 만지고 눈으로 보면서 주변 세상을 탐색하는 것을 좋아합니다.''',
    'ISFJ' : '''별명은 "용감한 수호자"입니다. \n ISFJ 특징은 책임감과 인내력 또한 매우 강하며, 본인의 친한 친구나 가족에게 애정이 가득합니다. 
이들은 언제나 진솔하려 노력하고 가볍지 않기 때문에 관계를 맺기에 가장 믿음직스러운 유형입니다.''',
    'ISFP' : '''별명은 "호기심 많은 예술가"입니다. \n ISFP 특징은 말없이 다정하고 온화하며 사람들에게 친절하고 상대방을 잘 알게 될 때까지 내면의 
모습이 잘 보이지 않습니다. 의견 충돌을 피하고, 인화를 중시합니다. 인간과 관계되는 일을 할 때 자신의 감정과 타인의 감정에 
지나치게 민감한 경향이 있습니다. 이들은 결정력과 추진력을 기를 필요가 있습니다.''',
    'INTJ' : '''별명은 "용의주도한 전략가"입니다. \n INTJ 특징은 대개 친구들 사이에서는 놀림의 표현임에도 불구하고 전혀 개의치 않아 하며, 
오히려 깊고 넓은 지식을 가지고 있는 그들 자신에게 남다른 자부심을 느낍니다. 이들은 또한 관심 있는 특정 분야에 대한 
그들의 방대한 지식을 다른 이들과 공유하고 싶어 하기도 합니다.''',
    'INTP' : '''별명은 "논리적인 사색가"입니다. \n INTP 특징은 조용하고 과묵하며 논리와 분석으로 문제를 해결하기를 좋아합니다. 먼저 대화를 
시작하지 않는 편이나 관심이 있는 분야에 대해서는 말을 많이 하는 편입니다. 이해가 빠르고 직관력으로 통찰하는 능력이 있으며 
지적 호기심이 많아, 분석적이고 논리적입니다''',
    'INFJ' : '''별명은 "선의의 옹호자"입니다. \n INFJ 특징은 인내심이 많고 통찰력과 직관력이 뛰어나며 화합을 추구합니다. 창의력이 좋으며, 성숙한 
경우엔 강한 직관력으로 타인에게 말없이 영향력을 끼칩니다. 독창성과 내적 독립심이 강하며, 확고한 신념과 열정으로 자신의 
영감을 구현시켜 나가는 정신적 지도자들이 많습니다.''',
    'INFP' : '''별명은 "열정적인 중재자"입니다. \n INFP 특징은 최악의 상황이나 악한 사람에게서도 좋은 면만을 바라보며 긍정적이고 더 나은 상황을 
만들고자 노력하는 진정한 이상주의자입니다. 간혹 침착하고 내성적이며 심지어는 수줍음이 많은 사람처럼 비추어지기도 하지만, 
이들 안에는 불만 지피면 활활 타오를 수 있는 열정의 불꽃이 숨어있습니다.''',
    'ESTJ' : '''별명은 "엄격한 관리자"입니다. \n ESTJ 특징은 그들 생각에 반추하여 무엇이 옳고 그른지를 따져 사회나 가족을 하나로 단결시키기 위해 
사회적으로 받아들여지는 통념이나 전통 등 필요한 질서를 정립하는 데 이바지하는 대표적인 유형입니다. 정직하고 헌신적이며 위풍당당한 
이들은 비록 험난한 가시밭길이라도 조언을 통하여 그들이 옳다고 생각하는 길로 사람들을 인도합니다.''',
    'ESTP' : '''별명은 "모험을 즐기는 사업가"입니다. \n ESTP 특징은 여러 사람이 모인 행사에서 이 자리 저 자리 휙휙 옮겨 다니는 무리 중에서 어렵지 않게 
찾아볼 수 있습니다. 직설적이면서도 친근한 농담으로 주변 사람을 웃게 만드는 이들은 주변의 이목을 끄는 것을 좋아합니다.''',
    'ESFJ' : '''별명은 "사교적인 외교관"입니다. \n ESFJ 특징은 보편적인 성격 유형으로, 이를 미루어 보면 왜 이 유형의 사람이 인기가 많은지 이해가 갑니다.
이들은 분위기를 좌지우지하며 여러 사람의 스포트라이트를 받거나 학교에 승리와 명예를 불러오도록 팀을 이끄는 역할을 하기도 합니다.
이들은 또한 훗날 다양한 사교 모임이나 어울림을 통해 주위 사람들에게 끊임없는 관심과 애정을 보임으로써 다른 이들을 행복하고 
즐겁게 해주고자 노력합니다.''',
    'ESFP' : '''별명은 "자유로운 영혼의 연예인"입니다. \n ESFP 특징은 순간의 흥분되는 감정이나 상황에 쉽게 빠져들며, 주위 사람들 역시 그런 느낌을 만끽
하기를 원합니다. 다른 이들을 위로하고 용기를 북돋아 주는 데 이들보다 더 많은 시간과 에너지를 소비하는 사람 없을 겁니다.''',
    'ENFJ' : '''별명은 "정의로운 사회운동가"입니다. \n ENFJ 특징은 따뜻하고 적극적이며 책임감이 강하고 사교성이 풍부하고 동정심이 많습니다. 
상당히 이타적이고 민첩하고 인화를 중요시하며 참을성이 많으며, 다른 사람들의 생각이나 의견에 진지한 관심을 가지고 공동선을 위하여 
다른 사람의 의견에 대체로 동의합니다. 미래의 가능성을 추구하며 편안하고 능란하게 계획을 제시하고 집단을 이끌어가는 능력이 있습니다.''',
    'ENTJ' : '''별명은 "대담한 통솔자"입니다. \n ENTJ 특징은 넘치는 카리스마와 자신감으로 공통의 목표 실현을 위해 다른 이들을 이끌고 진두지휘합니다.
예민한 성격의 사회운동가형 사람과 달리 이들은 진취적인 생각과 결정력, 그리고 냉철한 판단력으로 그들이 세운 목표 달성을 위해 
가끔은 무모하리만치 이성적 사고를 하는 것이 특징입니다. ''',
    'ENTP' : '''별명은 "뜨거운 논쟁을 즐기는 변론가"입니다. \n ENTP 특징은 변론가형 사람은 타인이 믿는 이념이나 논쟁에 반향을 일으킴으로써 군중을 선동
하는 일명 선의의 비판자입니다. 결단력 있는 성격 유형이 논쟁 안에 깊이 내재한 숨은 의미나 상대의 전략적 목표를 꼬집기 위해 
논쟁을 벌인다고 한다면, 변론가형 사람은 단순히 재미를 이유로 비판을 일삼습니다.''',
    'ENFP' : '''별명은 "재기발랄한 활동가"입니다. \n ENFP 특징은 자유로운 사고의 소유자입니다.종종 분위기 메이커 역할을 하기도 하는 이들은 단순한 
인생의 즐거움이나 그때그때 상황에서 주는 일시적인 만족이 아닌 타인과 사회적, 정서적으로 깊은 유대 관계를 맺음으로써 행복을 
느낍니다'''}

    return demos[type]

def idx_to_mbti(idx:int) -> str:
    '''
    Same operation as label encoder's inverse_transform function.
    '''
    mbti_list = ['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP','INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP']
    return mbti_list[idx]

def preprocess_text(text):
    '''
    Preprocessing mbti_bert's input text by removing URLs and non-alphanumeric characters, Converting to lowercase,
    spliting into words, executing nltk tools operation.
    '''
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_text = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]

    return ' '.join(cleaned_text)

if __name__ == '__main__':
    # Save BERT for translation and for predict MBTI to reduce main.py's running time.
    mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)

    mbart_model.save_pretrained("mbart_model")
    mbart_tokenizer.save_pretrained("mbart_tokenizer")
    bert_tokenizer.save_pretrained("bert_tokenizer")
    bert_model.save_pretrained("bert_model")
