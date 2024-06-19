import argparse
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import subprocess
import warnings

from utility import QuestionGenerator, MBTIDemo, MBTIConverter, TextPreprocessor, TranslatorService

# 경고 메시지 무시
warnings.filterwarnings('ignore')

class MBTIPredictor:
    def __init__(self, ques_num):
        self.ques_num = ques_num
        self.question_generator = QuestionGenerator()
        self.mbti_demo = MBTIDemo()
        self.mbti_converter = MBTIConverter()
        self.text_preprocessor = TextPreprocessor()
        self.translator_service = TranslatorService()

    def get_user_answers(self):
        questions = self.question_generator.make_question(self.ques_num)
        answers = ''
        for question in questions:
            answer = input(f'{question}: ')
            answers += answer
        return answers

    def translate_answers(self, answers):
        return self.translator_service.translate_text(answers, src='ko', dest='en')

    def preprocess_answers(self, translated_answers):
        return self.text_preprocessor.preprocess_text(translated_answers)

    def predict_mbti(self, preprocessed_text):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=16)

        state_dict = torch.load(os.path.join('models', 'bestmodel.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        encoded_input = tokenizer.encode_plus(
            preprocessed_text,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        predicted_label_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]

        return self.mbti_converter.idx_to_mbti(predicted_label_idx)

    def run(self):
        user_answers = self.get_user_answers()
        translated_answers = self.translate_answers(user_answers)
        preprocessed_answers = self.preprocess_answers(translated_answers)
        predicted_mbti = self.predict_mbti(preprocessed_answers)
        mbti_demo = self.mbti_demo.print_demo(predicted_mbti)
        print(f'당신의 MBTI는 {predicted_mbti}이고 이 MBTI의 {mbti_demo}')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(['python', os.path.join(current_dir, 'scrapping/get_playlist.py'), predicted_mbti])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MBTI 예측 프로그램')
    parser.add_argument('--ques_num', type=int, default=5, help='질문의 수')
    args = parser.parse_args()

    mbti_predictor = MBTIPredictor(args.ques_num)
    mbti_predictor.run()
