import utility
import torch
import argparse
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import BertTokenizer, BertForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('-qn', '--ques_num', type=int, default=3, help='The number of Questions.') # can replace the num of questions.
args = parser.parse_args()

ques_num = args.ques_num
questions = utility.make_question(ques_num)
qn = 0
answer = '' # save user's answer
answers = ''

while qn < len(questions):
    answer = input(f'{questions[qn]}: ')
    answers += answer
    qn += 1

# Translate answer from kor to Eng.
model = MBartForConditionalGeneration.from_pretrained('mbart_model')
tokenizer = MBart50TokenizerFast.from_pretrained('mbart_tokenizer')
tokenizer.src_lang = 'ko_KR'
encoded_hi = tokenizer(answers, return_tensors='pt')
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id['en_XX']
)
translated_input = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0] # save translated text.

filtered_input = utility.preprocess_text(translated_input)

# Run our mbti_bert model to predict user's MBTI.
tokenizer = BertTokenizer.from_pretrained('bert_tokenizer') # can replace another model as huggingface T5 or else.
model = BertForSequenceClassification.from_pretrained('bert_model')
# when run in local, we need to set map_location arg to 'cpu'
model.load_state_dict(torch.load(os.path.join('models', 'best_model.pth'), map_location=torch.device('cpu')))
model.eval()

# Ready to calculate.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
encoded_input = tokenizer.encode_plus(
    filtered_input,
    add_special_tokens=True,
    max_length=64,
    padding='max_length',
    return_attention_mask=True,
    truncation=True,
    return_tensors='pt'
)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

# Predict user's MBTI.
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
predicted_label_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]

# Print output prediction.
predicted_label = utility.idx_to_mbti(predicted_label_idx)
pred_type_demo = utility.print_demo(predicted_label)
print(f'''당신의 MBTI는 {predicted_label}이고 이 MBTI의 {pred_type_demo}''')
