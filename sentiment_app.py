# -*- coding:utf-8 -*-
import json
from unittest import result
from flask import Flask, jsonify, request, make_response

# module import 예시
from utils.model import EFLContrastiveLearningModel
import os
import torch
from types import SimpleNamespace
from transformers import BertTokenizer
from kss import split_sentences
from konlpy.tag import Mecab

device = torch.device('cuda:0')

args = SimpleNamespace(
                        model_name_or_path='./model/checkpoint-2000000',
                        vocab_path='/home/tmax/kibong_choi/EFL_ContrastiveLearning/tokenizer/version_1.9',
                        method='efl_scl',
                        pooler_option='cls'
                        )
tokenizer = tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                                do_lower_case=False,
                                                unk_token='<unk>',
                                                sep_token='</s>',
                                                pad_token='<pad>',
                                                cls_token='<s>',
                                                mask_token='<mask>',
                                                model_max_length=256)

model_path = 'model/saved_model/STEP_400_efl_scl_TASKsentiment_LR5e-05_WD0.1_LAMBDA0.9_POOLERcls_TEMP0.05_ACC0.8278'
model = EFLContrastiveLearningModel(args)
model_state_dict = torch.load(os.path.join(model_path, 'model_state_dict.pt'))    
model.load_state_dict(model_state_dict)
model.eval()
model.to(device)

mecab = Mecab()

label_descriptions = ['이것은 부정 문장입니다.', '이것은 긍정 문장입니다.']

def classifier(sent):
    sents = split_sentences(sent)
    num_sents = len(sents)
    
    # Make (sentence, label_description) pairs
    texts = [[(s, ld) for ld in label_descriptions] for s in sents]
    # Flatten
    texts = sum(texts, [])
    
    """
    texts = [
        (sent1, ld1), (sent1, ld2) ...
        (sent2, ld1), (sent2, ld2) ...
        (sent3, ld1), (sent3, ld2) ...
        ]
    """
    batch = tokenizer(texts,
                       padding=True,
                       return_tensors='pt',
                       return_token_type_ids=False,
                       )
    batch = {k: v.to(device) for k, v in batch.items()}

    output = model(**batch)
    logits = output['logits'].cpu()
    
    ret = {}
    
    for idx in range(num_sents):
        if logits[idx * 2][1] > logits[idx * 2 + 1][1]:
            result = '부정'
        else:
            result = '긍정'
            
        # print(sents[idx])
        # print(f'{label_descriptions[0]}\tNOT ENTAIL : {logits[idx * 2][0]:.4f}\tENTAIL : {logits[idx * 2][1]:.4f}')
        # print(f'{label_descriptions[1]}\tNOT ENTAIL : {logits[idx * 2 + 1][0]:.4f}\tENTAIL : {logits[idx * 2 + 1][1]:.4f}')
        # print(f'RESULT : {result}')
        
        ret[idx] = {}
        ret[idx]['sent'] = sents[idx]
        ret[idx][label_descriptions[0]] = f'NOT ENTAIL : {logits[idx * 2][0]:.4f} ENTAIL : {logits[idx * 2][1]:.4f}'
        ret[idx][label_descriptions[1]] = f'NOT ENTAIL : {logits[idx * 2 + 1][0]:.4f} ENTAIL : {logits[idx * 2 + 1][1]:.4f}'
        ret[idx]['reslut'] = result

    return ret


# Flask
app = Flask(__name__)

# POST method Example
@app.route("/sentiment", methods=["POST"])
def sentiment():
		# 웹서버가 에러가 나더라도 계속 구동되도록 하는 try-except 구문
    try:
        # 입력 텍스트 정제 + 문장 줄바꿈 단위로 바꾸기
        data = request.get_json(force=True, silent=True)
        sent = data["sent"].strip()
        ...
        # AI 모델을 통한 inference
        results = classifier(sent)

        # json response wrapping
        json_results = results

        # http format response
        resp = make_response(json.dumps(json_results, ensure_ascii=False))
        resp.headers["Content-Type"] = "application/json"

        return resp

    except Exception as e:

        resp = make_response("ERROR", 400)
        resp.headers["Content-Type"] = "application/json"

        return resp

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=23899, debug=True)
    
    import csv
    
    positive_file = 'pos_call.csv'    
    negative_file = 'neg_call.csv'
    
    p = open(positive_file, 'w')
    cr_positive = csv.writer(p)
    
    n = open(negative_file, 'w')
    cr_negative = csv.writer(n)

    
    path = '/home/tmax/kibong_choi/cs_sharing/data'

    for folder in os.listdir(path):
        if 'DS_Store' in folder:
            continue
        
        for file in os.listdir(os.path.join(path, folder)):
            file_path = os.path.join(path, folder, file)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                text = lines[0]
                
                try:
                    ret = classifier(text)
                except:
                    pass
                if len(ret) == 0:
                    continue
                
                infer_result = []
                
                for key in ret.keys():
                    if ret[key]['reslut'] == '부정':
                        infer_result.append(0)
                    elif ret[key]['reslut'] == '긍정':
                        infer_result.append(1)
                
                if sum(infer_result) == len(infer_result):
                    # print('긍정', text)
                    # positive_calls.append((file_path, text))
                    cr_positive.writerow((os.path.join(folder, file), text))
                
                elif sum(infer_result) < 0.8 * len(infer_result):
                    # print(f'부정 전체 문장수: {len(infer_result)} 부정 문장 수: {len(infer_result) - sum(infer_result)}')
                    # print(text)
                    cr_negative.writerow((os.path.join(folder, file), len(infer_result), len(infer_result) - sum(infer_result), text))

                    # negative_calls.append((file_path, text))
    p.close()
    n.close()