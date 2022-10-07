import torch
from transformers import BertTokenizer, RobertaForSequenceClassification, AutoConfig
from kss import split_sentences
import json
import numpy as np
from utils.model import EFLContrastiveLearningModel
from types import SimpleNamespace
import os
import logging
import traceback

args = SimpleNamespace(
    sentiment_model_name_or_path='./model/saved_model/sentiment_model_7_3/STEP_1200_efl_scl_TASKsentiment_LR5e-05_WD0.1_LAMBDA0.9_POOLERcls_TEMP0.05_ACC0.8572',
    category_model_name_or_path='./model/saved_model/all_category_model_7_3/STEP_700_efl_scl_TASKcategory_LR5e-05_WD0.1_LAMBDA0.1_POOLERcls_TEMP0.1_ACC0.7056',
    model_name_or_path='./model/checkpoint-2000000',
    vocab_path='./tokenizer/version_1.9',
    method='efl_scl',
    pooler_option='cls',
    device=torch.device('cpu'),
    max_length=256,
    padding='max_length'
)

sentiment_label_descriptions = ['이것은 부정 문장입니다.', '이것은 긍정 문장입니다.']
num_sentiment_label_descriptions = len(sentiment_label_descriptions)

category_label_descriptions = ['이것은 배송에 대한 불만입니다.', '이것은 제품에 대한 불만입니다.', '이것은 처리에 대한 불만입니다.',
                               '이것은 배송 제품 처리와 관련 없는 문장입니다.']
num_category_label_descriptions = len(category_label_descriptions)

tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                          do_lower_case=False,
                                          unk_token='<unk>',
                                          sep_token='</s>',
                                          pad_token='<pad>',
                                          cls_token='<s>',
                                          mask_token='<mask>',
                                          model_max_length=args.max_length)

sentiment_model = EFLContrastiveLearningModel(args)
category_model = EFLContrastiveLearningModel(args)

sentiment_model_state_dict = torch.load(os.path.join(args.sentiment_model_name_or_path, 'model_state_dict.pt'), map_location=torch.device('cpu'))
category_sentiment_model_state_dict = torch.load(os.path.join(args.category_model_name_or_path, 'model_state_dict.pt'), map_location=torch.device('cpu'))

sentiment_model.load_state_dict(sentiment_model_state_dict)
category_model.load_state_dict(category_sentiment_model_state_dict)

sentiment_model.eval()
category_model.eval()

sentiment_model.to(args.device)
category_model.to(args.device)


def classifier(sent):
    # Split sentences
    sents = split_sentences(sent)
    num_sents = len(sents)

    # Make (sentence, label_description) pairs
    sentiment_texts = [[(s, ld) for ld in sentiment_label_descriptions] for s in sents]

    # Flatten
    sentiment_texts = sum(sentiment_texts, [])

    """
    texts = [
        (sent1, ld1), (sent1, ld2) ...
        (sent2, ld1), (sent2, ld2) ...
        (sent3, ld1), (sent3, ld2) ...
        ]
    """

    # Tokenize the texts
    sentiment_batch = tokenizer(sentiment_texts,
                                padding=args.padding,
                                max_length=args.max_length,
                                truncation=True,
                                return_token_type_ids=False,
                                )

    negative_sents = []

    with torch.no_grad():
        sentiment_batch = {k: torch.tensor(v).to(args.device) for k, v in sentiment_batch.items()}
        outputs = sentiment_model(**sentiment_batch)

        logits = outputs['logits']

        n_complaint = 0

        for idx in range(num_sents):
            # 부정
            if logits[idx * 2][1] > logits[idx * 2 + 1][1]:
                n_complaint += 1
                negative_sents.append(sents[idx])

        if n_complaint < 0.25 * num_sents:
            return 1, []

        num_negative_sents = len(negative_sents)

        # Make (sentence, label_description) pairs
        category_texts = [[(s, ld) for ld in category_label_descriptions] for s in negative_sents]

        # Flatten
        category_texts = sum(category_texts, [])

        category_batch = tokenizer(category_texts,
                                   padding=args.padding,
                                   max_length=args.max_length,
                                   truncation=True,
                                   return_token_type_ids=False,
                                   )

        category_batch = {k: torch.tensor(v).to(args.device) for k, v in category_batch.items()}
        outputs = category_model(**category_batch)

        logits = outputs['logits']

        category_score = [0, 0, 0, 0]

        for idx in range(num_negative_sents):
            preds = torch.argmax(
                logits[idx * num_category_label_descriptions: (idx + 1) * num_category_label_descriptions, 1])

            # if logits[idx * num_category_label_descriptions][0] > logits[idx * num_category_label_descriptions][1]:
            #     if logits[idx * num_category_label_descriptions + 1][0] > logits[idx * num_category_label_descriptions + 1][1]:
            #         if logits[idx * num_category_label_descriptions + 2][0] > logits[idx * num_category_label_descriptions + 2][1]:
            #             preds = 3
            category_score[preds] += 1

        complaint_category = np.argwhere(category_score == np.amax(category_score))
        complaint_category = ['0' + str(element[0]) for element in complaint_category]

    """
    is_positive :
    0 => 불만
    1 => 일반

    complaint_category :
    [] => 해당사항 X (일반 문장)
    ['00'] => 배송 불만
    ['01'] => 제품 불만
    ['02'] => 처리 불만
    ['03'] => 기타 불만 
    """

    return 0, complaint_category


if __name__ == '__main__':
    is_positive, complaint_category = classifier('안녕하세요 코아 상담사 문사입니다 무엇을 도와드릴까요 네고 혹시 아이디에서 주문했는데 제가은 이십 팔 일 날 통과했는데 지금 배송이 안 와서요 어 이십 팔 일 날요 오 구 그러시면 정확한 확인을 위해서 운송장 번호 한번 불러주시겠어요 무통장장 번호가 지금 네 뭐 어떤 거 봐야가요 어 없어요라고 판매자가 전달을 하거든요 한 관리번호 말고요 네 화물 관리번호 아니구요 그 화물 추작번호 한 아이디에서도 투자번호가 있나요 혹시 영문으로 보시면 트랙킹 한버라고 써져 있을 거예요 아 트킹 논부를 저 조회가 안되 되는데 호수 어 판매자가 다 전달하는 걸로 알고 있어요 배송 추작이 되면은 그 번호가 있다는 거 선생님하고 반품팅하고 완복할까요 맞기도 하고 때문 좀 시작적이요 그게 건가요 네 아 이게 또 그 무료배송이면 또 안 안되고 그런가요 혹시 아니 아니 그런 건 없어요 그니까니까 추작이 이제 배송사 추작은 안 안될 수가 있는데요 그 통관 과정 자체는 추적이 되거든요 이 유 패스로 지금 주 화물 조회를 해봤는데 유니 패스로 조회되면며 있는 거예요 거기에 에이 디 비엘이라고 혹시 없어요 에이치 비에 에이치 티고 비 슬라시 에 이런 식으로 써져 있거든요 에이치 비이를 안 보이는데요 화목 관리 어 유니패스 보금 아이 무인가요 아 이거 마일 넘어오는 게 언제 들어오셨는데요 좀 들어오시지 이십 팔 일이라고 하시지 않으셨어요 이 십 팔 일 날 주문해갖고 저번 달이요 네 그러니까요 그 정도다면 어느 정도까지 밀리지는 않거든요 아 그러면 왔 써야 되는 거 같은데 통과 신청 완료하려고 이 월 이십 팔 일 날 선 관세청에서 뭐 어떤 정보를 보고 계신 거예요 화물 관리번호요 그거밖에 안 나와요 조금 저희가 제가 조회한 게 엔 주문 추적에도 엔 엔으로 제 어디 그렇게 원 뭐예요 아 그래요 그럼 불러드릴까요 네 네 엔 공 공 공 네 공 여섯 개의 구 사 칠 칠 사 칠 삼 구 사 칠 칠 사 칠 삼 번 예 네 거의 성함은 어떻게 되실까요 초구현이요 주차차 구차 연차식을 해야 돼 확인 감사합니다 어 예 이십 팔 일 날에 반출 신고 강의 통관이 완료가 되신 거구요 이제 엔으로 시작하는 거는 이제 이게 그 택배가 아니라 일반 우편이라고 우편함으로 이제 배송하는 방식이시거든요 이 지금 이미 배송이 되고도 남았어야 되는데 어 저희 오만만 원도 없어가지고 계속 기다려도 이거 지금 뭐 코로나땜 거기가 선택한 그쪽 쪽에서 아 안 보여가지고 제가 그런 거 아니에요 기다리고 있 봐 저는 일 정도 돼가지고 왜 평택 강의에서 이십 팔 일 날 나온 거예요 이 나왔다고 해 네 나와서 우체국으로 인계를를 한 거고요 이제 일반 우편이라서 배송 조회가 안되시는 거구요 지금까지 혹시 못 받아보셔서 혹시 주소는 어떻게 몇 동 몇 호세요 혹시 삼백 이 동 구백 팔 호요 꼭 내 주소도 맞는데 오성로 이쪽 맞 시 오 종로 오 오 팔이요 이제 삼백 이 동 구백 팔 호 이거고 정확하게 되어있고요 어 만약에 지금까지 못 받으신 거면은 어 지금 분실로 보셔야 될 것 같고요 이게 이제 분쟁 얘기를 하시는데 이제 고객님께서 그 제 주문 방식을 의사택하시는데 이제 주문 방식에 이게 삼을 출작 가능 추적할 수 없는 이런 식으로 써져 있거든요 네 네 근데 고객님께서서 추작 가능으로 선택을 하셨는데 판매자가 임의로 출적 불가능한 걸로 보내낼 가능면서성이 있거든요 그래서 그런 경우에는 네 무조건 판매자 규책 사유로 들어가서 네 그거는 분위기계에서 대상 받을 수 있는 걸로 알고 있습니다 아 저 그 코로나가가지고 바쁘다 해야 어 과학하지만 안 온 줄 알고 그런 거 아니고 이미 배송이 됐어야 돼요 아 뭐 우체국에도 물어봐도 답이 없겠네요 이게 우체국에서도 조회하는거든요 전으로 이게 뭐 우체국으로 그 이제 택배기사가 아니라 그 막 오토바이 타고하고 편지에달하시는 부분 있잖아요 일윤 배달하시는 거거든요 아 그래요 네 뭐 문자도 안 오고 하니까 어차피 모르잖아요 맞아요 아무런 그런 게 없어요 그냥 우편함에다 넣고 끝인 그런 배송 방식이어서 배송 조회는 불가능합니다 저번에도 저 주소는 똑같이 하는데 나는 아파트로 가본 경우 두 번인가 있었거든요 아 그래요 받아거든요 근데 이제 무 우 책 물어봐도 책임을 못 준다고 하니까 전혀 넘어갔는데 그러면은 나서 딱 이렇게 되 했는데 네 그런 경우가 있으셨으면은 어 지금도 막 그런 경우가 있을 수도 있거든요 이 고자가 말 때 있는데 네 그러면 한 번 그 우체국에다가 어 어 이거 뭐 따른 데로 갔다는데 이런 식으로 한번 연락을 해보시는 것도 좋을 것 같아요 아마 이게 한 번 그랬으면 네 또 그분들이 이제 뭐 어 두 번 뭐 그러지 말라는 법 없어서 에 네 한번 그렇게라도 연락을 한번 해보시는 게 좋을 것 같습니다 네 알겠습니다 감사합니다 네 감사합니다하세요상이었습니다 좋은 하루 보내세요')
    if is_positive:
        print('일반 문장입니다.')
    else:
        print('불만 문장입니다.')
        print(f'불만 카테고리는 {complaint_category}입니다.')
