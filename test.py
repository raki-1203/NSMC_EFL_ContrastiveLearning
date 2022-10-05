# %%
import torch
from transformers import RobertaModel, BertTokenizer
from utils.model import EFLContrastiveLearningModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='./model/checkpoint-2000000')
parser.add_argument('--vocab_path', type=str, default='./tokenizer/version_1.9')
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--pooler_option', type=str, default='base')

args = parser.parse_args(args=[])

model = EFLContrastiveLearningModel(args=args)
tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                                  do_lower_case=False,
                                                  unk_token='<unk>',
                                                  sep_token='</s>',
                                                  pad_token='<pad>',
                                                  cls_token='<s>',
                                                  mask_token='<mask>',
                                                  model_max_length=args.max_length)

# %%
batch = [
'수고 많으세요 문의 좀 드릴게요 신속하게 답변 드리겠습니다 신규 가입 시 무료 배송 혜택은 어떻게 사용하나요 저희 온라인 몰에서 신규 가입 시 주문 건 십 회 배송비가 면제됩니다 주문창에서 배송비 선택 창에 보시면 무료 쿠폰이 생성되어 있습니다 그럼 주문 마지막 단계에서 쿠폰이 있다는 거죠 네 그렇습니다 고객님 네 알겠습니다 친절하게 설명해 주셔서 감사합니다 더 문의 하실 내용이나 궁금한 점이 있으실까요 아니요 괜찮습니다 수고하세요',
'안녕하세요 궁금한 게 있어서 연락드립니다 네 무엇을 도와드릴까요 주문한 상품이 상품 준비 중으로 표시되고 있는데요 상품 정상적으로 수령 가능한가요 네 문의하신 내용에 답변을 드리겠습니다 상품이 인도장에 도착하기 전까지는 주문 상태가 상품 준비 중으로 표시되요 상품은 평균적으로 출국 두 세시간 전 인도장에 도착하니까요 출국일 당일 인도장에 방문하시면 정상적으로 수령할 수 있으세요 친절한 안내 감사합니다',
'수고 많으십니다 다름이 아니라 궁금한 게 있어서요 네 어떤 궁금한 점으로 연락을 주셨나요 예 제가 제품 주문을 하려고 하는데요 네 그러세요 고객님 계속 말씀해 주세요 수령자 배송지 주소를 먼저 변경하고 싶어서요 실시간 상담 또는 문자 상담을 통해 요청이 가능합니다 알겠습니다 친절한 설명 감사합니다 예 좋은 하루 보내겠습니다 감사합니다',
'근데 아직도 배송 준비 중이네요 고객님 우선 저희도 좀 확인해보도록 하겠습니다 성함이랑 연락처 좀 먼저 말씀 부탁드리겠습니다 확인 감사드리구요 잠시만 좀 기다려 주시겠습니까 고객님 제가 우선 이거 업체 측으로 한 번 먼저 전화를 좀 해볼 건데요 금일은 늦어서 오늘 바로 연결이 좀 안 될 수가 있습니다 우선은 전화를 해보고 문자로 답변 남기도록 하겠습니다 고객님 네 그러세요 이 번호로 주세요 네 고객님 확인하고 연락드리겠습니다',
'안녕하세요 무엇을 도와드릴까요 제가 해외배송으로 주문을 했는데요 네 어떤 부분이 궁금하실까요 사은품이 배송이 되는지가 궁금하네요 해외배송을 선택하는 경우 사은품은 배송되지 않습니다 왜 같은 물건을 시켰는데 사은품이 안 오나요 국가 세관당국이 판단하는 세금을 사은품은 측정하기가 어렵습니다 그거랑 무슨 상관이 있나요 관세가 부관될 수 있고 통관 등의 문제로 배송하지 않는 걸 원칙으로 합니다 네 알려주셔서 감사합니다',
'내가 저 제품을 구매를 했었는데요 몇 가지 같이 했는 거 중에서 한 가지가 안 와서요 그거 확인을 좀 부탁합니다 죄송합니다 고객님 주문자분 성함이랑 휴대폰 번호 말씀해주시겠습니까 확인 감사합니다 혹시 어떤 상품 지금 못 받아보신 걸까요 예 죄송합니다 고객님 확인해보니까 오늘 오후 네 시에서 여섯 시 사이 배송 예정이라고 합니다 그런데 내가 다른 상품은 받아놓고 나니까 여기에서 이제 문자가 왔어요 네 그러셨군요 고객님 예 안 보냈으면 안 보낸 걸로 문자를 남겨놔야지 똑같이 다 한꺼번에 왔다 하니깐 헷갈리지 전화도 어렵게 한 십 분 이상 기다려 가지고 이제 통화 연결됐는데 말이에요 앞으로 주의 좀 시켜주세요 그런거는요 담당 부서로 개선 요청 드리겠습니다 예 알겠습니다 수고하세요 네 불편드려 죄솝합니다 감사합니다',
'예 안녕하세요 이제야 통화가 됐네요 많이 기다렸어요 죄송합니다 고객님 무엇을 도와드릴까요 제가 다름이 아니라 나이가 많아서요 육십이 넘었어요 그래서 컴퓨터를 못 다뤄요 그래서 전화했어요 네 고객님 어떤 도움이 필요하세요 네 고객님 지금 컴퓨터 사용 가능하신가요 예 바로 앞에 앉아 있어요',
'네 고객님 잠시만요 기다려주셔서 감사합니다 아직 확인되지 않았습니다 그래요 확인이 좀 늦네요 왜냐하면 송장 번호가 있으면 바로 확인이 되는데 송장번호가 없어서요 이름 전화번호로도 확인이 안 되나요 네 배송 업체 쪽으로 전화해서 확인을 요청했는데 아직 답변이 없습니다 택배기사분이 송장을 주고 가든가 했어야 되는데 답답하네요 네 그 보통은 현관 앞에 둬도 상품 가져가고 송장은 놔두고 가시는데 근데 그게 전혀 없어요 어쨌든 그 없어도 뭐 수거해가셨다 하시니까 일단 저희가 확인해볼게요 예 그렇게 해주시겠어요 만약에 오늘 확인 안 되면은 제가 내일 오전까지 좀 연락드릴게요 고객님 네 이 번호로 전화 주시면 됩니다 예 알겠습니다 고객님 불편드려 죄송합니다 네 늦어도 내일 오전까지는 연락 주세요 수고하세요'
]
batch = tokenizer(batch, 
                  padding=True,
                  return_token_type_ids=False,
                  return_tensors='pt'
                  )

# %%
output = model(**batch)
features = output['embeddings']

# %%

features = torch.unsqueeze(features, 1)
labels = torch.Tensor([1,1,0,1,0,1,1,0])

temperature = 1

contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
print(contrast_feature.shape)
anchor_feature = contrast_feature

batch_size = features.shape[0]
anchor_count = features.shape[1]
contrast_count = features.shape[1]
anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)

anchor_dot_contrast

# anchor_dot_contrast : (BS, BS)

logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
logits = anchor_dot_contrast - logits_max.detach()

labels = labels.contiguous().view(-1, 1)
mask = torch.eq(labels, labels.T).float()
print(mask)
mask = mask.repeat(anchor_count, contrast_count)
logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1),
            0
        )
print(logits_mask)
# logit_mask 는 diagonal 빼고 다 1

logits_mask
mask = mask * logits_mask
print(mask)

exp_logits = torch.exp(logits) * logits_mask
log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

loss = - mean_log_prob_pos
print(loss)

loss = loss.view(anchor_count, batch_size).mean()
print(loss)


