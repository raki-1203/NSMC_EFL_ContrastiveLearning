std_label_table = {
    'shipping': 0,
    'product': 1,
    'processing': 2
}

efl_category_label_descriptions = {
    'shipping': '이것은 배송 관련 문장입니다.',
    'product': '이것은 제품 관련 문장입니다.',
    'processing': '이것은 처리 관련 문장입니다.'
}

scl_label_table = {
    'shipping': {
        'shipping': 0,
        'product': 1,
        'processing': 2
    },
    'product': {
        'shipping': 3,
        'product': 4,
        'processing': 5
    },
    'processing': {
        'shipping': 6,
        'product': 7,
        'processing': 8
    }
}

std_sentiment_label_table = {
    'negative': 0,
    'positive': 1
}

efl_sentiment_label_descriptions = {
    'negative': '이것은 부정 입니다.',
    'positive': '이것은 긍정 입니다.',
}

sentiment_scl_label_table = {
    'negative': {
        'negative': 0,
        'positive': 1
    },
    'positive': {
        'negative': 2,
        'positive': 3
    }
}
