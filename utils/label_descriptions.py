std_label_table = {
    'shipping': 0,
    'product': 1,
    'processing': 2,
    'etc': 3,
}

efl_category_label_descriptions = {
    'shipping': '이것은 배송과 관계 있는 문장입니다.',
    'product': '이것은 제품과 관계 있는 문장입니다.',
    'processing': '이것은 처리와 관계 있는 문장입니다.',
    'etc': '이것은 배송, 제품, 처리와 관계가 없는 문장입니다.',
}

scl_label_table = {
    'shipping': {
        'shipping': 0,
        'product': 1,
        'processing': 2,
        'etc': 3,
    },
    'product': {
        'shipping': 4,
        'product': 5,
        'processing': 6,
        'etc': 7,
    },
    'processing': {
        'shipping': 8,
        'product': 9,
        'processing': 10,
        'etc': 11,
    },
    'etc': {
        'shipping': 12,
        'product': 13,
        'processing': 14,
        'etc': 15,
    },
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
