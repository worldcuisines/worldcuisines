import logging
import os

logging.basicConfig(level=logging.DEBUG)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(CUR_DIR, "resources")
TOP_K = 15
N_THREADS = 32
RANDOM_SEED = 42

MAX_ANS_FROM_FINEGRAINED = 2
MAX_ANS_FROM_COARSEGRAINED = 3
MAX_ANS_ALL = 4

LANGUAGE_CODE_MAPPING = {
    'en': 'English',
    'id_formal': 'Indonesian_Formal',
    'id_casual': 'Indonesian_Casual',
    'zh-CN': 'Chinese_Mandarin',
    'ko_formal': 'Korean_Formal',
    'ko_casual': 'Korean_Casual',
    'ja_formal': 'Japanese_Formal',
    'ja_casual': 'Japanese_Casual',
    'su_loma': 'Sundanese_Loma',
    'jv_krama': 'Javanese_Krama',
    'jv_ngoko': 'Javanese_Ngoko',
    'cs': 'Czech',
    'es': 'Spanish',
    'fr': 'French',
    'ar': 'Arabic_MSA',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'mr': 'Marathi',
    'si_formal_spoken': 'Sinhala_formal_spoken',
    'yo': 'Yoruba',
    'yue': 'Cantonese',
    'nan': 'HokkienMedan',
    'nan_spoken': 'HokkienMedan_Spoken',
    'tl': 'Tagalog',
    'th': 'Thai',
    'az': 'Azerbaijani',
    'ru_formal': 'Russian_Formal',
    'ru_casual': 'Russian_Casual',
    'it': 'Italian',
    'sc': 'Sardinian_Logudorese'
}

ALL_LANGUAGES = list(LANGUAGE_CODE_MAPPING.keys())

LOCATION_MAPPING = {
    'English': 'English_<LOCATION>',
    'Indonesian_Formal': 'Indonesian_<LOCATION>',
    'Indonesian_Casual': 'Indonesian_<LOCATION>',
    'Chinese_Mandarin': 'Chinese_<LOCATION>',
    'Korean_Formal': 'Korean_<LOCATION>',
    'Korean_Casual': 'Korean_<LOCATION>',
    'Japanese_Formal': 'Japanese_<LOCATION>',
    'Japanese_Casual': 'Japanese_<LOCATION>',
    'Sundanese_Loma': 'Sundanese_<LOCATION>',
    'Javanese_Krama': 'Javanese_<LOCATION>',
    'Javanese_Ngoko': 'Javanese_<LOCATION>',
    'Czech': 'Czech_<LOCATION>',
    'Spanish': 'Spanish_<LOCATION>',
    'French': 'French_<LOCATION>',
    'Arabic_MSA': 'Arabic_<LOCATION>',
    'Hindi': 'Hindi_<LOCATION>',
    'Bengali': 'Bengali_<LOCATION>',
    'Marathi': 'Marathi_<LOCATION>',
    'Sinhala_formal_spoken': 'Sinhala_<LOCATION>',
    'Yoruba': 'Yoruba_<LOCATION>',
    'Cantonese': 'Cantonese_<LOCATION>',
    'HokkienMedan': 'HokkienMedan_<LOCATION>',
    'HokkienMedan_Spoken': 'HokkienMedan_<LOCATION>', # Note since Hokkien Medan (Spoken) is missing, we replace with Hokkien_<LOCATION>
    'Tagalog': 'Tagalog_<LOCATION>',
    'Thai': 'Thai_<LOCATION>',
    'Azerbaijani': 'Azerbaijani_<LOCATION>',
    'Russian_Casual': 'Russian_<LOCATION.nom>',
    'Russian_Formal': 'Russian_<LOCATION.nom>',
    'Italian': 'Italian_<LOCATION>',
    'Sardinian_Logudorese': 'Sardinian_<LOCATION>'
}

ALIAS_MAPPING = {
    'English': ['Simple English', 'Old English', 'Jamaican Creole English', 'Inggeris'],
    'Indonesian_Formal': ['Tiếng Indonesian', 'Indonesian'],
    'Indonesian_Casual': ['Tiếng Indonesian', 'Indonesian'],
    'Chinese_Mandarin': ['Chinese', 'Literary Chinese'],
    'Korean_Formal': ['Korea', 'Korean'],
    'Korean_Casual': ['Korea', 'Korean'],
    'Japanese_Formal': ['Japanese'],
    'Japanese_Casual': ['Japanese'],
    'Sundanese_Loma': ['Sundanese'],
    'Javanese_Krama': ['Jawa', 'Javanese', 'Tiếng Java'],
    'Javanese_Ngoko': ['Jawa', 'Javanese', 'Tiếng Java'],
    'Czech': ['Czech'],
    'Spanish': ['Spanish'],
    'French': ['French', 'Perancis'],
    'Arabic_MSA': ['Arabic'],
    'Hindi': ['Hindi'],
    'Bengali': ['Bangla'],
    'Marathi': ['Marathi'],
    'Sinhala_formal_spoken': ['Sinhala'],
    'Yoruba': ['Yoruba'],
    'Cantonese': ['Cantonese'],
    'HokkienMedan': ['Minnan'],
    'HokkienMedan_Spoken': ['Minnan'],
    'Tagalog': ['Tiếng Tagalog', 'Tagalog'],
    'Thai': ['Thai'],
    'Azerbaijani': ['Azerbaijani', 'South Azerbaijani'],
    'Russian_Casual': ['Russian'],
    'Russian_Formal': ['Russian'],
    'Italian': ['Itali', 'Italy', 'Tiếng Italian']
}

CONTEXT_TYPE_ACTION = {
    'context:\nN/A\n\nans:\n<NAME>': 'Name',
    'context:\nN/A\n\nans:\n<LOCATION>\n<CUISINE>': 'Location',
    'context:\n<LOCATION>\n<CUISINE>\n\nans:\n<NAME>': 'Name',
    'context:\n<CATEGORY>\n<FINE_CATEGORY>\nans:\n<NAME>\n<CUISINE>': '<DROP>'
}

# These are agreed numbers
NUM_MAX_DISHES_EVAL = 550 # The first 500 randomly shuffled dishes will always be used for eval; 
                          # Set 550 as for a buffer/reserved space of extra 50 dishes for eval
                          # for problematic dishes (since ~1800 needs to be in train)
PROMPT_EVAL_PORTION = 0.3 # 0.3 here is the eval portion
ODDS_ADVERSARIAL = 0.8
