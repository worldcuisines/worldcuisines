import logging
import os
import random

logging.basicConfig(level=logging.INFO)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(CUR_DIR, "resources")
TOP_K = 15
N_THREADS = 32
RANDOM_SEED = 42

MAX_ANS_FROM_FINEGRAINED = 2
MAX_ANS_FROM_COARSEGRAINED = 3
MAX_ANS_ALL = 4 

random.seed(RANDOM_SEED)

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
    'Hokkien': 'Hokkien_<LOCATION>',
    'Hokkien_Medan_spoken': 'Hokkien (Medan)_<LOCATION>',
    'Tagalog': 'Tagalog_<LOCATION>',
    'Thai': 'Thai_<LOCATION>',
    'Azerbaijani': 'Azerbaijani_<LOCATION>',
    'Russian_Casual': 'Russian_<LOCATION>',
    'Russian_Formal': 'Russian_<LOCATION>',
    'Italian': 'Italian_<LOCATION>'
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
    'Hokkien': ['Minnan'],
    'Hokkien_Medan_spoken': ['Minnan'],
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