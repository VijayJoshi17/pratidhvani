"""Transliteration feeds the aligner, rate filters and CER, so mistakes here
silently corrupt the training set rather than raising errors."""
from voiceclone.text import STAR, cer, romanize, star_tokens, syllables, translit


def test_inherent_vowel_is_kept():
    # unidecode turns this into "nimm mne"; the MMS aligner then cannot find the vowels
    assert translit("ನಿಮ್ಮ") == "nimma"
    assert translit("ಮನೆ") == "mane"


def test_virama_suppresses_inherent_vowel():
    assert translit("ಆಗ್ಬೇಕು") == "aagbeeku"


def test_anusvara_visarga_and_conjuncts():
    assert translit("ಅಹಂ") == "aham"
    assert translit("ದುಃಖಿ") == "duhkhi"
    assert translit("ಕ್ಷಮಾ") == "kshamaa"


def test_output_is_only_mms_vocabulary_letters():
    out = translit("ಹ್ಯಾಗೆ, ರೀ! 123 ‌")
    assert set(out) <= set("abcdefghijklmnopqrstuvwxyz")


def test_syllables_count_vowel_groups():
    # rate filters divide by this: "ನಮಸ್ಕಾರ" = na-mas-kaa-ra
    assert syllables("ನಮಸ್ಕಾರ") == 4
    assert syllables("") == 0


def test_romanize_drops_spaces_and_punctuation():
    assert romanize("ನಿಮ್ಮ ಮನೆ.") == "nimmamane"


def test_cer_basics():
    assert cer("abc", "abc") == 0
    assert cer("abd", "abc") == 1 / 3
    assert cer("", "abcd") == 1.0
    assert cer("x", "") == 1.0    # no division by zero on empty reference


def test_star_tokens_wrap_words_and_absorb_trailing_audio():
    toks, txt = star_tokens("ನಿಮ್ಮ , ಮನೆ")
    # punctuation-only "words" are dropped, each word is preceded by <star>,
    # and a trailing <star> keeps extra audio off the last word
    assert toks == [STAR, "n i m m a", STAR, "m a n e", STAR]
    assert txt == [STAR, "ನಿಮ್ಮ", STAR, "ಮನೆ", STAR]
    assert star_tokens("... !") == ([], [])
