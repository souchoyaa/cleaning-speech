"""ITN prompts for the 30 languages supported by Qwen3-ASR.

Two parallel dictionaries (``PROMPTS_EN``, ``PROMPTS_NATIVE``) hold one
template per language. Every template:
  - lists what to convert (numbers, dates, times, currencies, %, ordinals,
    measurements) and what to preserve verbatim;
  - states the "leave idioms alone when in doubt" rule;
  - embeds the locale-specific separators / currency placement / time and
    date conventions for that language only;
  - contains 2-3 natural examples in the target language plus at least one
    negative (idiom that must NOT be normalized);
  - ends with ``{ambiguity_section}`` followed by an ``Input:`` / ``Output:``
    block — labels in English for ``PROMPTS_EN``, in the target language for
    ``PROMPTS_NATIVE``.

``build_prompt`` is the single entry point; the caller (asr_join.itn)
formats ``{ambiguity_section}`` itself (it depends on rover state, not
on the language).
"""

from __future__ import annotations


# =============================================================================
# PROMPTS_EN — instructions in English, examples & I/O labels in target lang
# =============================================================================
PROMPTS_EN: dict[str, str] = {
    "en": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "one of them", "nine times out of ten"), leave it as-is. Output only the rewritten text, nothing else.

Locale: decimal point "." (3.14), thousands comma (1,500), currency before amount ($100), percent attached "50%", 24h times "15:30", dates "10/5/2024".

Examples:
Input: I have sixty nine apples and three hundred dollars
Output: I have 69 apples and $300
Input: the meeting is at three thirty pm
Output: the meeting is at 3:30 pm
Input: she is one of them
Output: she is one of them
{ambiguity_section}
Input: {text}
Output:""",

    "zh": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "一个人", "三心二意"), leave it as-is. Output only the rewritten text, nothing else.

Locale: Arabic digits inline with measure words preserved (三百零五个 → 305个); no thousands separator for short numbers; currency "¥100" or "100元"; times "3点30分" or "15:30"; percent "50%".

Examples:
Input: 我有三百零五个苹果
Output: 我有305个苹果
Input: 下午三点半开会
Output: 下午3点半开会
Input: 他三心二意做不成事
Output: 他三心二意做不成事
{ambiguity_section}
Input: {text}
Output:""",

    "yue": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Use Traditional Chinese characters as in the input. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate. When in doubt (Cantonese idioms like "一個人", "九唔搭八"), leave it as-is. Output only the rewritten text.

Locale: Arabic digits inline with measure words preserved (三百零五個 → 305個); no thousands separator for short numbers; currency "$100" or "100蚊"; times "3點半" / "15:30"; percent "50%".

Examples:
Input: 我有三百零五個蘋果
Output: 我有305個蘋果
Input: 而家三點半,食緊飯
Output: 而家3點半,食緊飯
Input: 佢成日九唔搭八
Output: 佢成日九唔搭八
{ambiguity_section}
Input: {text}
Output:""",

    "ar": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements to standard digit form. Match the digit system of the input (Western 0-9 OR Arabic-Indic ٠-٩). Preserve all other words, casing, punctuation, spacing, and order exactly. Respect RTL flow. Do not translate. When in doubt (idioms like "واحد منهم"), leave it as-is. Output only the rewritten text.

Locale: decimal "." or "٫"; thousands "," or "٬"; currency after the amount with space (100 ر.س, 50 د.إ); percent "50%" or "٥٠٪"; 24h times "15:30"; dates "5/10/2024".

Examples:
Input: عندي ثلاثة وعشرون كتابا
Output: عندي 23 كتابا
Input: الموعد الساعة الثالثة والنصف
Output: الموعد الساعة 3:30
Input: هو واحد منهم
Output: هو واحد منهم
{ambiguity_section}
Input: {text}
Output:""",

    "de": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "einer von ihnen", "auf Wolke sieben"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after amount with space (100 €), percent "50 %" with space, times "15:30 Uhr", dates "5.10.2024".

Examples:
Input: er hat neunundsechzig Äpfel und dreihundert Euro
Output: er hat 69 Äpfel und 300 €
Input: das Treffen ist um halb vier nachmittags
Output: das Treffen ist um 15:30 Uhr
Input: er ist einer von ihnen
Output: er ist einer von ihnen
{ambiguity_section}
Input: {text}
Output:""",

    "fr": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "un jour", "des centaines de fois"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands space (1 500), currency after with space (100 €), percent with space (50 %), 24h times "15h30", dates "5/10/2024".

Examples:
Input: il a vingt trois ans et trois cents euros
Output: il a 23 ans et 300 €
Input: rendez-vous à quinze heures trente
Output: rendez-vous à 15h30
Input: un jour je partirai
Output: un jour je partirai
{ambiguity_section}
Input: {text}
Output:""",

    "es": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "uno de ellos", "a las mil maravillas"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 €), percent "50 %" with space, times "15:30", dates "5/10/2024".

Examples:
Input: tengo veintitrés años y trescientos euros
Output: tengo 23 años y 300 €
Input: la reunión es a las tres y media de la tarde
Output: la reunión es a las 15:30
Input: él es uno de ellos
Output: él es uno de ellos
{ambiguity_section}
Input: {text}
Output:""",

    "pt": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "um deles", "à beça"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 €, 50 R$), percent "50 %" with space, times "15:30", dates "5/10/2024".

Examples:
Input: ele tem vinte e três anos e trezentos euros
Output: ele tem 23 anos e 300 €
Input: a reunião é às três e meia da tarde
Output: a reunião é às 15:30
Input: ele é um deles
Output: ele é um deles
{ambiguity_section}
Input: {text}
Output:""",

    "id": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "salah satu", "tujuh keliling"), leave it as-is. Output only the rewritten text.

Locale: decimal dot (3.14), thousands comma (1,500), currency before amount (Rp100, $100), percent attached "50%", times "15:30", dates "5/10/2024".

Examples:
Input: saya punya dua puluh tiga apel
Output: saya punya 23 apel
Input: rapat jam tiga lewat tiga puluh sore
Output: rapat jam 15:30 sore
Input: dia salah satu dari mereka
Output: dia salah satu dari mereka
{ambiguity_section}
Input: {text}
Output:""",

    "it": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "uno di loro", "in quattro e quattr'otto"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 €), percent "50 %" with space, times "15:30", dates "5/10/2024".

Examples:
Input: ha ventitré anni e trecento euro
Output: ha 23 anni e 300 €
Input: la riunione è alle tre e mezza del pomeriggio
Output: la riunione è alle 15:30
Input: lui è uno di loro
Output: lui è uno di loro
{ambiguity_section}
Input: {text}
Output:""",

    "ko": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve native/Sino number readings: convert the digit, keep the counter and particle as spoken (세 시 삼십 분 → 3시 30분). Preserve all other words, punctuation, spacing, and order exactly. Do not translate. When in doubt (idioms like "그중 하나", "하나 마나"), leave it as-is. Output only the rewritten text.

Locale: decimal dot (3.14), thousands comma (1,500), currency "₩100" or "100원", percent attached "50%", times "15:30" or "오후 3시 30분", dates "2024년 10월 5일".

Examples:
Input: 사과가 스물세 개 있어요
Output: 사과가 23개 있어요
Input: 회의는 오후 세 시 삼십 분이에요
Output: 회의는 오후 3시 30분이에요
Input: 그는 그중 하나예요
Output: 그는 그중 하나예요
{ambiguity_section}
Input: {text}
Output:""",

    "ru": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "один из них", "семь пятниц на неделе"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 ₽, 50 €), percent "50 %" with space, times "15:30", dates "05.10.2024".

Examples:
Input: ему двадцать три года и триста рублей
Output: ему 23 года и 300 ₽
Input: встреча в три тридцать дня
Output: встреча в 15:30
Input: он один из них
Output: он один из них
{ambiguity_section}
Input: {text}
Output:""",

    "th": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to Western digit form (0-9). Preserve all other words, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "หนึ่งในนั้น"), leave it as-is. Output only the rewritten text.

Locale: Western digits 0-9, decimal dot (3.14), thousands comma (1,500), currency before amount (฿100) or "100 บาท", percent attached "50%", times "15:30", dates "5/10/2567".

Examples:
Input: ฉันมีแอปเปิ้ลยี่สิบสามลูก
Output: ฉันมีแอปเปิ้ล 23 ลูก
Input: ประชุมบ่ายสามโมงครึ่ง
Output: ประชุม 15:30
Input: เขาเป็นหนึ่งในนั้น
Output: เขาเป็นหนึ่งในนั้น
{ambiguity_section}
Input: {text}
Output:""",

    "vi": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "một trong số họ", "ba chân bốn cẳng"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100.000 ₫), percent "50%" attached, times "15:30" or "3 giờ chiều", dates "5/10/2024".

Examples:
Input: tôi có hai mươi ba quả táo
Output: tôi có 23 quả táo
Input: cuộc họp lúc ba giờ rưỡi chiều
Output: cuộc họp lúc 15:30
Input: anh ấy là một trong số họ
Output: anh ấy là một trong số họ
{ambiguity_section}
Input: {text}
Output:""",

    "ja": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve native/Sino reading duality: convert the digit, keep the counter and particle as spoken (三時半 → 3時半). Preserve all other words, punctuation, spacing, and order exactly. Do not translate. When in doubt (idioms like "そのうちの一人", "一期一会"), leave it as-is. Output only the rewritten text.

Locale: decimal dot (3.14), thousands comma (1,500), currency "¥100" before or "100円" after, percent attached "50%", times "15:30" or "午後3時30分", dates "2024年10月5日".

Examples:
Input: りんごが二十三個あります
Output: りんごが23個あります
Input: 会議は午後三時半です
Output: 会議は午後3時半です
Input: 彼はそのうちの一人です
Output: 彼はそのうちの一人です
{ambiguity_section}
Input: {text}
Output:""",

    "tr": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Attach Turkish case suffixes to digits with an apostrophe (üçte → 3'te, üç buçukta → 3:30'da). Do not translate. When in doubt (idioms like "onlardan biri"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 ₺, 50 €), percent prefix "%50" no space, times "15:30", dates "5.10.2024".

Examples:
Input: yirmi üç elmam var
Output: 23 elmam var
Input: toplantı saat üç buçukta
Output: toplantı saat 3:30'da
Input: o onlardan biri
Output: o onlardan biri
{ambiguity_section}
Input: {text}
Output:""",

    "hi": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form (use Western digits 0-9 unless the input uses Devanagari digits, in which case match). Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "उनमें से एक"), leave it as-is. Output only the rewritten text.

Locale: decimal dot (3.14), Indian thousands grouping with comma (1,500 or 1,00,000), currency before amount (₹100), percent attached "50%", times "15:30" or "दोपहर 3 बजे", dates "5/10/2024".

Examples:
Input: मेरे पास तेईस सेब हैं
Output: मेरे पास 23 सेब हैं
Input: मीटिंग साढ़े तीन बजे है
Output: मीटिंग 3:30 बजे है
Input: वह उनमें से एक है
Output: वह उनमें से एक है
{ambiguity_section}
Input: {text}
Output:""",

    "ms": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "salah seorang", "seribu satu"), leave it as-is. Output only the rewritten text.

Locale: decimal dot (3.14), thousands comma (1,500), currency before amount (RM100, $100), percent attached "50%", times "15:30" or "pukul 3 petang", dates "5/10/2024".

Examples:
Input: saya ada dua puluh tiga biji epal
Output: saya ada 23 biji epal
Input: mesyuarat pukul tiga setengah petang
Output: mesyuarat pukul 15:30
Input: dia salah seorang daripada mereka
Output: dia salah seorang daripada mereka
{ambiguity_section}
Input: {text}
Output:""",

    "nl": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "een van hen", "negen van de tien keer"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 €), percent "50 %" with space, times "15:30", dates "5-10-2024".

Examples:
Input: hij heeft drieëntwintig appels en driehonderd euro
Output: hij heeft 23 appels en 300 €
Input: de vergadering is om half vier 's middags
Output: de vergadering is om 15:30
Input: hij is een van hen
Output: hij is een van hen
{ambiguity_section}
Input: {text}
Output:""",

    "sv": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "en av dem", "i elfte timmen"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands space (1 500), currency after with space (100 kr, 50 €), percent "50 %" with space, times "15:30", dates "2024-10-05".

Examples:
Input: han har tjugotre äpplen och trehundra kronor
Output: han har 23 äpplen och 300 kr
Input: mötet är halv fyra på eftermiddagen
Output: mötet är 15:30
Input: han är en av dem
Output: han är en av dem
{ambiguity_section}
Input: {text}
Output:""",

    "da": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "en af dem", "i tide og utide"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 kr, 50 €), percent "50 %" with space, times "15:30", dates "5.10.2024".

Examples:
Input: han har treogtyve æbler og trehundrede kroner
Output: han har 23 æbler og 300 kr
Input: mødet er halv fire om eftermiddagen
Output: mødet er 15:30
Input: han er en af dem
Output: han er en af dem
{ambiguity_section}
Input: {text}
Output:""",

    "fi": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "yksi heistä", "tuhannen taalan paikka"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands space (1 500), currency after with space (100 €), percent "50 %" with space, times "15.30" or "15:30", dates "5.10.2024".

Examples:
Input: hänellä on kaksikymmentäkolme omenaa ja kolmesataa euroa
Output: hänellä on 23 omenaa ja 300 €
Input: kokous on puoli neljä iltapäivällä
Output: kokous on 15.30
Input: hän on yksi heistä
Output: hän on yksi heistä
{ambiguity_section}
Input: {text}
Output:""",

    "pl": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "jeden z nich", "raz na ruski rok"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot or space (1.500 or 1 500), currency after with space (150 zł, 100 €), percent attached "50%", times "15:30", dates "5.10.2024".

Examples:
Input: ma dwadzieścia trzy jabłka i sto pięćdziesiąt złotych
Output: ma 23 jabłka i 150 zł
Input: spotkanie o wpół do czwartej po południu
Output: spotkanie o 15:30
Input: on jest jednym z nich
Output: on jest jednym z nich
{ambiguity_section}
Input: {text}
Output:""",

    "cs": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "jeden z nich", "jednou za uherský rok"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands space or dot (1 500 or 1.500), currency after with space (200 Kč, 100 €), percent "50 %" with space, times "15:30", dates "5.10.2024".

Examples:
Input: má dvacet tři jablek a dvě stě korun
Output: má 23 jablek a 200 Kč
Input: schůzka je v půl čtvrté odpoledne
Output: schůzka je v 15:30
Input: je jedním z nich
Output: je jedním z nich
{ambiguity_section}
Input: {text}
Output:""",

    "fil": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "isa sa kanila", "balat-sibuyas"), leave it as-is. Output only the rewritten text.

Locale: decimal dot (3.14), thousands comma (1,500), currency before amount (₱100, $100), percent attached "50%", times "15:30" or "alas tres ng hapon", dates "5/10/2024".

Examples:
Input: may dalawampu't tatlong mansanas ako
Output: may 23 na mansanas ako
Input: ang pulong ay alas tres y medya ng hapon
Output: ang pulong ay 15:30
Input: siya ay isa sa kanila
Output: siya ay isa sa kanila
{ambiguity_section}
Input: {text}
Output:""",

    "fa": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Match the digit system of the input (Western 0-9 or Persian ۰-۹). Preserve all other words, casing, punctuation, spacing, and order exactly. Respect RTL flow. Do not translate. When in doubt (idioms like "یکی از آنها"), leave it as-is. Output only the rewritten text.

Locale: decimal "," or "٫"; thousands "." or "٬"; currency after the amount with space (100 تومان); percent attached "50%"; 24h times "15:30"; dates "1403/07/14".

Examples:
Input: من بیست و سه کتاب دارم
Output: من 23 کتاب دارم
Input: جلسه ساعت سه و نیم بعد از ظهر است
Output: جلسه ساعت 15:30 بعد از ظهر است
Input: او یکی از آنهاست
Output: او یکی از آنهاست
{ambiguity_section}
Input: {text}
Output:""",

    "el": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "ένας από αυτούς", "στο πι και φι"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 €), percent attached "50%", times "15:30", dates "5/10/2024".

Examples:
Input: έχει είκοσι τρία μήλα και τριακόσια ευρώ
Output: έχει 23 μήλα και 300 €
Input: η συνάντηση είναι στις τρεις και μισή το απόγευμα
Output: η συνάντηση είναι στις 15:30
Input: αυτός είναι ένας από αυτούς
Output: αυτός είναι ένας από αυτούς
{ambiguity_section}
Input: {text}
Output:""",

    "hu": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "egy közülük", "tizenkettő egy tucat"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands space (1 500), currency after with space (100 €, 1500 Ft), percent attached "50%", times "15:30", dates "2024.10.05.".

Examples:
Input: huszonhárom almája és háromszáz eurója van
Output: 23 almája és 300 € van
Input: a megbeszélés fél négykor délután van
Output: a megbeszélés 15:30-kor van
Input: ő egy közülük
Output: ő egy közülük
{ambiguity_section}
Input: {text}
Output:""",

    "mk": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Use Cyrillic script as in the input. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "еден од нив", "ни на крај памет"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 ден, 50 €), percent attached "50%", times "15:30", dates "5.10.2024".

Examples:
Input: има дваесет и три јаболка и триста денари
Output: има 23 јаболка и 300 ден
Input: состанокот е во три и пол попладне
Output: состанокот е во 15:30
Input: тој е еден од нив
Output: тој е еден од нив
{ambiguity_section}
Input: {text}
Output:""",

    "ro": """Convert spoken numbers, dates, times, currencies, percentages, ordinals, and measurements in the input to standard digit form. Preserve all other words, casing, punctuation, spacing, and order exactly. Do not translate or paraphrase. When in doubt (idioms like "unul dintre ei", "la paștele cailor"), leave it as-is. Output only the rewritten text.

Locale: decimal comma (3,14), thousands dot (1.500), currency after with space (100 lei, 50 €), percent attached "50%", times "15:30" or "trei și jumătate", dates "5.10.2024".

Examples:
Input: are douăzeci și trei de mere și trei sute de lei
Output: are 23 de mere și 300 lei
Input: ședința este la trei și jumătate după-amiaza
Output: ședința este la 15:30
Input: el este unul dintre ei
Output: el este unul dintre ei
{ambiguity_section}
Input: {text}
Output:""",
}


# =============================================================================
# PROMPTS_NATIVE — instructions, examples, AND I/O labels in target language
# =============================================================================
PROMPTS_NATIVE: dict[str, str] = {
    "en": PROMPTS_EN["en"],  # native == English

    "zh": """将输入中的口语数字、日期、时间、货币、百分比、序数和度量转换为标准数字形式。其他词语、大小写、标点、空格和顺序完全保持不变。不要翻译或改写。遇到不确定的情况(如成语"一个人"、"三心二意"),保持原样。只输出改写后的文本,不输出其他内容。

规则:阿拉伯数字,量词保留(三百零五个 → 305个);短数字不加千位分隔符;货币符号保留(¥100、100元);时间"3点30分"或"15:30";百分比"50%"。

示例:
输入:我有三百零五个苹果
输出:我有305个苹果
输入:下午三点半开会
输出:下午3点半开会
输入:他三心二意做不成事
输出:他三心二意做不成事
{ambiguity_section}
输入:{text}
输出:""",

    "yue": """將輸入嘅口語數字、日期、時間、貨幣、百分比、序數同度量轉成標準數字。用返繁體字。其他字詞、大小寫、標點、空格、次序完全保留。唔好翻譯,唔好改寫。如果唔肯定(粵語俗語如「一個人」、「九唔搭八」),保留原樣。只輸出改寫後嘅文,其他乜都唔好出。

規則:阿拉伯數字,量詞保留(三百零五個 → 305個);短數字唔用千位分隔;貨幣「$100」或「100蚊」;時間「3點半」或「15:30」;百分比「50%」。

例子:
輸入:我有三百零五個蘋果
輸出:我有305個蘋果
輸入:而家三點半,食緊飯
輸出:而家3點半,食緊飯
輸入:佢成日九唔搭八
輸出:佢成日九唔搭八
{ambiguity_section}
輸入:{text}
輸出:""",

    "ar": """حوّل الأرقام والتواريخ والأوقات والعملات والنسب المئوية والترتيبيات والقياسات المنطوقة في النص إلى أرقام قياسية. التزم بنظام الأرقام في الإدخال (٠-٩ عربية شرقية أو 0-9 غربية). احتفظ بجميع الكلمات الأخرى وعلامات الترقيم والمسافات والترتيب كما هي. لا تترجم ولا تعِد الصياغة. عند الشك (تعابير مثل "واحد منهم")، اتركها كما هي. أخرج النص المعدَّل فقط.

القواعد: الفاصلة العشرية "." أو "٫"؛ فاصل الآلاف "," أو "٬"؛ العملة بعد المبلغ مع مسافة (100 ر.س)؛ النسبة "50%"؛ الوقت بصيغة 24 ساعة "15:30".

أمثلة:
الإدخال: عندي ثلاثة وعشرون كتابا
الإخراج: عندي 23 كتابا
الإدخال: الموعد الساعة الثالثة والنصف
الإخراج: الموعد الساعة 3:30
الإدخال: هو واحد منهم
الإخراج: هو واحد منهم
{ambiguity_section}
الإدخال: {text}
الإخراج:""",

    "de": """Wandle gesprochene Zahlen, Daten, Uhrzeiten, Währungen, Prozentangaben, Ordnungszahlen und Maße im Text in Ziffernform um. Behalte alle anderen Wörter, Groß-/Kleinschreibung, Satzzeichen, Leerzeichen und Reihenfolge exakt bei. Nicht übersetzen, nicht umformulieren. Im Zweifel (Redewendungen wie "einer von ihnen", "auf Wolke sieben") unverändert lassen. Gib nur den umgeschriebenen Text aus.

Konventionen: Dezimalkomma (3,14), Tausenderpunkt (1.500), Währung nach dem Betrag mit Leerzeichen (100 €), Prozent "50 %" mit Leerzeichen, Uhrzeit "15:30 Uhr", Datum "5.10.2024".

Beispiele:
Eingabe: er hat neunundsechzig Äpfel und dreihundert Euro
Ausgabe: er hat 69 Äpfel und 300 €
Eingabe: das Treffen ist um halb vier nachmittags
Ausgabe: das Treffen ist um 15:30 Uhr
Eingabe: er ist einer von ihnen
Ausgabe: er ist einer von ihnen
{ambiguity_section}
Eingabe: {text}
Ausgabe:""",

    "fr": """Convertis les nombres parlés, dates, heures, devises, pourcentages, ordinaux et mesures en chiffres standard. Conserve exactement les autres mots, la casse, la ponctuation, les espaces et l'ordre. Ne traduis pas, ne paraphrase pas. En cas de doute (expressions comme "un jour", "des centaines de fois"), laisse tel quel. Sortie : uniquement le texte réécrit, rien d'autre.

Conventions : décimale virgule (3,14), milliers espace (1 500), devise après avec espace (100 €), pourcentage avec espace (50 %), heures 24h avec "h" (15h30), dates "5/10/2024".

Exemples :
Entrée : il a vingt trois ans et trois cents euros
Sortie : il a 23 ans et 300 €
Entrée : rendez-vous à quinze heures trente
Sortie : rendez-vous à 15h30
Entrée : un jour je partirai
Sortie : un jour je partirai
{ambiguity_section}
Entrée : {text}
Sortie :""",

    "es": """Convierte los números, fechas, horas, monedas, porcentajes, ordinales y medidas hablados a su forma estándar en dígitos. Conserva exactamente las demás palabras, mayúsculas, puntuación, espacios y orden. No traduzcas ni parafrasees. Ante la duda (modismos como "uno de ellos", "a las mil maravillas"), déjalo igual. Devuelve solo el texto reescrito.

Convenciones: decimal coma (3,14), miles punto (1.500), moneda después con espacio (100 €), porcentaje "50 %" con espacio, hora "15:30", fecha "5/10/2024".

Ejemplos:
Entrada: tengo veintitrés años y trescientos euros
Salida: tengo 23 años y 300 €
Entrada: la reunión es a las tres y media de la tarde
Salida: la reunión es a las 15:30
Entrada: él es uno de ellos
Salida: él es uno de ellos
{ambiguity_section}
Entrada: {text}
Salida:""",

    "pt": """Converta os números, datas, horas, moedas, percentuais, ordinais e medidas falados para a forma padrão em dígitos. Mantenha exatamente as demais palavras, maiúsculas, pontuação, espaços e ordem. Não traduza nem parafraseie. Na dúvida (expressões como "um deles", "à beça"), mantenha igual. Devolva apenas o texto reescrito.

Convenções: decimal vírgula (3,14), milhar ponto (1.500), moeda depois com espaço (100 €, 50 R$), percentual "50 %" com espaço, hora "15:30", data "5/10/2024".

Exemplos:
Entrada: ele tem vinte e três anos e trezentos euros
Saída: ele tem 23 anos e 300 €
Entrada: a reunião é às três e meia da tarde
Saída: a reunião é às 15:30
Entrada: ele é um deles
Saída: ele é um deles
{ambiguity_section}
Entrada: {text}
Saída:""",

    "id": """Ubah angka, tanggal, waktu, mata uang, persentase, bilangan urut, dan satuan ukur yang diucapkan dalam teks menjadi bentuk digit standar. Pertahankan kata-kata lain, huruf besar/kecil, tanda baca, spasi, dan urutan persis seperti aslinya. Jangan menerjemahkan atau memparafrasekan. Jika ragu (ungkapan seperti "salah satu", "tujuh keliling"), biarkan apa adanya. Keluarkan hanya teks yang sudah diubah.

Konvensi: desimal titik (3.14), ribuan koma (1,500), mata uang sebelum nilai (Rp100, $100), persen tanpa spasi "50%", waktu "15:30", tanggal "5/10/2024".

Contoh:
Masukan: saya punya dua puluh tiga apel
Keluaran: saya punya 23 apel
Masukan: rapat jam tiga lewat tiga puluh sore
Keluaran: rapat jam 15:30 sore
Masukan: dia salah satu dari mereka
Keluaran: dia salah satu dari mereka
{ambiguity_section}
Masukan: {text}
Keluaran:""",

    "it": """Converti i numeri, le date, gli orari, le valute, le percentuali, gli ordinali e le misure parlati nel testo in forma standard con cifre. Mantieni invariati tutte le altre parole, maiuscole/minuscole, punteggiatura, spazi e ordine. Non tradurre né parafrasare. Nel dubbio (modi di dire come "uno di loro", "in quattro e quattr'otto"), lascia invariato. Restituisci solo il testo riscritto.

Convenzioni: decimale virgola (3,14), migliaia punto (1.500), valuta dopo con spazio (100 €), percentuale "50 %" con spazio, ora "15:30", data "5/10/2024".

Esempi:
Ingresso: ha ventitré anni e trecento euro
Uscita: ha 23 anni e 300 €
Ingresso: la riunione è alle tre e mezza del pomeriggio
Uscita: la riunione è alle 15:30
Ingresso: lui è uno di loro
Uscita: lui è uno di loro
{ambiguity_section}
Ingresso: {text}
Uscita:""",

    "ko": """입력의 음성 숫자, 날짜, 시간, 통화, 백분율, 서수, 측정 단위를 표준 숫자 형태로 변환하세요. 고유어/한자어 수사 이중성은 유지: 숫자만 변환하고 단위/조사는 발화 그대로 둡니다(세 시 삼십 분 → 3시 30분). 다른 모든 단어, 대소문자, 구두점, 띄어쓰기, 순서는 정확히 보존하세요. 번역하거나 바꿔 쓰지 마세요. 확신이 없으면(관용구 "그중 하나", "하나 마나") 그대로 두세요. 변환된 텍스트만 출력하세요.

규칙: 소수점 마침표(3.14), 천 단위 쉼표(1,500), 통화 "₩100" 또는 "100원", 백분율 "50%" 붙여서, 시각 "15:30" 또는 "오후 3시 30분", 날짜 "2024년 10월 5일".

예시:
입력: 사과가 스물세 개 있어요
출력: 사과가 23개 있어요
입력: 회의는 오후 세 시 삼십 분이에요
출력: 회의는 오후 3시 30분이에요
입력: 그는 그중 하나예요
출력: 그는 그중 하나예요
{ambiguity_section}
입력: {text}
출력:""",

    "ru": """Преобразуй произнесённые числа, даты, время, валюты, проценты, порядковые числительные и единицы измерения в стандартную цифровую форму. Остальные слова, регистр, знаки препинания, пробелы и порядок сохрани без изменений. Не переводи и не перефразируй. В случае сомнения (идиомы "один из них", "семь пятниц на неделе") оставь как есть. Выведи только переписанный текст.

Правила: десятичная запятая (3,14), разряды через точку (1.500), валюта после числа с пробелом (100 ₽, 50 €), процент с пробелом "50 %", время "15:30", дата "05.10.2024".

Примеры:
Ввод: ему двадцать три года и триста рублей
Вывод: ему 23 года и 300 ₽
Ввод: встреча в три тридцать дня
Вывод: встреча в 15:30
Ввод: он один из них
Вывод: он один из них
{ambiguity_section}
Ввод: {text}
Вывод:""",

    "th": """แปลงตัวเลข วันที่ เวลา สกุลเงิน เปอร์เซ็นต์ ลำดับ และหน่วยวัดที่พูดออกมาในข้อความให้เป็นรูปเลขอารบิก (0-9) คงคำอื่น ๆ ตัวพิมพ์ใหญ่/เล็ก เครื่องหมายวรรคตอน ช่องว่าง และลำดับเดิมไว้ทุกประการ ห้ามแปลหรือเรียบเรียงใหม่ หากไม่แน่ใจ (สำนวนเช่น "หนึ่งในนั้น") ให้คงไว้ตามเดิม ส่งกลับเฉพาะข้อความที่แก้แล้วเท่านั้น

กฎ: เลขอารบิก 0-9, ทศนิยมจุด (3.14), หลักพันคอมมา (1,500), สกุลเงินก่อนจำนวน (฿100) หรือ "100 บาท", เปอร์เซ็นต์ติดกัน "50%", เวลา "15:30", วันที่ "5/10/2567"

ตัวอย่าง:
ป้อน: ฉันมีแอปเปิ้ลยี่สิบสามลูก
ผลลัพธ์: ฉันมีแอปเปิ้ล 23 ลูก
ป้อน: ประชุมบ่ายสามโมงครึ่ง
ผลลัพธ์: ประชุม 15:30
ป้อน: เขาเป็นหนึ่งในนั้น
ผลลัพธ์: เขาเป็นหนึ่งในนั้น
{ambiguity_section}
ป้อน: {text}
ผลลัพธ์:""",

    "vi": """Chuyển các con số, ngày tháng, giờ giấc, tiền tệ, phần trăm, số thứ tự và đơn vị đo lường nói trong văn bản sang dạng chữ số chuẩn. Giữ nguyên hoàn toàn các từ khác, chữ hoa/thường, dấu câu, khoảng trắng và trật tự. Không dịch, không diễn giải lại. Nếu phân vân (thành ngữ như "một trong số họ", "ba chân bốn cẳng"), giữ nguyên. Chỉ xuất văn bản đã chuyển đổi.

Quy ước: dấu thập phân phẩy (3,14), hàng nghìn dấu chấm (1.500), tiền tệ sau số có khoảng trắng (100.000 ₫), phần trăm "50%" liền số, giờ "15:30" hoặc "3 giờ chiều", ngày "5/10/2024".

Ví dụ:
Đầu vào: tôi có hai mươi ba quả táo
Đầu ra: tôi có 23 quả táo
Đầu vào: cuộc họp lúc ba giờ rưỡi chiều
Đầu ra: cuộc họp lúc 15:30
Đầu vào: anh ấy là một trong số họ
Đầu ra: anh ấy là một trong số họ
{ambiguity_section}
Đầu vào: {text}
Đầu ra:""",

    "ja": """入力中の話し言葉の数字、日付、時刻、通貨、パーセント、序数、計量を標準の数字表記に変換してください。和語・漢語読みの二重性は維持し、数字のみ変換して助数詞や助詞は発話どおりに残します(三時半 → 3時半)。他の語句、大文字小文字、句読点、空白、順序は厳密に保持してください。翻訳・言い換えは禁止。迷ったとき(慣用句「そのうちの一人」「一期一会」など)はそのまま残します。書き換え後のテキストのみを出力してください。

規則:小数点はピリオド(3.14)、千の位はカンマ(1,500)、通貨は前に「¥100」または後ろに「100円」、パーセントは「50%」と詰めて、時刻は「15:30」または「午後3時30分」、日付は「2024年10月5日」。

例:
入力:りんごが二十三個あります
出力:りんごが23個あります
入力:会議は午後三時半です
出力:会議は午後3時半です
入力:彼はそのうちの一人です
出力:彼はそのうちの一人です
{ambiguity_section}
入力:{text}
出力:""",

    "tr": """Metindeki sözlü sayıları, tarihleri, saatleri, para birimlerini, yüzdeleri, sıra sayılarını ve ölçüleri standart rakam biçimine çevir. Diğer kelimeleri, büyük/küçük harfleri, noktalama işaretlerini, boşlukları ve sırayı aynen koru. Türkçe çekim eklerini rakamlara kesme işaretiyle ekle (üçte → 3'te, üç buçukta → 3:30'da). Çeviri ya da yeniden ifade yok. Kararsız kaldığında (deyimler "onlardan biri" gibi) olduğu gibi bırak. Yalnızca yeniden yazılmış metni ver.

Kurallar: ondalık virgül (3,14), binlik nokta (1.500), para birimi sayıdan sonra boşlukla (100 ₺, 50 €), yüzde önde boşluksuz "%50", saat "15:30", tarih "5.10.2024".

Örnekler:
Giriş: yirmi üç elmam var
Çıkış: 23 elmam var
Giriş: toplantı saat üç buçukta
Çıkış: toplantı saat 3:30'da
Giriş: o onlardan biri
Çıkış: o onlardan biri
{ambiguity_section}
Giriş: {text}
Çıkış:""",

    "hi": """इनपुट में बोले गए नंबर, तारीख, समय, मुद्रा, प्रतिशत, क्रम सूचक और माप को मानक अंक रूप में बदलें (पश्चिमी 0-9 अंक प्रयोग करें, जब तक इनपुट में देवनागरी अंक न हों)। अन्य सभी शब्द, बड़े/छोटे अक्षर, विराम चिह्न, स्पेस और क्रम बिल्कुल वैसे ही रखें। अनुवाद या पुनर्कथन नहीं। संदेह होने पर (मुहावरे जैसे "उनमें से एक") ज्यों का त्यों छोड़ दें। केवल बदला हुआ टेक्स्ट दें।

नियम: दशमलव बिंदु (3.14), भारतीय हजार समूहन कॉमा से (1,500 या 1,00,000), मुद्रा रकम से पहले (₹100), प्रतिशत बिना स्पेस "50%", समय "15:30" या "दोपहर 3 बजे", तारीख "5/10/2024"।

उदाहरण:
इनपुट: मेरे पास तेईस सेब हैं
आउटपुट: मेरे पास 23 सेब हैं
इनपुट: मीटिंग साढ़े तीन बजे है
आउटपुट: मीटिंग 3:30 बजे है
इनपुट: वह उनमें से एक है
आउटपुट: वह उनमें से एक है
{ambiguity_section}
इनपुट: {text}
आउटपुट:""",

    "ms": """Tukarkan nombor, tarikh, masa, mata wang, peratusan, nombor turutan dan ukuran yang dituturkan dalam input kepada bentuk digit piawai. Kekalkan semua perkataan lain, huruf besar/kecil, tanda baca, ruang dan susunan dengan tepat. Jangan terjemah atau parafrasakan. Jika ragu (simpulan bahasa seperti "salah seorang", "seribu satu"), biarkan apa adanya. Keluarkan hanya teks yang ditulis semula.

Konvensyen: titik perpuluhan (3.14), koma ribuan (1,500), mata wang sebelum amaun (RM100, $100), peratus tanpa ruang "50%", masa "15:30" atau "pukul 3 petang", tarikh "5/10/2024".

Contoh:
Masukan: saya ada dua puluh tiga biji epal
Keluaran: saya ada 23 biji epal
Masukan: mesyuarat pukul tiga setengah petang
Keluaran: mesyuarat pukul 15:30
Masukan: dia salah seorang daripada mereka
Keluaran: dia salah seorang daripada mereka
{ambiguity_section}
Masukan: {text}
Keluaran:""",

    "nl": """Zet gesproken getallen, data, tijden, valuta, percentages, rangtelwoorden en maten in de tekst om naar de standaard cijfervorm. Behoud alle andere woorden, hoofdlettergebruik, leestekens, spaties en volgorde exact. Niet vertalen of parafraseren. Bij twijfel (uitdrukkingen als "een van hen", "negen van de tien keer") onveranderd laten. Geef alleen de herschreven tekst terug.

Conventies: decimaal komma (3,14), duizendtallen punt (1.500), valuta achter het bedrag met spatie (100 €), percentage "50 %" met spatie, tijd "15:30", datum "5-10-2024".

Voorbeelden:
Invoer: hij heeft drieëntwintig appels en driehonderd euro
Uitvoer: hij heeft 23 appels en 300 €
Invoer: de vergadering is om half vier 's middags
Uitvoer: de vergadering is om 15:30
Invoer: hij is een van hen
Uitvoer: hij is een van hen
{ambiguity_section}
Invoer: {text}
Uitvoer:""",

    "sv": """Omvandla talade siffror, datum, tider, valutor, procent, ordningstal och måttenheter i texten till standardiserad sifferform. Bevara alla andra ord, versaler/gemener, skiljetecken, mellanslag och ordningsföljd exakt. Översätt eller omformulera inte. Vid tvekan (uttryck som "en av dem", "i elfte timmen") lämna oförändrat. Returnera enbart den omskrivna texten.

Konventioner: decimal med komma (3,14), tusental med mellanslag (1 500), valuta efter beloppet med mellanslag (100 kr, 50 €), procent "50 %" med mellanslag, tid "15:30", datum "2024-10-05".

Exempel:
Inmatning: han har tjugotre äpplen och trehundra kronor
Utmatning: han har 23 äpplen och 300 kr
Inmatning: mötet är halv fyra på eftermiddagen
Utmatning: mötet är 15:30
Inmatning: han är en av dem
Utmatning: han är en av dem
{ambiguity_section}
Inmatning: {text}
Utmatning:""",

    "da": """Omdan talte tal, datoer, klokkeslæt, valutaer, procenter, ordenstal og mål i teksten til standardiseret cifferform. Bevar alle andre ord, store/små bogstaver, tegnsætning, mellemrum og rækkefølge nøjagtigt. Ingen oversættelse eller omskrivning. I tvivlstilfælde (vendinger som "en af dem", "i tide og utide") lad det stå. Returnér kun den omskrevne tekst.

Konventioner: decimal med komma (3,14), tusinder med punktum (1.500), valuta efter beløbet med mellemrum (100 kr, 50 €), procent "50 %" med mellemrum, klokkeslæt "15:30", dato "5.10.2024".

Eksempler:
Inddata: han har treogtyve æbler og trehundrede kroner
Uddata: han har 23 æbler og 300 kr
Inddata: mødet er halv fire om eftermiddagen
Uddata: mødet er 15:30
Inddata: han er en af dem
Uddata: han er en af dem
{ambiguity_section}
Inddata: {text}
Uddata:""",

    "fi": """Muuta tekstissä esiintyvät puhutut numerot, päivämäärät, kellonajat, valuutat, prosentit, järjestysluvut ja mittayksiköt vakiomuotoisiksi numeroiksi. Säilytä kaikki muut sanat, isot/pienet kirjaimet, välimerkit, välilyönnit ja järjestys täsmälleen ennallaan. Ei käännöstä eikä uudelleenmuotoilua. Epäselvissä tilanteissa (sanonnat kuten "yksi heistä", "tuhannen taalan paikka") jätä ennalleen. Tulosta vain uudelleenkirjoitettu teksti.

Säännöt: desimaalipilkku (3,14), tuhaterotin välilyönti (1 500), valuutta luvun jälkeen välilyönnillä (100 €), prosentti "50 %" välilyönnillä, kellonaika "15.30" tai "15:30", päivämäärä "5.10.2024".

Esimerkkejä:
Syöte: hänellä on kaksikymmentäkolme omenaa ja kolmesataa euroa
Tuloste: hänellä on 23 omenaa ja 300 €
Syöte: kokous on puoli neljä iltapäivällä
Tuloste: kokous on 15.30
Syöte: hän on yksi heistä
Tuloste: hän on yksi heistä
{ambiguity_section}
Syöte: {text}
Tuloste:""",

    "pl": """Zamień wypowiedziane w tekście liczby, daty, godziny, waluty, procenty, liczebniki porządkowe i miary na zapis cyfrowy w standardowej formie. Pozostałe słowa, wielkość liter, interpunkcję, odstępy i kolejność zachowaj bez zmian. Nie tłumacz i nie parafrazuj. W razie wątpliwości (idiomy "jeden z nich", "raz na ruski rok") zostaw bez zmian. Zwróć wyłącznie przepisany tekst.

Zasady: przecinek dziesiętny (3,14), separator tysięcy kropka lub spacja (1.500 lub 1 500), waluta po kwocie ze spacją (150 zł, 100 €), procent bez spacji "50%", godzina "15:30", data "5.10.2024".

Przykłady:
Wejście: ma dwadzieścia trzy jabłka i sto pięćdziesiąt złotych
Wyjście: ma 23 jabłka i 150 zł
Wejście: spotkanie o wpół do czwartej po południu
Wyjście: spotkanie o 15:30
Wejście: on jest jednym z nich
Wyjście: on jest jednym z nich
{ambiguity_section}
Wejście: {text}
Wyjście:""",

    "cs": """Převeď mluvená čísla, data, časy, měny, procenta, řadové číslovky a míry v textu do standardní číslicové podoby. Ostatní slova, velká/malá písmena, interpunkci, mezery a pořadí zachovej přesně. Nepřekládej a nepřeformulovávej. Pokud váháš (frazémy "jeden z nich", "jednou za uherský rok"), nech to být. Vrať pouze přepsaný text.

Pravidla: desetinná čárka (3,14), oddělovač tisíců mezera nebo tečka (1 500 nebo 1.500), měna za částkou s mezerou (200 Kč, 100 €), procenta "50 %" s mezerou, čas "15:30", datum "5.10.2024".

Příklady:
Vstup: má dvacet tři jablek a dvě stě korun
Výstup: má 23 jablek a 200 Kč
Vstup: schůzka je v půl čtvrté odpoledne
Výstup: schůzka je v 15:30
Vstup: je jedním z nich
Výstup: je jedním z nich
{ambiguity_section}
Vstup: {text}
Výstup:""",

    "fil": """I-convert ang mga binibigkas na numero, petsa, oras, salapi, porsyento, panunuran at sukat sa loob ng input sa karaniwang anyo ng digit. Panatilihin nang eksakto ang lahat ng iba pang salita, malalaki/maliliit na titik, bantas, espasyo at pagkakasunod-sunod. Huwag isalin o muling iparirala. Kapag hindi sigurado (mga idyoma gaya ng "isa sa kanila", "balat-sibuyas"), hayaan ito. Ilabas lamang ang muling isinulat na teksto.

Mga kumbensyon: tuldok bilang desimal (3.14), kuwit bilang libuhan (1,500), salapi bago ang halaga (₱100, $100), porsyento nakadikit "50%", oras "15:30" o "alas tres ng hapon", petsa "5/10/2024".

Mga halimbawa:
Pasok: may dalawampu't tatlong mansanas ako
Labas: may 23 na mansanas ako
Pasok: ang pulong ay alas tres y medya ng hapon
Labas: ang pulong ay 15:30
Pasok: siya ay isa sa kanila
Labas: siya ay isa sa kanila
{ambiguity_section}
Pasok: {text}
Labas:""",

    "fa": """اعداد، تاریخ‌ها، زمان‌ها، ارزها، درصدها، اعداد ترتیبی و واحدهای اندازه‌گیری گفتاری در متن را به شکل عددی استاندارد تبدیل کن. سیستم عدد ورودی را حفظ کن (لاتین 0-9 یا فارسی ۰-۹). همه‌ی کلمات دیگر، حروف بزرگ/کوچک، نقطه‌گذاری، فاصله‌ها و ترتیب را دقیقاً نگه دار. جهت RTL را رعایت کن. ترجمه یا بازنویسی نکن. اگر شک داشتی (اصطلاحاتی مانند «یکی از آنها»)، تغییری نده. فقط متن بازنویسی‌شده را برگردان.

قواعد: ممیز اعشار "," یا "٫"؛ جداکننده هزارگان "." یا "٬"؛ ارز پس از مبلغ با فاصله (100 تومان)؛ درصد "50%"؛ ساعت ۲۴ ساعته "15:30"؛ تاریخ "1403/07/14".

مثال‌ها:
ورودی: من بیست و سه کتاب دارم
خروجی: من 23 کتاب دارم
ورودی: جلسه ساعت سه و نیم بعد از ظهر است
خروجی: جلسه ساعت 15:30 بعد از ظهر است
ورودی: او یکی از آنهاست
خروجی: او یکی از آنهاست
{ambiguity_section}
ورودی: {text}
خروجی:""",

    "el": """Μετάτρεψε τους προφορικούς αριθμούς, ημερομηνίες, ώρες, νομίσματα, ποσοστά, τακτικούς αριθμούς και μονάδες μέτρησης στο κείμενο σε τυπική ψηφιακή μορφή. Διατήρησε όλες τις άλλες λέξεις, πεζά/κεφαλαία, σημεία στίξης, κενά και σειρά ακριβώς όπως είναι. Μην μεταφράζεις και μην παραφράζεις. Όταν αμφιβάλλεις (εκφράσεις όπως «ένας από αυτούς», «στο πι και φι»), άφησέ το ως έχει. Δώσε μόνο το ξαναγραμμένο κείμενο.

Κανόνες: υποδιαστολή κόμμα (3,14), διαχωριστικό χιλιάδων τελεία (1.500), νόμισμα μετά το ποσό με κενό (100 €), ποσοστό χωρίς κενό "50%", ώρα "15:30", ημερομηνία "5/10/2024".

Παραδείγματα:
Είσοδος: έχει είκοσι τρία μήλα και τριακόσια ευρώ
Έξοδος: έχει 23 μήλα και 300 €
Είσοδος: η συνάντηση είναι στις τρεις και μισή το απόγευμα
Έξοδος: η συνάντηση είναι στις 15:30
Είσοδος: αυτός είναι ένας από αυτούς
Έξοδος: αυτός είναι ένας από αυτούς
{ambiguity_section}
Είσοδος: {text}
Έξοδος:""",

    "hu": """Alakítsd át a szövegben elhangzó számokat, dátumokat, időpontokat, pénznemeket, százalékokat, sorszámokat és mértékegységeket a szokásos számjegyes alakra. Az összes többi szót, kis-/nagybetűt, írásjelet, szóközt és sorrendet pontosan őrizd meg. Ne fordíts és ne fogalmazz át. Ha bizonytalan vagy (kifejezések mint "egy közülük", "tizenkettő egy tucat"), hagyd változatlanul. Csak az átírt szöveget add vissza.

Konvenciók: tizedesvessző (3,14), ezres elválasztó szóköz (1 500), pénznem az összeg után szóközzel (100 €, 1500 Ft), százalék szóköz nélkül "50%", idő "15:30", dátum "2024.10.05.".

Példák:
Bemenet: huszonhárom almája és háromszáz eurója van
Kimenet: 23 almája és 300 € van
Bemenet: a megbeszélés fél négykor délután van
Kimenet: a megbeszélés 15:30-kor van
Bemenet: ő egy közülük
Kimenet: ő egy közülük
{ambiguity_section}
Bemenet: {text}
Kimenet:""",

    "mk": """Претвори ги изговорените броеви, датуми, времиња, валути, проценти, редни броеви и мерки во текстот во стандардна цифрена форма. Користи кирилично писмо како во влезот. Зачувај ги другите зборови, голи/мали букви, интерпункција, празни места и редослед точно. Не преведувај и не преформулирај. Кога си во дилема (изрази како „еден од нив", „ни на крај памет"), остави ги непроменети. Врати само препишаниот текст.

Правила: децимална запирка (3,14), илјадарка точка (1.500), валута по сумата со празно место (100 ден, 50 €), процент без празно место "50%", време "15:30", датум "5.10.2024".

Примери:
Влез: има дваесет и три јаболка и триста денари
Излез: има 23 јаболка и 300 ден
Влез: состанокот е во три и пол попладне
Излез: состанокот е во 15:30
Влез: тој е еден од нив
Излез: тој е еден од нив
{ambiguity_section}
Влез: {text}
Излез:""",

    "ro": """Transformă numerele, datele, orele, valutele, procentele, numeralele ordinale și măsurile pronunțate în text în forma standard cu cifre. Păstrează exact celelalte cuvinte, majusculele/minusculele, semnele de punctuație, spațiile și ordinea. Nu traduce și nu reformula. Când ai dubii (expresii precum „unul dintre ei", „la paștele cailor"), lasă neschimbat. Returnează doar textul rescris.

Convenții: virgulă zecimală (3,14), separator de mii punct (1.500), valuta după sumă cu spațiu (100 lei, 50 €), procent fără spațiu "50%", oră "15:30", dată "5.10.2024".

Exemple:
Intrare: are douăzeci și trei de mere și trei sute de lei
Ieșire: are 23 de mere și 300 lei
Intrare: ședința este la trei și jumătate după-amiaza
Ieșire: ședința este la 15:30
Intrare: el este unul dintre ei
Ieșire: el este unul dintre ei
{ambiguity_section}
Intrare: {text}
Ieșire:""",
}


def build_prompt(
    lang: str,
    text: str,
    ambiguity_section: str = "",
    native: bool = True,
) -> str:
    """Return the final prompt with ``text`` and optional ``ambiguity_section`` filled in.

    ``ambiguity_section`` is opaque to this module — the caller (asr_join.itn)
    formats the ROVER-tie instructions and passes them in. It should be empty
    string when there are no ambiguous words, otherwise a block starting with
    a newline so it inserts cleanly between the examples and the final
    ``Input:``/``Output:`` block (see ``itn._format_ambiguity_section``).

    Raises ``KeyError`` if ``lang`` is not in the selected dict — caller is
    responsible for fallback (typically to ``"en"``).
    """
    template = (PROMPTS_NATIVE if native else PROMPTS_EN)[lang]
    return template.format(text=text, ambiguity_section=ambiguity_section)
