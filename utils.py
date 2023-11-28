import re
import pymorphy2
import os
import pandas as pd
import pickle

lemmatizer = pymorphy2.MorphAnalyzer(lang='ru')
extensionStopWords = ['опыт', "работа", 'знание', 'требование', 'компания', "команда", "оклад", "хотеть",
                      'плюс', 'навык', 'сервис', 'предлагать', 'развитие', 'продукт',
                      'принцип', 'уровень', 'бизнес', 'наш', 'участие', 'процесс', 'новый',
                      'технический', 'условие', 'офис', 'решение', 'приложение', 'год',
                      'возможность', 'работать', 'умение', 'задача', 'понимание', 'проект',
                      'данные', 'система', 'разработка', 'использование', 'профессиональный',
                      'свой', 'написание', 'высокий', 'оформление', 'рабочий', 'поддержка',
                      'рф', 'дмс', 'заработный', 'который', 'график', 'удалённый', 'день',
                      'корпоративный', 'сотрудник', 'уверенный', 'формат', 'гибкий', 'результат',
                      'хороший', 'иметь', 'сложный', 'владение', 'это', 'желание', 'также',
                      'готовый', 'предстоять', 'инструмент', 'преимущество', 'кандидат',
                      'официальный', 'заказчик', 'гибридный', 'различный',
                      'искать', 'оплата', 'подход', 'развиваться', 'любой',
                      'помощь', 'обработка', 'ваш', 'модуль', 'комфортный', 'испытательный',
                      'месяц', 'курс', 'желательно', 'премия', 'отличный', 'обеспечение', 'связь',
                      'трудоустройство', 'собеседование', 'карьерный','основный', 'участвовать',
                      'москва', 'полный', 'ждать', 'знать', 'скидка', "менее", "более",
                      'выплата', 'здорово', 'деплый', 'уделять', 'ревить', 'изделие',
                      'розничный', 'нуль', 'конкурент', 'осуществление', 'отлично',
                      'отличие', 'трансформация', 'смелый', 'влияние', 'кабинет',
                      'требоваться', 'расширять', 'ценить', 'присутствие', 'мыслить',
                      "теннис", 'внутренний', "огромный", "потеряться", "прочее", 'требовать',
                      'сладость', 'мск', 'учить', 'индивидуально', 'скилла', 'дневный', 'идея',
                      'understanding', 'years', 'year', 'отклонение', 'инициативность',
                      'атмосфера', 'кофе', 'чай', "еда", 'ежемесячный', 'кресло', "вне",
                      'потребоваться', 'потребность', 'and', 'заказ', 'фиксировать',
                      'комьюнити', 'внешний', 'принимать', 'привет', 'согласно', 'супруг',
                      'оплачивать', 'фитнес', 'среди', 'предоставить', 'ценный', 'деятельность',
                      'продолжать', 'ключевой', 'пн', 'петербург', 'рядом', 'удобно', 'столовая',
                      'квартальный', 'гарантия', 'старший', 'полноценный', 'мтс', 'мфть',
                      'савёловский', 'белорусский', 'подразделение', 'читать', 'код', "разработчик",
                      "сфера", "интересный", "тк", 'россия', 'обучение', "программирование", 'ит',
                      "аналитика", "аналитик", "интеграция"]
REPLACE_TOKENS = {'k8s': 'kubernetes',
                  'джава': 'java',
                  'javascript': 'js',
                  'с++': 'c++', # first symbol is russian, second is english
                  ' си ': ' c ',
                  '+': ' ',
                  '++': '+'}

def token_replace(string):
    rep = dict((re.escape(k), v) for k, v in REPLACE_TOKENS.items())
    pattern = re.compile("|".join(rep.keys()))
    string = pattern.sub(lambda m: rep[re.escape(m.group(0))], string)
    return string

def strDictParse(x: str, pattern: str,
                 integr: bool = False,
                 leftBound: int = 0,
                 rightBound: int = 0,
                 save_all: bool = False) -> list:
    if x != x:
        return None
    else:
        s = re.findall(pattern, x)
        if len(s) < 1:
            return None
        else:
            if not save_all:
                s = s[0]
                if rightBound == 0: rightBound = len(s)
                return int(s[leftBound:rightBound]) if integr else s[leftBound:rightBound]
            else:
                return [skill[leftBound:rightBound].lower() for skill in s]


def lemmatize(text: str, delSymbPattern: str,
              stops: list = [],
              tokens: list = None,
              bounds: bool = True,
              centerSlice: float = 0.5,
              sliceRadius: int = 100) -> str:

    text_preps = re.sub(delSymbPattern, ' ', text.lower())
    text_preps = token_replace(text_preps)

    lenText = len(text_preps.split())
    if tokens:
        s = []
        for word in text_preps.split():
            prep_word = lemmatizer.parse(word)[0].normal_form
            if (prep_word in tokens) and (prep_word not in stops):
                s.append(prep_word)
        out = ' '.join(s)

    else:
        if bounds and (lenText > sliceRadius/(1-centerSlice)):
            lB = int(lenText * centerSlice-sliceRadius)
            rB = int(lenText * centerSlice+sliceRadius)
        else:
            lB = 0
            rB = lenText

        s = []
        for word in text_preps.split()[lB:rB]:
            if (len(word) >= 2) and not (word.isdigit()):
                prep_word = lemmatizer.parse(word)[0].normal_form
                if prep_word not in stops:
                    s.append(prep_word)
        out = ' '.join(s)
    return out


def saveData(data: object,
             filename: str) -> None:
    _, extension = os.path.splitext(filename)
    if extension == '.csv':
        data.to_csv(filename)
    elif extension == '.pkl':
        with open(filename, 'wb') as saveFile:
            pickle.dump(data, saveFile)
    else:
        assert False, 'Specified wrong file extension!'

    print(f"Файл сохранен в {filename}")


def loadData(filename: str):
    assert os.path.exists(filename), 'Specified file doesnt exist!'
    _, extension = os.path.splitext(filename)
    if extension == '.csv':
        data = pd.read_csv(filename, index_col=0)
    elif extension == '.pkl':
        with open(filename, 'rb') as saveFile:
            data = pickle.load(saveFile)
    else:
        assert False, 'Specified wrong file extension!'
    print(f"Данные загружены c ({filename})")
    return data