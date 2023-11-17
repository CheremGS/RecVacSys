import pandas as pd
import numpy as np
import re
import os
from utils import strDictParse, lemmatize, lemmatizer, extensionStopWords, saveData, loadData
from tqdm import tqdm, trange

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")


class DataPreparation:
    def __init__(self, csv_file):
        self.csv_path = csv_file
        self.originDF = None
        self.prepDF = None
        self.skillSet = []
        self.oneHotSkills = None
        self.regexPatterns = {'Employer': [re.compile(r"'name': '\w+"), False, 9, None, False],
                              'Schedule': [re.compile(r"'name': '.+'"), False, 9, -1, False],
                              'Experience': [re.compile(r"'name': '.+'"), False, 9, -1, False],
                              'Keys': [re.compile(r"\'[^(\[\]'\s,)]+\'"), True, 1, -1, True],
                              'Description': re.compile(r'[^\w\#\+]+')}

    def _read_csv(self) -> None:
        self.originDF = pd.read_csv(self.csv_path)
        assert set(['Ids', 'Name', 'Description', 'Keys']) <= set(self.originDF.columns), \
            'Your csv-data have include next columns: [Ids, Name, Description, Keys]'
        useColumns = ['Ids', 'Employer', 'Name', 'Salary', 'From', 'To', 'Experience', 'Schedule', 'Keys', 'Description']
        drop_columns = set(self.originDF.columns) - set(useColumns)
        self.prepDF = self.originDF.copy().drop(columns=drop_columns, axis=1)

    def _get_skillSet(self) -> None:
        print('Создание списка всех навыков...')
        vacancies = list(self.prepDF["Keys"].apply(lambda x: x if x else ['None']).values)
        skill_set = list(set([skill for vacancy_skills in vacancies for skill in vacancy_skills]))
        assert len(skill_set) > 0, 'Длина массива названий навыков равна нулю! (len(skillSet) = 0)'
        self.skillSet = list(set([lemmatizer.parse(word)[0].normal_form for word in skill_set]))

    def parseDictCols(self, parseColumns: list,
                      skillTokens:bool = True,
                      stopWords:list = [],
                      saveDF: str = './data/prepdf.csv') -> None:

        if os.path.exists(saveDF):
            self.prepDF = loadData(saveDF)
            useColumns = ['Ids', 'Employer', 'Name', 'Salary', 'From', 'To', 'Experience', 'Schedule', 'Keys',
                          'Description']
            drop_columns = set(self.prepDF.columns) - set(useColumns)
            self.prepDF.drop(columns=drop_columns, axis=1, inplace=True)

            self.prepDF['Description'] = self.prepDF['Description'].replace(np.nan, ' ')
            self.prepDF.Name = self.prepDF.Name.replace(np.nan, ' ')
            for col in parseColumns[:-1]:
                self.prepDF[col] = self.prepDF[col].apply(
                    lambda x: strDictParse(x, *[self.regexPatterns[col][i] for i in range(5)]))

            self._get_skillSet()
        else:
            print("Обработка текстового описания всех вакансий...")
            self.prepDF = self.prepDF[~self.prepDF.Description.isnull()]

            for col in parseColumns[:-1]:
                self.prepDF[col] = self.prepDF[col].apply(
                    lambda x: strDictParse(x, *[self.regexPatterns[col][i] for i in range(5)]))

            self._get_skillSet()
            self.prepDF['Description'] = self.prepDF['Description'].apply(lambda x: lemmatize(x,
                                                                          self.regexPatterns['Description'],
                                                                          stops=stopWords,
                                                                          tokens=self.skillSet if skillTokens else None))

            saveData(self.prepDF, saveDF)

    def compute_oneHotSkill(self, savePath: str):
        if os.path.exists(savePath):
            self.oneHotSkills = loadData(savePath)
        else:
            self.oneHotSkills = np.zeros((len(self.skillSet), self.prepDF.shape[0]), dtype=np.uint)
            for i_vac in trange(self.prepDF.shape[0], desc='Процесс oneHot кодировки навыков по текстам вакансий'):
                for i_skill in range(len(self.skillSet)):
                    if self.skillSet[i_skill] in self.prepDF.iloc[i_vac, list(self.prepDF.columns).index('Description')]:
                        self.oneHotSkills[i_skill, i_vac] = 1

            self.oneHotSkills = pd.DataFrame(self.oneHotSkills.T, columns=self.skillSet)
            saveData(self.oneHotSkills, savePath)

    def run(self, baseTokenIsSkills:bool = True,
            pathSaveLemmasTexts = './data/prepdf.csv',
            oneHotSavePath='./data/oneHotSkills.csv'):
        self._read_csv()
        stops = stopwords.words('russian')
        stops_en = stopwords.words('english')
        stops.extend(stops_en)
        stops.extend(extensionStopWords)

        parseColumns = list(self.regexPatterns.keys())
        self.parseDictCols(parseColumns=parseColumns,
                           skillTokens=baseTokenIsSkills,
                           stopWords=stops,
                           saveDF=pathSaveLemmasTexts)
        self.compute_oneHotSkill(savePath=oneHotSavePath)

