import re
import os
import pandas as pd
import numpy as np
from utils import lemmatize, saveData, loadData
from modelBuilder import LDAmodel, NMFmodel, CatBoostModel


class ModelsRunner:
    def __init__(self,
                 prepareDF: pd.DataFrame,
                 descriptionRegexPattern: str,
                 vocab: list,
                 oneHotSkills: str,
                 modelPath: str,
                 modelType: str,
                 modelConfig: dict,
                 regrConfig: dict):
        self.resDF = prepareDF
        self.id2word = None
        self.encodeCorpus = None
        self.descRP = descriptionRegexPattern

        self.vocab = vocab
        self.oneHotSkills = oneHotSkills

        self.model = None
        self.modelType = modelType
        self.modelPath = modelPath
        self.modelConfig = modelConfig
        self.regrConfig = regrConfig

    def run_process(self, resume: str):
        # init model object
        self.oneHotSkills = loadData(self.oneHotSkills)

        if self.modelType == 'LDA':
            modelWrap = LDAmodel(self.resDF, descriptionRegexPattern=self.descRP,
                                 vocab=self.vocab, oneHotSkills=self.oneHotSkills)
        else:
            modelWrap = NMFmodel(self.resDF, descriptionRegexPattern=self.descRP,
                                 vocab=self.vocab, oneHotSkills=self.oneHotSkills)

        # fit model
        modelWrap.prepare_input()
        if not os.path.exists(self.modelPath):
            modelWrap.fit(modelConfig=self.modelConfig, savePath=self.modelPath)
        else:
            modelWrap.model = loadData(self.modelPath)

        # eval model
        modelWrap.predict()
        modelWrap.model_eval(topicTermData='./data/descriptionTopics.csv')
        clust, prepResume = modelWrap.inference(resume)

        self.model = modelWrap
        return clust, prepResume

    def recomend_prof(self, listProffesions: list[str], stopwordsProfs: list[str]) -> None:
        normProfName = []
        for prof in listProffesions:
            normProfName.extend(lemmatize(prof, self.descRP,
                                          stops=stopwordsProfs,
                                          tokens=self.vocab).split())

        resProf = np.unique(np.array(normProfName), return_counts=True)
        topInd = np.argpartition(resProf[1], -2)[-2:]
        print("Рекомендуемая профессия: " + " ".join([resProf[0][i] for i in topInd]))

    def recomend_skills(self, clust: int,
                        prepResume: list[str],
                        nRecSkills: int) -> None:
        top_terms = []
        if self.modelType == "LDA":
            top_terms = self.model.model.print_topic(clust, topn=len(prepResume) + int(nRecSkills * 1.5))
            # out in top_terms for example: 0.042*"react" + 0.040*"js" + 0.030*"git" + 0.029*"frontend"
            top_terms = re.findall(re.compile(r'"\w+"'), top_terms)
            top_terms = [x[1:-1] for x in top_terms if '_' not in x]

        elif self.modelType == "NMF":
            feature_names = self.model.vectorizer.get_feature_names_out()
            top_ids_words = self.model.model.components_[clust].argsort()[-(len(prepResume) + int(nRecSkills * 1.5)):][
                            ::-1]
            top_terms = [feature_names[i] for i in top_ids_words]

        outerSkills = list(set(top_terms) - set(prepResume))
        print(outerSkills)

        simCosine = np.zeros(shape=(len(prepResume), len(outerSkills)))
        for i, resumeSkill in enumerate(prepResume):
            a = self.oneHotSkills.loc[:, resumeSkill].values
            for j, clustSkill in enumerate(outerSkills):
                b = self.oneHotSkills.loc[:, clustSkill].values
                simCosine[i, j] = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

        print("Рекомедую изучить следующие навыки: (близость к вашим навыкам)")
        topInds = np.argpartition(simCosine.mean(axis=0), kth=-nRecSkills)[-nRecSkills:]
        for x in topInds:
            print(f'{outerSkills[x]}: {simCosine.mean(axis=0)[x]}', end='\n')

    def recomend_vacancies(self,
                           clust: int,
                           prepResume: list[str],
                           nRecVacs: int,
                           pathOrigData: str,
                           pathSaveResultVacs: str) -> None:
        resumeTokenVect = np.array([1 if token in prepResume else 0 for token in self.vocab], dtype=np.uint)
        currentClustVacs = self.oneHotSkills[(self.model.resDF['TopicLabel'] == clust).values]
        cosMetr = currentClustVacs.values.dot(resumeTokenVect) / np.linalg.norm(currentClustVacs.values, axis=1)
        topCos = np.argpartition(cosMetr, kth=-nRecVacs)[-nRecVacs:]
        topVacsIndex = currentClustVacs.index[topCos]

        # отсюда достать индексы и отправить в ориг датасет с них достать строки
        recVacsDF = self.model.resDF.iloc[topVacsIndex, :]
        dataOrig = pd.read_csv(pathOrigData, index_col=0)

        useColumns = ['Ids', 'Employer', 'Name', 'Salary', 'From', 'To', 'Experience', 'Schedule', 'Keys',
                      'Description']
        drop_columns = set(dataOrig.columns) - set(useColumns)
        dataOrig.drop(columns=drop_columns, axis=1, inplace=True)
        recDf = dataOrig.iloc[recVacsDF.index, :]
        recDf['resume similarity'] = cosMetr[topCos]
        saveData(recDf, pathSaveResultVacs)

    def recomend_salary(self, prepResume: list[str]):
        salaryModel = CatBoostModel(config=self.regrConfig)
        target = self.model.resDF[self.model.resDF['Salary'] & ~self.model.resDF['Description'].isnull()][
            ['From', 'To']].mean(axis=1)
        target = target[(target < 500000) & (target > 40000)]
        X_data = self.model.resDF.loc[target.index, ['Schedule', 'Experience', 'Description']]

        salaryModel.train(X_data, target.values)
        saveData(salaryModel.inference(' '.join(prepResume)),
                 './data/SalaryEstimation.csv')

    def run_recomends(self,
                      clust: int,
                      prepResume: list,
                      nRecVacs: int = 5,
                      nRecSkills: int = 5,
                      pathOrigData:str = './data/database.csv',
                      pathSaveRecsVacs: str = './data/Recomendations.csv') -> None:

        nameProfs = self.model.resDF[self.model.resDF['TopicLabel'] == clust]['Name'].values

        self.recomend_prof(listProffesions=nameProfs, stopwordsProfs=['junior', 'senior', 'middle'])

        self.recomend_skills(clust=clust,
                             prepResume=prepResume,
                             nRecSkills=nRecSkills)

        self.recomend_vacancies(clust=clust,
                                prepResume=prepResume,
                                nRecVacs=nRecVacs,
                                pathOrigData=pathOrigData,
                                pathSaveResultVacs=pathSaveRecsVacs)

        self.recomend_salary(prepResume=prepResume)






