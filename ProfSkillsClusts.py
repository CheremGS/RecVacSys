import numpy as np
import gensim
from gensim.models import CoherenceModel
import re
import os

import pandas as pd

from utils import lemmatize, saveData, loadData
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.corpora as corpora
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


class modelProcess():
    def __init__(self, prepareDF,
                 descriptionRegexPattern: str,
                 vocab: list,
                 oneHotSkills: pd.DataFrame,
                 modelType: str):
        self.resDF = prepareDF.copy()
        self.id2word = None
        self.model = None
        self.encodeCorpus = None
        self.descRP = descriptionRegexPattern
        self.vocab = vocab
        self.oneHotSkills = oneHotSkills
        self.modelType = modelType

    def inference(self, resume: str) -> (int, list):
        important_words=lemmatize(resume, delSymbPattern=self.descRP, tokens=self.vocab)
        assert len(important_words) > 0, \
            'Опишите свои навыки более конкретно (после обработки резюме не было найдено ни одного навыка)'

        if self.modelType == 'NMF':
            preps = important_words.split()
            return self.model.transform(self.vectorizer.transform(preps)).sum(axis=0).argmax(), preps

        elif self.modelType == "LDA":
            ques_vec = []
            ques_vec = self.id2word.doc2bow(important_words.split())

            topic_vec = []
            topic_vec = self.model[ques_vec]
            word_count_array = np.empty((len(topic_vec), 2), dtype=np.object)
            for i in range(len(topic_vec)):
                word_count_array[i, 0] = topic_vec[i][0]
                word_count_array[i, 1] = topic_vec[i][1]

            idx = np.argsort(word_count_array[:, 1])
            idx = idx[::-1]
            word_count_array = word_count_array[idx]

            return word_count_array[0][0], important_words.split()

    def _prepare_LDA_input(self) -> None:
        text = [text.split() for text in self.resDF.Description.values]

        # higher threshold fewer phrases.
        bigram = gensim.models.Phrases(text, min_count=5, threshold=5)
        trigram = gensim.models.Phrases(bigram[text], threshold=5)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        textPrepData = trigram_mod[bigram_mod[text]]

        self.id2word = corpora.Dictionary(textPrepData)
        self.encodeCorpus = [self.id2word.doc2bow(text) for text in textPrepData]

        self.resDF['VacancyCorpus'] = [word for word in textPrepData]

    def _prepare_MNF_input(self):
        text = self.resDF.Description.values
        self.vectorizer = TfidfVectorizer(max_features=1000,
                                          max_df=0.997,
                                          min_df=0.003)
        self.encodeCorpus = self.vectorizer.fit_transform(text)

    def NMF_fit_predict(self,
                        modelConfig: dict,
                        modelName: str = 'NMFmodel.pkl',
                        savePath: str = './models'):
        self._prepare_MNF_input()
        modelPath = os.path.join(savePath, modelName)
        if not os.path.exists(modelPath):
            print('Создание модели NMF. Обучение модели...')
            self.model = NMF(**modelConfig)
            self.model.fit(self.encodeCorpus)
            saveData(self.model, modelPath)
        else:
            self.model = loadData(modelPath)

        self.resDF['TopicLabel'] = self.model.transform(self.encodeCorpus).argmax(axis=1).astype(np.int)
        self.modelEval()

    def LDA_fit_predict(self,
                        modelConfig: dict,
                        modelName='LdaModel.pkl',
                        savePath = './models', ) -> None:
        self._prepare_LDA_input()
        modelPath = os.path.join(savePath, modelName)
        if not os.path.exists(modelPath):
            print('Создание модели LDA. Обучение модели...')
            self.model = gensim.models.LdaModel(self.encodeCorpus,
                                                id2word=self.id2word,
                                                **modelConfig)
            saveData(self.model, modelPath)
        else:
            self.model = loadData(modelPath)

        resTopics = self.model.get_document_topics(self.encodeCorpus)
        self.resDF['TopicLabel'] = np.array([i[0][0] if len(i)>0 else -1 for i in resTopics], dtype=np.int)
        self.resDF['TopicProb'] = [i[0][1] if len(i)>0 else None for i in resTopics]

        self.modelEval()

    def modelEval(self, topicTermData = './data/descriptionTopics.csv'):
        descrTopics = {}
        if self.modelType == 'LDA':
            # Compute Perplexity # a measure of how good the model is. lower the better.
            print('\nPerplexity: ', self.model.log_perplexity(self.encodeCorpus))

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=self.model, texts=self.resDF.VacancyCorpus,
                                                 dictionary=self.id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: ', coherence_lda)

            for topic in self.model.show_topics():
                sks = re.findall(re.compile(r'"\w+"'), topic[1])
                descrTopics[topic[0]] = ' '.join([x[1:-1] for x in sks])

        elif self.modelType == 'NMF':
            feature_names = self.vectorizer.get_feature_names_out()
            sns.heatmap(cosine_similarity(self.model.components_)).set(xticklabels=[], yticklabels=[])
            plt.title('Косинусная близость выделенных тематик')
            plt.show()

            for topic_idx, topic_words in enumerate(self.model.components_):
                top_words_idx = topic_words.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                descrTopics[topic_idx + 1] = ' '.join(top_words)

        saveData(pd.Series(descrTopics), topicTermData)

    def recommendProfsSkillsVacs(self,
                                 resume: str,
                                 nRecVacs: int = 5,
                                 nRecSkills: int = 5,
                                 pathOrigData:str = './data/database.csv',
                                 pathSaveRecsVacs: str = './data/Recomendations.csv') -> None:
        clust, prepResume = self.inference(resume)
        nameProfs = self.resDF[self.resDF['TopicLabel'] == clust]['Name'].values

        normProfName = []
        for prof in nameProfs:
            normProfName.extend(lemmatize(prof, self.descRP,
                                          stops=['junior', 'senior', 'middle'],
                                          tokens=self.vocab).split())

        resProf = np.unique(np.array(normProfName), return_counts=True)
        topInd = np.argpartition(resProf[1], -2)[-2:]
        print("Рекомендуемая профессия: " + " ".join([resProf[0][i] for i in topInd]))
        print('Самые частые навыки для подобранного кластера профессий (помимо указанных в вашем резюме):')
        top_terms = []

        if self.modelType == "LDA":
            top_terms = self.model.print_topic(clust, topn=len(prepResume)+int(nRecSkills*1.5)) # 0.042*"react" + 0.040*"js" + 0.030*"git" + 0.029*"frontend"
            top_terms = re.findall(re.compile(r'"\w+"'), top_terms)
            top_terms = [x[1:-1] for x in top_terms if '_' not in x]

        elif self.modelType == "NMF":
            feature_names = self.vectorizer.get_feature_names_out()
            top_ids_words = self.model.components_[clust].argsort()[-(len(prepResume)+int(nRecSkills*1.5)):][::-1]
            top_terms = [feature_names[i] for i in top_ids_words]

        outerSkills = list(set(top_terms) - set(prepResume))
        print(outerSkills)


        simCosine = np.zeros(shape=(len(prepResume), len(outerSkills)))
        for i, resumeSkill in enumerate(prepResume):
            a = self.oneHotSkills.loc[:, resumeSkill].values
            for j, clustSkill in enumerate(outerSkills):
                b = self.oneHotSkills.loc[:, clustSkill].values
                simCosine[i, j] = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))

        print("Рекомедую изучить следующие навыки: (близость к вашим навыкам)")
        topInds = np.argpartition(simCosine.mean(axis=0), kth=-nRecSkills)[-nRecSkills:]
        for x in topInds:
            print(f'{outerSkills[x]}: {simCosine.mean(axis=0)[x]}', end='\n')

        resumeTokenVect = np.array([1 if token in prepResume else 0 for token in self.vocab], dtype=np.uint)
        currentClustVacs = self.oneHotSkills[(self.resDF['TopicLabel'] == clust).values]
        cosMetr = currentClustVacs.values.dot(resumeTokenVect)/np.linalg.norm(currentClustVacs.values, axis=1)
        topCos = np.argpartition(cosMetr, kth=-nRecVacs)[-nRecVacs:]
        topVacsIndex = currentClustVacs.index[topCos]

        # отсюда достать индексы и отправить в ориг датасет с них достать строки
        recVacsDF = self.resDF.iloc[topVacsIndex, :]
        dataOrig = pd.read_csv(pathOrigData, index_col=0)

        useColumns = ['Ids', 'Employer', 'Name', 'Salary', 'From', 'To', 'Experience', 'Schedule', 'Keys',
                      'Description']
        drop_columns = set(dataOrig.columns) - set(useColumns)
        dataOrig.drop(columns=drop_columns, axis=1, inplace=True)
        recDf = dataOrig.iloc[recVacsDF.index, :]
        recDf['resume similarity'] = cosMetr[topCos]
        del recDf[recDf['resume similarity'] == 0]

        saveData(recDf, pathSaveRecsVacs)





