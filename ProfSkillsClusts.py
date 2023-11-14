import numpy as np
import gensim
import re
import os

import pandas as pd

from utils import lemmatize
import pickle
import gensim.corpora as corpora


class buildLDAmodel():
    def __init__(self, prepareDF, descriptionRegexPattern: str, vocab: list, oneHotSkills: pd.DataFrame):
        self.resDF = prepareDF.copy()
        self.id2word = None
        self.model = None
        self.encodeCorpus = None
        self.descRP = descriptionRegexPattern
        self.vocab = vocab
        self.oneHotSkills = oneHotSkills

    def inference(self, resume: str) -> (int, list):
        important_words=lemmatize(resume, delSymbPattern=self.descRP, tokens=self.vocab)
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

    def fit_predict(self,
                    save_model=True,
                    modelName='LdaModel.pkl',
                    savePath = './models') -> None:
        self._prepare_LDA_input()
        modelPath = os.path.join(savePath, modelName)
        if not os.path.exists(modelPath):
            print('Создание модели LDA. Обучение модели...')
            self.model = gensim.models.LdaModel(self.encodeCorpus,
                                                id2word=self.id2word,
                                                minimum_probability=0.3,
                                                eta=0.8,
                                                num_topics=20,
                                                passes=2,
                                                random_state=0)
            if save_model:
                with open(modelPath, 'wb') as saveFile:
                    pickle.dump(self.model, saveFile)
                print(f"Файл сохранен в {modelPath}")
        else:
            print("Указанный файл уже существует, модель будет загружена с ранее сохранненого файла")
            with open(modelPath, 'rb') as saveFile:
                self.model = pickle.load(saveFile)

        self.resDF['TopicLabel'] = np.array([i[0][0] if len(i)>0 else -1 for i in self.model.get_document_topics(self.encodeCorpus)], dtype=np.int)
        self.resDF['TopicDistr'] = [i[0][1] if len(i)>0 else None for i in self.model.get_document_topics(self.encodeCorpus)]


    def recommendProfsSkillsVacs(self,
                                 resume: str,
                                 pathOrigData:str = './data/database.csv') -> None:
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

        top_terms = self.model.print_topic(clust, topn=30) # 0.042*"react" + 0.040*"js" + 0.030*"git" + 0.029*"frontend"
        top_terms = re.findall(re.compile(r'[^\d\.\"\*\s\+]+'), top_terms)
        top_terms = [x for x in top_terms if '_' not in x]

        outerSkills = list(set(top_terms) - set(prepResume))
        print(outerSkills)

        print("Рекомедую изучить следующие навыки:")
        simCosine = np.zeros(shape=(len(prepResume), len(outerSkills)))
        for i, resumeSkill in enumerate(prepResume):
            a = self.oneHotSkills.loc[:, resumeSkill].values
            for j, clustSkill in enumerate(outerSkills):
                b = self.oneHotSkills.loc[:, clustSkill].values
                simCosine[i, j] = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))
        print([outerSkills[x] for x in np.argpartition(simCosine.mean(axis=0), kth=-5)[-5:]])

        print("Рекомендуемые вакансии")
        resumeTokenVect = np.array([1 if token in prepResume else 0 for token in self.vocab], dtype=np.uint)
        currentClustVacs = self.oneHotSkills[(self.resDF['TopicLabel']==clust).values]
        cosMetr = currentClustVacs.values.dot(resumeTokenVect)/np.linalg.norm(currentClustVacs.values, axis=1)

        # отсюда достать индексы и отправить в ориг датасет с них достать строки
        recVacsDF = self.resDF.iloc[np.argpartition(cosMetr, -5)[-5:], :]
        dataOrig = pd.read_csv(pathOrigData, index_col=0)

        print(dataOrig.iloc[recVacsDF.index, :])





