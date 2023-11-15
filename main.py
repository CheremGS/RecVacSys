from DataPreparation import DataPreparation
from ProfSkillsClusts import buildLDAmodel


def main(mConfig: dict,
         dataPath: str = './data/database9000.csv',
         pathLemmasTexts: str = './data/prepdf9000.csv',
         nameClustModel: str = 'LdaModel9000.pkl',
         saveDirModels: str = './models',
         resume: str = 'Знаю на хорошем уровне плис, soc, pild',
         Nrecs: int = 10):

    dataPrep = DataPreparation(dataPath)
    dataPrep.run(baseTokenIsSkills=True,
                 pathSaveLemmasTexts=pathLemmasTexts)

    preparedDf = dataPrep.prepDF
    LDAmodel = buildLDAmodel(prepareDF=preparedDf,
                             descriptionRegexPattern=dataPrep.regexPatterns['Description'],
                             vocab=dataPrep.skillSet,
                             oneHotSkills=dataPrep.oneHotSkills)

    LDAmodel.fit_predict(modelName=nameClustModel,
                         savePath=saveDirModels,
                         modelConfig=mConfig)

    LDAmodel.recommendProfsSkillsVacs(resume,
                                      nRecVacs=Nrecs,
                                      pathOrigData=dataPath)


if __name__ == '__main__':
    modelConfig = {'minimum_probability': 0.3,
                   "num_topics": 45,
                   'eta': 0.65,
                   "alpha": 'auto',
                   "random_state": 0,
                   "update_every": 1,
                   "chunksize": 100
                   }
    modelName = 'LdaModel9000.pkl'
    resume = 'Программирую на C++. Опыт работы с stm32, jetson nano'

    main(mConfig=modelConfig,
         nameClustModel=modelName,
         resume=resume)