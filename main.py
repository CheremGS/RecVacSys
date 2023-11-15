from DataPreparation import DataPreparation
from ProfSkillsClusts import buildLDAmodel


def main(mConfig: dict,
         dataPath: str = './data/database.csv',
         pathLemmasTexts: str = './data/prepdf.csv',
         nameClustModel: str = 'LdaModel.pkl',
         saveDirModels: str = './models',
         resume: str = 'Знаю на хорошем уровне плис, soc, pild'
         ):

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

    LDAmodel.recommendProfsSkillsVacs(resume)


if __name__ == '__main__':
    modelConfig = {'minimum_probability': 0.3,
                   "eta": 0.8,
                   "num_topics": 30,
                   "passes": 20,
                   "random_state": 0}
    modelName = 'LdaModel20.pkl'
    resume = 'ml python алгоритм'

    main(mConfig=modelConfig,
         nameClustModel=modelName,
         resume=resume)