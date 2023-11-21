from DataPreparation import DataPreparation
from ProfSkillsClusts import modelProcess


def main(mConfig: dict,
         resume: str,
         dataPath: str = './data/database.csv',
         pathLemmasTexts: str = './data/prepdf9000.csv',
         nameClustModel: str = 'LdaModel9000.pkl',
         saveDirModels: str = './models',
         Nrecs: int = 10,
         NrecSkills: int = 10,
         modelType: str = 'LDA',
         oneHotSkillsPath: str = './data/oneHotSkills9000.csv'):

    dataPrep = DataPreparation(dataPath)
    dataPrep.run(baseTokenIsSkills=True,
                 pathSaveLemmasTexts=pathLemmasTexts,
                 oneHotSavePath=oneHotSkillsPath)

    preparedDf = dataPrep.prepDF
    topicModel = modelProcess(prepareDF=preparedDf,
                              descriptionRegexPattern=dataPrep.regexPatterns['Description'],
                              vocab=dataPrep.skillSet,
                              oneHotSkills=dataPrep.oneHotSkills,
                              modelType=modelType)
    if modelType == 'LDA':
        topicModel.LDA_fit_predict(modelName=nameClustModel,
                                   savePath=saveDirModels,
                                   modelConfig=mConfig)
    elif modelType == 'NMF':
        topicModel.NMF_fit_predict(modelName=nameClustModel,
                                   savePath=saveDirModels,
                                   modelConfig=mConfig)

    topicModel.recommendProfsSkillsVacs(resume=resume,
                                        nRecVacs=Nrecs,
                                        nRecSkills=NrecSkills,
                                        pathOrigData=dataPath)


if __name__ == '__main__':
    dataPath: str = './data/db10500.csv'
    pathLemmasTexts: str = './data/prepdf10500.csv'
    saveDirModels: str = './models'
    oneHotSkill: str = './data/OHS10500.csv'
    NVacRecs: int = 20
    NskillsRecs: int = 7

    LDAmodelConfig = {"num_topics": 80,
                      'eta': 0.8,
                      "alpha": 'auto',
                      "random_state": 0,
                      "update_every": 1,
                      "chunksize": 100}
    LDAmodelName = 'LdaModel10500.pkl'

    NMFmodelConfig = {'n_components': 80,
                      'random_state': 0}
    NMFmodelName = 'NMFmodel10500.pkl'

    modelType = 'NMF'
    resume = 'Знаю на хорошем уровне плис, микроконтроллер, stm32'

    main(dataPath=dataPath,
         pathLemmasTexts=pathLemmasTexts,
         saveDirModels=saveDirModels,
         Nrecs=NVacRecs,
         NrecSkills=NskillsRecs,
         mConfig=NMFmodelConfig,
         nameClustModel=NMFmodelName,
         resume=resume,
         modelType=modelType,
         oneHotSkillsPath=oneHotSkill)