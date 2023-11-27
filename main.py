from DataPreparation import DataPreparation
from PecomendProcess import ModelsRunner


def main(mConfig: dict,
         resume: str,
         regrConfig: dict,
         dataPath: str = './data/database.csv',
         pathLemmasTexts: str = './data/prepdf9000.csv',
         nameClustModel: str = 'LdaModel9000.pkl',
         regrNameModel: str = './models/CatBoostModel10500.cbm',
         Nrecs: int = 10,
         NrecSkills: int = 10,
         modelType: str = 'LDA',
         oneHotSkillsPath: str = './data/oneHotSkills9000.csv'):

    dataPrep = DataPreparation(dataPath)
    dataPrep.run(baseTokenIsSkills=True,
                 pathSaveLemmasTexts=pathLemmasTexts,
                 oneHotSavePath=oneHotSkillsPath)

    mr = ModelsRunner(prepareDF=dataPrep.prepDF,
                      descriptionRegexPattern=dataPrep.regexPatterns['Description'],
                      vocab=dataPrep.skillSet,
                      oneHotSkills=oneHotSkillsPath,
                      modelType=modelType,
                      modelPath=nameClustModel,
                      modelConfig=mConfig,
                      regrConfig=regrConfig,
                      regrModelPath=regrNameModel)

    cluster_num, prep_resume = mr.run_process(resume=resume)
    mr.run_recomends(clust=cluster_num,
                     prepResume=prep_resume,
                     nRecVacs=Nrecs,
                     nRecSkills=NrecSkills,
                     pathOrigData=dataPath)


if __name__ == '__main__':
    dataPath: str = './data/foreigndb_20000.csv'
    pathLemmasTexts: str = './data/prepdf20000.csv'
    oneHotSkill: str = './data/OHS20000.csv'
    NVacRecs: int = 20
    NskillsRecs: int = 7

    modelType = 'LDA'
    if modelType == 'LDA':
        modelConfig = {"num_topics": 100,
                          'eta': 0.8,
                          "alpha": 'auto',
                          "random_state": 0,
                          "update_every": 1,
                          "chunksize": 100}
        modelName = './models/LdaModel20.pkl'

    elif modelType == 'NMF':
        modelConfig = {'n_components': 80,
                          'random_state': 0}
        modelName = './models/NMFmodel10500.pkl'

    regrModelName = './models/CatBoostModel20.cbm'
    resume = 'Знаю на хорошем уровне python, api, ml'

    regrConfig = {'text_features': ['Description'],
                  'cat_features': ['Experience', 'Schedule'],
                  'loss_function': "RMSE",
                  'learning_rate': 0.25,
                  'iterations': 300,
                  'depth': 7,
                  'verbose': False,
                  'random_state': 0,
                  'task_type': "GPU"}

    main(regrConfig=regrConfig,
         dataPath=dataPath,
         pathLemmasTexts=pathLemmasTexts,
         Nrecs=NVacRecs,
         NrecSkills=NskillsRecs,
         mConfig=modelConfig,
         nameClustModel=modelName,
         resume=resume,
         modelType=modelType,
         oneHotSkillsPath=oneHotSkill)