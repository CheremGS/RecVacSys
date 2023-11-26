from DataPreparation import DataPreparation
from PecomendProcess import ModelsRunner


def main(mConfig: dict,
         resume: str,
         regrConfig: dict,
         dataPath: str = './data/database.csv',
         pathLemmasTexts: str = './data/prepdf9000.csv',
         nameClustModel: str = 'LdaModel9000.pkl',
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
                      regrConfig=regrConfig)

    cluster_num, prep_resume = mr.run_process(resume=resume)
    mr.run_recomends(clust=cluster_num,
                     prepResume=prep_resume,
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

    modelType = 'NMF'
    if modelType == 'LDA':
        modelConfig = {"num_topics": 80,
                          'eta': 0.8,
                          "alpha": 'auto',
                          "random_state": 0,
                          "update_every": 1,
                          "chunksize": 100}
        modelName = './models/LdaModel10500.pkl'

    elif modelType == 'NMF':
        modelConfig = {'n_components': 80,
                          'random_state': 0}
        modelName = './models/NMFmodel10500.pkl'


    resume = 'Знаю на хорошем уровне python, api, ml'

    regrConfig = {'text_features': ['Description'],
                  'cat_features': ['Experience', 'Schedule'],
                  'loss_function': "RMSE",
                  'learning_rate': 0.25,
                  'iterations': 300,
                  'depth': 7,
                  'verbose': 30,
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