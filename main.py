from DataPreparation import DataPreparation
from ProfSkillsClusts import buildLDAmodel


def main(dataPath: str = './data/database.csv',
         pathLemmasTexts: str = './data/prepdf.csv',
         nameClustModel: str = 'LdaModel.pkl',
         saveDirModels: str = './models',
         resume: str = 'Знаю на хорошем уровне плис, soc, pild'):

    dataPrep = DataPreparation(dataPath)
    dataPrep.run(baseTokenIsSkills=True,
                 pathSaveLemmasTexts=pathLemmasTexts)

    preparedDf = dataPrep.prepDF
    LDAmodel = buildLDAmodel(prepareDF=preparedDf,
                             descriptionRegexPattern=dataPrep.regexPatterns['Description'],
                             vocab=dataPrep.skillSet,
                             oneHotSkills=dataPrep.oneHotSkills)

    LDAmodel.fit_predict(modelName=nameClustModel,
                         savePath=saveDirModels)

    LDAmodel.recommendProfsSkillsVacs(resume)


if __name__ == '__main__':
    main()