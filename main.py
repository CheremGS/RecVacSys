from DataPreparation import DataPreparation
from ProfSkillsClusts import buildLDAmodel

dataPrep = DataPreparation('./data/database.csv')
dataPrep.run(baseTokenIsSkills=True)

preparedDf = dataPrep.prepDF
LDAmodel = buildLDAmodel(prepareDF=preparedDf,
                         descriptionRegexPattern=dataPrep.regexPatterns['Description'],
                         vocab=dataPrep.skillSet,
                         oneHotSkills=dataPrep.oneHotSkills)
LDAmodel.fit_predict()

resume = 'Знаю на хорошем уровне плис, soc, pild'
LDAmodel.recommendProfsSkills(resume)