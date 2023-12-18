### Project description

Program for job recommendations. Recommendations are selected from a database stored locally.

> Program output includes:
> * Professional specialization(for example android-developer, web-developer, программист-инженер)
> * Most popular skills for return specialization
> * Most similar with job seeker and needed for prof field skills(skills recomendations)
> * .csvfile with jobs recomendations
> * .txt file with key skill in each specializations
> * .csv file with salary estimation for cv user


### Run
Clone repo or run colab link:
[Colab demo](https://colab.research.google.com/drive/1EfZaISinrE69M_mJ6qTiUmRdKxoSiS6l#scrollTo=zXxYO9fwQLA2)

> One can run either colab code neither main fuction in repo
> 1) You should choose path for source table and path for save other data and recomendations 
>    * _dataPath_ - data source 
>    * _pathLemmasTexts_ - file with processed job description 
>    * _oneHotSkills_ - table with skills labels
>    * _regrNameModel_ - path for save regression model
>    * _modelName_ - topic model save path
> 2) Define models configs 
>    * _modelConfig_ - configuration topic model 
>    * _regrConfig_ - configuration CatBoostModel
>    * _modelType_ - 'NMF' or 'LDA' topic model type
> 3) Set output parameters:
>    * _NVacRecs_ - number of saved recomendation in csv table
>    * _NskillsRecs_ - number of skill recomendations
> 4) Run main func in repo or kernel with _process input data_ and _run recomends process_ in colab

### ToDo:
> Program
> * ~~Metrics for models (ipynb, Jacarde Index Between cluster skills set (NMF), coherence in LDA)~~
> * ~~Code optimization for saving data~~
> * ~~Formatted print recomendation info~~
> * Code optimize for lemmatize text in prepare DF
> * ~~Correct prof names (exist prob for "c++ c++" name)~~
> * ~~NMF clusterization~~
> * ~~Refactor main function and model Class~~
> * ~~Filter keySkill words~~
> * ~~Use foreign data source~~

> Report preparation
> * ~~Git repo~~
> * Related work
> * Data collection process description
> * Task description
> * Solution approach description

