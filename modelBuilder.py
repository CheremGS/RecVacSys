import pandas as pd
from catboost import CatBoostRegressor, Pool


class CatBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = CatBoostRegressor(**config)

    def train(self, X, y, printLoss=True):
        self.model.fit(X, y)

        self.infer_table = X.groupby(['Schedule', 'Experience'], as_index=False).size()
        if printLoss:
            # print relative RMSE
            print(f"Best model relative RMSEloss: {self.model.best_score_['learn']['RMSE'] / (y.max() - y.min())}")

    def inference(self, resume):
        del self.infer_table['size']
        self.infer_table['Description'] = [resume]*self.infer_table.shape[0]
        self.infer_table['Salary'] = self.model.predict(self.infer_table)
        del self.infer_table['Description']

        return self.infer_table