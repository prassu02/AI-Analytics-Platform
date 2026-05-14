from sklearn.ensemble import IsolationForest


def detect_anomalies(X):

    model = IsolationForest()

    anomalies = model.fit_predict(X)

    return anomalies.tolist()