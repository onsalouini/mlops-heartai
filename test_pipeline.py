from model_pipeline import prepare_data, train_model, evaluate_model

def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    print("test_prepare_data OK")

def test_train_model():
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    assert model is not None
    print("test_train_model OK")

def test_evaluate_model():
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    print("test_evaluate_model OK")
