import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML Heart Disease")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.prepare:
        X_train, X_test, y_train, y_test = prepare_data()

    if args.all or args.train:
        if "X_train" not in dir():
            X_train, X_test, y_train, y_test = prepare_data()
        model = train_model(X_train, y_train)

    if args.all or args.evaluate:
        if "model" not in dir():
            model = load_model()
        if "X_test" not in dir():
            _, X_test, _, y_test = prepare_data()
        evaluate_model(model, X_test, y_test)

    if args.all or args.save:
        if "model" not in dir():
            print("Lance --train d'abord !")
            return
        save_model(model)

    if args.load:
        load_model()


if __name__ == "__main__":
    main()
