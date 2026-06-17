from fair_safe_opt.evaluation.evaluate import evaluate_model


def main():
    evaluate_model(architecture='logistic', method="erm", seed=0, epochs=10)


if __name__ == "__main__":
    main()
