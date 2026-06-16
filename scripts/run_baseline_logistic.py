from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.data.adult import load_adult, AdultSplit

from fair_safe_opt.evaluation.evaluate import evaluate_model, set_seed


def main():
    set_seed(2)
    split: AdultSplit = load_adult()
    model = LogisticRegression(split.x_train.shape[1])

    evaluate_model(model, split, method="erm", show_plots=True)


if __name__ == "__main__":
    main()
