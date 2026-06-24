import torch
import yaml
from pathlib import Path
from fair_safe_opt.utils.directories import create_report_dir
from fair_safe_opt.utils.seeds import set_seed

from fair_safe_opt.data.adult import load_adult

from fair_safe_opt.models.build import build_model

from fair_safe_opt.training.train import train_model

from fair_safe_opt.evaluation.evaluate import create_metric_table, save_evaluation_plots
from fair_safe_opt.attacks.attack import attack_features

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

def main():
    set_seed(config['seed'])
    out_dir = create_report_dir(config['architecture'], config['method'], config['seed'])

    split = load_adult(random_state=config['seed'])

    model = build_model(config['architecture'], split, config['hidden_dim'], config['dropout'])

    history = train_model(model, config['method'], split.x_train, split.y_train, split.x_val, split.y_val, out_dir, epochs=config['epochs'], patience=config['patience'], learning_rate=config['learning_rate'])

    adv_features = attack_features(model, config['attack'], split.x_test, split.y_test, split.continuous_mask, split.feature_min, split.feature_max, config['epsilon'], config['steps'], config['alpha'])
    
    with torch.no_grad():
        logits = model(adv_features)
        
    probs = torch.sigmoid(logits)
    preds = (probs >= config['threshold'])
    
    save_evaluation_plots(preds, probs, split.y_test, split.a_test, history, out_dir)
    metric_table = create_metric_table(preds, probs, split.y_test, split.a_test)
    print(metric_table)

if __name__ == "__main__":
    main()


