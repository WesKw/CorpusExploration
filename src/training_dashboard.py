import json
import pandas as pd
import glob
import seaborn as sb
import re

from pathlib import Path
from simple_parsing import parse
from ExplorationArgs import DashboardArgs


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
KV_RE = re.compile(r'(\w+):\s*(\S+)')

def parse_message(msg: str) -> dict:
    clean = ANSI_RE.sub('', msg)
    metrics = dict(KV_RE.findall(clean))

    out = {}
    if 'step' in metrics:
        out['step'] = int(metrics['step'])
    if 'loss' in metrics:
        out['loss'] = float(metrics['loss'])
    if 'grad_norm' in metrics:
        out['grad_norm'] = float(metrics['grad_norm'])
    if 'tps' in metrics:
        out['tps'] = int(metrics['tps'].replace(',', ''))
    if 'tflops' in metrics:
        out['tflops'] = float(metrics['tflops'])
    if 'mfu' in metrics:
        out['mfu'] = float(metrics['mfu'].rstrip('%'))
    if 'memory' in metrics:
        m = re.match(r'([\d.]+)(\w+)\(([\d.]+)%\)', metrics['memory'])
        if m:
            out['memory_value'] = float(m.group(1))
            out['memory_unit'] = m.group(2)
            out['memory_pct'] = float(m.group(3))

    return out

# expand parsed metrics into new columns, aligned to df's index
metrics_df = df['message'].apply(parse_message).apply(pd.Series)
df = pd.concat([df, metrics_df], axis=1)


def generate_pd_df(ezpz_logs: list):
    """Generate a pandas dataframe from the training logs."""
    ezpz_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    for log in ezpz_logs:
        sub_df = pd.read_json(log, lines=True)
        basename = Path(log).name
        sub_df["log_file"] = basename
        # hardcoded based on file name
        metadata = basename
        metadata = metadata.replace("_model", "").split("_")
        sub_df["experiment"] = metadata[0]
        sub_df["config"] = metadata[1]
        sub_df["steps"] = int(metadata[2])
        sub_df["ordering"] = metadata[3].split(".")[0]
        ezpz_df = pd.concat([ezpz_df, sub_df], ignore_index=True)

        # load the lm_eval logs based on metadata
        file_name = f"{metadata[0]}_{metadata[1]}_{metadata[2]}_{metadata[3].split('.')[0]}_lm_eval.json"
        with open(Path(log).parent / file_name, "r") as f:
            lm_eval_data = json.load(f)
            for task in lm_eval_data["results"]:
                lm_eval_df = pd.from_dict(task, orient="index").reset_index()
                eval_df = pd.concat([eval_df, lm_eval_df], ignore_index=True)

    return (ezpz_df,eval_df)


def generate_dashboard(ezpz_data: pd.DataFrame, lm_eval_data: pd.DataFrame, ezpz_metrics: list, output_dir: Path):

    # use metrics module for ezpz logs
    ezpz_df = ezpz_data[data["module"] == "metrics"] # only concerned with metrics module
    # split out the message for each metric (thanks claude)
    metrics = ezpz_df['message'].apply(parse_message).apply(pd.Series)
    ezpz_df = pd.concat([ezpz_df, metrics], axis=1)
    ezpz_df = ezpz_df.sort_values(by=["experiment", "config", "ordering", "steps", "step"])

    # create dashboard for ezpz logs
    sb.set_theme(style="whitegrid")
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    for metric in enumerate(ezpz_metrics):
        sb.lineplot(data=ezpz_df, x="step", y=metric, hue="ordering", style="config", ax=axes[idx])
    
    # for lm bench uhhh use the other log
    for task in lm_eval_data["task"].unique():
        task_df = lm_eval_data[lm_eval_data["task"] == task]
        sb.barplot(data=task_df, x="config", y="accuracy", hue="ordering", ax=axes[0][idx])
    
    sb.barplot(data=lm_eval_data, x="task", y="accuracy", hue="ordering", style="config", ax=axes[1][0])

    fig.tight_layout()
    fig.savefig(output_dir / "training_dashboard.png")


if __name__ == "__main__":
    args = parse(DashboardArgs)

    ezpz_logs = glob.glob(args.ezpz_logs_glob)
    ezpz_logs = [Path(log) for log in ezpz_logs]

    ezpz_data,lm_eval_data = generate_pd_df(ezpz_logs)

    ezpz_metrics = args.ezpz_metrics
    llm_eval_tasks = args.lm_eval_tasks
    generate_dashboard(ezpz_data, lm_eval_data, ezpz_metrics, llm_eval_tasks, Path(args.output_dir))
