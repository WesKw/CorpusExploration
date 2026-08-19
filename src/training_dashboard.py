import json
import pandas as pd
import glob
import seaborn as sb
import re
import matplotlib.pyplot as plt

from pathlib import Path
from simple_parsing import parse
from ExplorationArgs import DashboardArgs


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
KV_RE = re.compile(r'(\w+):\s*(\S+)')
EVALUATION_FILENAME_RE = re.compile(
    r'^(?P<model>[^_]+)_(?P<config>[^_]+)_(?P<ordering>[^_]+)_step-(?P<step>\d+)_evaluation\.json$'
)
LOG_FILENAME_RE = re.compile(
    r'^(?P<model>[^_]+)_(?P<config>[^_]+)_(?P<ordering>[^_]+)\.jsonl$'
)


def parse_evaluation_filename(filepath: Path) -> dict:
    """Extract model, config, ordering, and step from an evaluation json filename."""
    name = filepath.name
    match = EVALUATION_FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Filename does not match expected evaluation format: {name}")
    return {
        "model": match.group("model"),
        "config": match.group("config"),
        "ordering": match.group("ordering"),
        "step": int(match.group("step")),
    }


def parse_log_filename(filepath: Path) -> dict:
    """Extract model, config, and ordering from a training log jsonl filename."""
    name = filepath.name
    match = LOG_FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Filename does not match expected log format: {name}")
    return {
        "model": match.group("model"),
        "config": match.group("config"),
        "ordering": match.group("ordering"),
    }


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
# metrics_df = df['message'].apply(parse_message).apply(pd.Series)
# df = pd.concat([df, metrics_df], axis=1)
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


def label_line_ends(ax, data: pd.DataFrame, x: str, y: str, group_cols: list, x_offset_px: float = 8, min_gap_px: float = 14):
    """Annotate the right end of each line with its group label, staggering labels
    vertically (in pixel space) so overlapping line-ends don't produce overlapping text."""
    points = []
    for group_vals, group_df in data.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_df = group_df.dropna(subset=[x, y])
        if group_df.empty:
            continue
        last_row = group_df.sort_values(x).iloc[-1]
        # label = "/".join(str(v) for v in group_vals)
        label=str(group_vals[0])
        points.append((label, last_row[x], last_row[y]))

    if not points:
        return

    # sort by y so we can push overlapping labels apart in a consistent direction
    points.sort(key=lambda p: p[2])
    disp_pts = [ax.transData.transform((px, py)) for _, px, py in points]
    disp_ys = [pt[1] for pt in disp_pts]
    for i in range(1, len(disp_ys)):
        if disp_ys[i] - disp_ys[i - 1] < min_gap_px:
            disp_ys[i] = disp_ys[i - 1] + min_gap_px

    inv = ax.transData.inverted()
    for (label, px, py), (disp_x, _), disp_y in zip(points, disp_pts, disp_ys):
        text_x, text_y = inv.transform((disp_x + x_offset_px, disp_y))
        ax.annotate(
            label,
            xy=(px, py),
            xytext=(text_x, text_y),
            va="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5, shrinkA=0, shrinkB=2),
        )


def generate_dashboard(loss_df: pd.DataFrame, eval_df: pd.DataFrame, eval_tasks: list, output_dir: Path):
    # use metrics module for ezpz logs
    ezpz_df = loss_df[loss_df["module"] == "metrics"] # only concerned with metrics module
    # split out the message for each metric (thanks claude)
    metrics = ezpz_df['message'].apply(parse_message).apply(pd.Series)
    ezpz_df = pd.concat([ezpz_df, metrics], axis=1)
    ezpz_df = ezpz_df.sort_values(by=["model", "config", "order", "step"])

    # create dashboard for ezpz logs
    sb.set_theme(style="whitegrid")
    # left column: loss metrics (2 rows). right column: eval task metrics (3 rows).
    fig,axes = plt.subplots(3, 2, figsize=(16,10))
    loss_axes = axes[:2, 0]
    eval_axes = axes[:, 1]
    legend_ax = axes[2, 0]  # unused cell below the loss/grad_norm column
    legend_ax.axis("off")

    loss_handles, loss_labels = None, None
    for idx,metric in enumerate(["loss", "grad_norm"]):
        plot_data = ezpz_df
        ax = loss_axes[idx]
        plot = sb.lineplot(data=plot_data, x="step", y=metric, hue="order", style="config", ax=ax)
        ax.set(yscale='log')
        plot.set_title(f"{metric} curve")
        ax.margins(x=0.12)

        # every axis gets its own full hue+style legend by default, which is far
        # too big to show inline. Grab the handles once and drop the per-axis legend
        # in favor of a single shared legend placed in the empty cell below.
        loss_handles, loss_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    eval_handles, eval_labels = None, None

    # arc_easy and openbookqa share an axis, so give them distinct markers on
    # the "name" (task) column -- style="config" only has one level, which is
    # why markers=["s", "+"] wasn't doing anything before.
    norm_acc_df = eval_df[eval_df["name"].isin(["arc_easy", "openbookqa"])]
    if not norm_acc_df.empty:
        plot = sb.lineplot(data=norm_acc_df, x="step", y="acc_norm,none", hue="order", style="name",
                            markers={"arc_easy": "o", "openbookqa": "s"}, dashes=False, ax=eval_axes[0])
        plot.set_title("Accuracy [arc_easy, openbookqa]")

    norm_acc_df = eval_df[eval_df["name"].isin(["lambada_openai", "truthfulqa_mc1", "truthfulqa_mc2"])]
    if not norm_acc_df.empty:
        plot = sb.lineplot(data=norm_acc_df, x="step", y="acc,none", hue="order", style="name",
                            markers={"lambada_openai": "o", "truthfulqa_mc1": "s", "truthfulqa_mc2": "p"}, 
                            dashes=False, ax=eval_axes[1])
        plot.set_title("Accuracy [lambada_openai, truthfulqa_mc1, truthfulqa_mc2]")

    # for lm bench uhhh use the other log
    for task in eval_tasks:
        task_df = eval_df[eval_df["name"] == task]

        # word and byte perplexity
        if task in ["wikitext"]:
            plot = sb.lineplot(data=task_df, x="step", y="bits_per_byte,none", hue="order", style="config", ax=eval_axes[2])
            plot.set_title("Bits-Per-Byte [wikitext]")


    #     sb.barplot(data=task_df, x="config", y="accuracy", hue="ordering", style="task", ax=axes[0][idx])

    # sb.barplot(data=lm_eval_data, x="task", y="accuracy", hue="ordering", style="config", ax=axes[1][0])

    # same problem as the loss column: drop each eval subplot's own giant legend
    # and keep a single shared one instead. Different eval axes have different
    # marker sets (task names), so merge handles from all of them rather than
    # just keeping the last axis -- otherwise the task/marker names get dropped.
    eval_handles, eval_labels = [], []
    seen_eval_labels = set()
    for ax in eval_axes:
        if ax.has_data():
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in seen_eval_labels:
                    eval_handles.append(handle)
                    eval_labels.append(label)
                    seen_eval_labels.add(label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.suptitle("Training Dashboard for Token Orderings")
    # finalize the subplot grid layout FIRST. Line-end label placement below depends
    # on each axis's on-screen (pixel) position via transData, so it has to happen
    # after layout is settled -- otherwise the later tight_layout call shifts the
    # axes and the carefully-staggered labels no longer line up, causing overlap.
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    for idx, metric in enumerate(["loss", "grad_norm"]):
        label_line_ends(loss_axes[idx], ezpz_df, x="step", y=metric, group_cols=["order", "config"])

    if loss_handles:
        legend_ax.legend(loss_handles, loss_labels, loc="center", fontsize=8, frameon=False, title="order/config")
    if eval_handles:
        fig.legend(eval_handles, eval_labels, loc="center left", bbox_to_anchor=(0.9, 0.5),
                   bbox_transform=fig.transFigure, fontsize=8, frameon=False, title="order / task / config")

    fig.savefig(output_dir / "training_dashboard.png", bbox_inches="tight")


def generate_loss_df(loss_logs: list) -> pd.DataFrame:
    """Generate a pandas dataframe from the training logs."""
    loss_df = pd.DataFrame()

    for log in loss_logs:
        sub_df = pd.read_json(log, lines=True)
        basename = log.name
        log_metadata = parse_log_filename(log)
        sub_df["log_file"] = basename
        # hardcoded based on file name
        sub_df["model"] = log_metadata["model"]
        sub_df["config"] = log_metadata["config"]
        sub_df["order"] = log_metadata["ordering"]
        loss_df = pd.concat([loss_df, sub_df], ignore_index=True)

    return loss_df

def generate_eval_df(eval_logs: list) -> pd.DataFrame:
    eval_df = pd.DataFrame()

    for log in eval_logs:
        with open(log, "r") as f:
            lm_eval_data = json.load(f)
            for task in lm_eval_data["results"]:
                sub_df = pd.DataFrame.from_dict(lm_eval_data["results"][task], orient="index").reset_index()
                eval_metadata = parse_evaluation_filename(log)
                # sub_df = sub_df.T
                sub_df = sub_df.T
                sub_df.columns = sub_df.iloc[0]
                sub_df = sub_df.drop(sub_df.index[0])
                sub_df["model"] = eval_metadata["model"]
                sub_df["config"] = eval_metadata["config"]
                sub_df["order"] = eval_metadata["ordering"]
                sub_df["step"] = eval_metadata["step"]
                # lm_eval_df = pd.from_dict(task, orient="index").reset_index()
                # df.index, df.columns = df.columns, df.index
                eval_df = pd.concat([eval_df, sub_df], ignore_index=True)

    eval_df = eval_df.drop(labels=["alias"], axis=1)
    # print(eval_df)
    return eval_df

if __name__ == "__main__":
    args = parse(DashboardArgs)

    loss_logs = glob.glob(args.loss_logs_glob)
    loss_logs = [Path(log) for log in loss_logs]
    eval_logs = glob.glob(args.eval_logs_glob)
    eval_logs = [Path(log) for log in eval_logs]

    loss_df = generate_loss_df(loss_logs)
    eval_df = generate_eval_df(eval_logs)

    eval_tasks = args.lm_eval_tasks
    generate_dashboard(loss_df, eval_df, eval_tasks, Path(args.output_dir))
