import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_curve_results(curve_results_glob="./benchmark/curve_results/*.json") -> pd.DataFrame:
	rows = []
	for fp in glob.glob(curve_results_glob):
		with open(fp, "r") as f:
			d = json.load(f)
		r = d.get("results", d)
		row = {
			"file": str(fp),
			"database": d.get("database", r.get("database")),
			"dataset": d.get("dataset", r.get("dataset")),
			"M": d.get("m", r.get("params", {}).get("M")),
			"ef_construction": d.get("ef_construction", r.get("params", {}).get("ef_construction")),
			"ef_search": d.get("ef_search", r.get("params", {}).get("ef_search")),
			"vectors": r.get("vectors"),
			"dim": r.get("dim"),
			"queries": r.get("queries"),
			"build_latency_s": r.get("build_latency", d.get("build_latency")),
			"recall_at_10": r.get("recall@10"),
			"avg_latency_s": r.get("search_avg_latency"),
			"p50_latency_s": r.get("search_p50_latency"),
			"p95_latency_s": r.get("search_p95_latency"),
			"p99_latency_s": r.get("search_p99_latency"),
			"qps": r.get("qps"),
			"build_mem_peak_mb": r.get("build_mem_peak_mb", d.get("build_mem_peak_mb")),
			"search_mem_peak_mb_avg": r.get("search_mem_peak_mb_avg"),
		}
		rows.append(row)

	return pd.DataFrame(rows)


def plot_recall_latency_tradeoff(
	df: pd.DataFrame,
	latency_metric: str = "p99_latency_s",
	group_by: str = "database",
	dataset: str = None,
	figsize: tuple = (12, 8),
	title: str = None,
):

	if dataset is not None:
		df = df[df["dataset"] == dataset]
		if df.empty:
			print(f"No data found for dataset: {dataset}")
			return

	df_clean = df.dropna(subset=["recall_at_10", latency_metric])

	if df_clean.empty:
		print("No valid data to plot!")
		return

	sns.set_style("whitegrid")
	fig, ax = plt.subplots(figsize=figsize)

	groups = df_clean[group_by].unique()

	markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
	linestyles = ['-']

	for idx, group in enumerate(groups):
		group_data = df_clean[df_clean[group_by] == group].sort_values("recall_at_10")

		marker_style = markers[idx % len(markers)]
		line_style = linestyles[idx % len(linestyles)]

		ax.plot(
			group_data["recall_at_10"],
			group_data[latency_metric] * 1000,
			marker=marker_style,
			linestyle=line_style,
			linewidth=2,
			markersize=8,
			label=group,
			color='black',
			markerfacecolor='white',
			markeredgewidth=1.5,
			markeredgecolor='black'
		)

	ax.set_xlabel("Recall@10", fontsize=12, fontweight='bold')
	ax.set_ylabel(f"Latency (ms)", fontsize=12, fontweight='bold')

	if title is None:
		latency_name = latency_metric.replace("_latency_s", "").replace("_", " ").upper()
		title = f"Recall vs {latency_name} Latency Trade-off"

	ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
	ax.legend(title=group_by.capitalize(), fontsize=10, title_fontsize=11)
	ax.grid(True, alpha=0.3)

	min_recall = df_clean["recall_at_10"].min()
	x_min = max(0, min_recall - 0.10)
	ax.set_xlim(x_min, 1.0)
	ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))

	plt.tight_layout()

	plt.savefig("./plots/latency_recall_curve.png", dpi=900, bbox_inches='tight')


def plot_memory_usage(
	df: pd.DataFrame,
	group_by: str = "database",
	dataset: str = None,
	figsize: tuple = (12, 8),
	title: str = None,
	save_path: str = None
):

	if dataset is not None:
		df = df[df["dataset"] == dataset]
		if df.empty:
			print(f"No data found for dataset: {dataset}")
			return
	
	df_clean = df.dropna(subset=["build_mem_peak_mb", "search_mem_peak_mb_avg"])
	
	if df_clean.empty:
		print("No valid memory data to plot!")
		return
	
	grouped = df_clean.groupby(group_by).agg({
		'build_mem_peak_mb': 'mean',
		'search_mem_peak_mb_avg': 'mean'
	}).reset_index()
	
	sns.set_style("whitegrid")
	fig, ax = plt.subplots(figsize=figsize)
	
	x = range(len(grouped))
	width = 0.35
	
	bars1 = ax.bar(
		[i - width/2 for i in x],
		grouped['build_mem_peak_mb'],
		width,
		label='Build Memory',
		color='white',
		edgecolor='black',
		linewidth=1.5,
		hatch='//'
	)
	
	bars2 = ax.bar(
		[i + width/2 for i in x],
		grouped['search_mem_peak_mb_avg'],
		width,
		label='Search Memory',
		color='lightgray',
		edgecolor='black',
		linewidth=1.5,
		hatch='\\\\'
	)
	
	ax.set_xlabel(group_by.capitalize(), fontsize=12, fontweight='bold')
	ax.set_ylabel("Memory Usage (MB)", fontsize=12, fontweight='bold')
	
	if title is None:
		title = "Memory Usage: Build vs Search"
		if dataset:
			title += f" ({dataset})"
	
	ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
	ax.set_xticks(x)
	ax.set_xticklabels(grouped[group_by], rotation=45, ha='right')
	ax.legend(fontsize=10)
	ax.grid(True, alpha=0.3, axis='y')
	
	for bars in [bars1, bars2]:
		for bar in bars:
			height = bar.get_height()
			ax.text(
				bar.get_x() + bar.get_width() / 2.,
				height,
				f'{height:.1f}',
				ha='center',
				va='bottom',
				fontsize=9
			)
	
	plt.tight_layout()
	
	plt.savefig("./plots/memory_bars.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
	df = load_curve_results("./benchmark/curve_results/*.json")

	print(f"Loaded {len(df)} results")
	print(f"\nDatabases: {df['database'].unique()}")
	print(f"Datasets: {df['dataset'].unique()}")

	plot_recall_latency_tradeoff(
		df,
		latency_metric="p95_latency_s",
		group_by="database",
		dataset="sift-128",
		title="Recall vs Latency Trade-off"
	)

	plot_memory_usage(
		df,
		group_by="database",
		dataset="sift-128"
	)