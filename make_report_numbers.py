import json
import re
from pathlib import Path

# Writes report/numbers.tex (one LaTeX macro per reported number) and the
# report tables in report/tables/ from the saved result files. No result is
# computed here: every value is read from data_processed/ and only rounded.
# A key the script expects but cannot find stops it with the key and file.
DATA_DIR = Path("data_processed")
TRUST_DIR = DATA_DIR / "trust"
REPORT_DIR = Path("report")
TABLE_DIR = REPORT_DIR / "tables"

# Added by the top-30 experiment; its macros and rows are skipped if absent.
OPTIONAL_METHOD = "question_focus_relative_top30"

SETTINGS = (
    "kind1", "kind2", "kind1_one_poisoned", "kind2_one_poisoned",
    "reworded", "diverse",
)
SETTING_LABELS = {
    "kind1": "Prefix",
    "kind2": "No prefix",
    "kind1_one_poisoned": "Prefix, 1",
    "kind2_one_poisoned": "No prefix, 1",
    "reworded": "Reworded",
    "diverse": "Diverse",
}
LABELS = {
    "retrieval_score": "Retrieval score",
    "retrieval_z": "Retrieval z-score",
    "set_density": "Set density",
    "shortness": "Short passage length",
    "perplexity": "Perplexity",
    "reader_support": "Reader support",
    "z_drop": "Relevance consistency",
    "z_drop_k3": "Relevance consistency",
    "combined_consistency": "Combined consistency",
    "question_focus": "Question focus",
    "question_sameness": "Question sameness",
    "question_focus_relative": "Set-relative focus",
    OPTIONAL_METHOD: "Set-relative focus (top 30)",
    "retrieval+focus": "Retrieval + focus",
    "rank_mean": "Rank mean (with focus)",
    "rank_max": "Rank max (with focus)",
    "rank_mean_no_focus": "Rank mean (no focus)",
    "rank_max_no_focus": "Rank max (no focus)",
    "rank_mean_relative": "Rank mean (relative focus)",
    "rank_max_relative": "Rank max (relative focus)",
    "no_defence": "No defence",
    "remove_all_poison": "Remove all poison",
}
ROBUSTNESS_METHODS = (
    "z_drop", "set_density", "reader_support", "retrieval_score",
    "shortness", "perplexity", "question_sameness", "question_focus",
    OPTIONAL_METHOD, "question_focus_relative", "combined_consistency",
    "retrieval+focus", "rank_mean_no_focus", "rank_max_no_focus",
    "rank_mean", "rank_max", "rank_mean_relative", "rank_max_relative",
)
DETECTION_METHODS = (
    "retrieval_score", "retrieval_z", "set_density", "shortness",
    "perplexity", "reader_support", "z_drop_k3", "combined_consistency",
)
END_TO_END_METHODS = (
    "no_defence", "remove_all_poison", "retrieval_score", "set_density",
    "shortness", "perplexity", "reader_support", "z_drop_k3",
    "combined_consistency", "question_focus", "retrieval+focus",
    "question_focus_relative", OPTIONAL_METHOD, "rank_max",
)
READERS = {
    "Qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "Extractive": "extractive_reader",
}
COST_METHODS = (
    "retrieval_score", "shortness", "set_density", "perplexity",
    "reader_support", "z_drop_k3", "combined_consistency", "question_focus",
)
# Header over the two groups of three columns in the detection and
# end-to-end tables.
KIND_GROUPS = (
    " & \\multicolumn{3}{c}{Prefix} & \\multicolumn{3}{c}{No prefix} \\\\"
    " \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}"
)
NUMBER_WORDS = {
    "1": "One", "2": "Two", "3": "Three", "5": "Five", "10": "Ten",
    "30": "Thirty", "95": "NinetyFive", "100": "Hundred",
}


def load(path):
    return {"path": path, "data": json.loads(Path(path).read_text())}


def get(source, *keys):
    value = source["data"]
    for depth, key in enumerate(keys):
        if not isinstance(value, (dict, list)) or (
            isinstance(value, dict) and key not in value
        ):
            path = "/".join(map(str, keys[: depth + 1]))
            raise SystemExit(f"missing key {path!r} in {source['path']}")
        value = value[key]
    return value


def camel(name):
    # LaTeX macro names hold letters only: "kind1_one_poisoned" becomes
    # "KindOneOnePoisoned" and "retrieval+focus" becomes "RetrievalPlusFocus".
    name = name.replace("+", "_plus_").replace("@", "_")
    parts = re.findall(r"[A-Za-z]+|\d+", name)
    for number in (p for p in parts if p.isdigit()):
        if number not in NUMBER_WORDS:
            raise SystemExit(f"no word for the number {number} in {name!r}")
    return "".join(NUMBER_WORDS.get(p, p.capitalize()) for p in parts)


def three(value):
    return f"{value:.3f}"


def rate(value):
    # Rates over 100 questions are whole percentages, so two decimals are exact.
    return f"{value:.2f}"


def p_value(value, bootstraps=None):
    if value == 0:
        # No resample reached the observed difference: report the bound, not 0.
        assert bootstraps, "p-value 0 needs the bootstrap count for a bound"
        return f"\\ensuremath{{<}}{1 / bootstraps:g}"
    text = f"{value:#.2g}"
    assert "e" not in text, f"p-value {value} needs a different format"
    return text


def percent(value):
    return f"{100 * value:.2f}"


def milliseconds(value):
    return f"{value:.1f}"


class Macros:
    def __init__(self):
        self.values = {}

    def add(self, name, text):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        assert name not in self.values, f"duplicate macro {name}"
        assert not re.search(r"nan|inf", text), f"\\{name} is {text}"
        self.values[name] = text

    def write(self, path):
        lines = [
            "% Generated by make_report_numbers.py; do not edit by hand.",
            *(
                f"\\newcommand{{\\{name}}}{{{text}}}"
                for name, text in sorted(self.values.items())
            ),
        ]
        path.write_text("\n".join(lines) + "\n")


def tabular(columns, header, rows, groups=""):
    lines = [
        "% Generated by make_report_numbers.py; do not edit by hand.",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        *([groups] if groups else []),
        " & ".join(header) + " \\\\",
        "\\midrule",
        *(" & ".join(row) + " \\\\" for row in rows),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def present(methods, source, *prefix):
    # Every method must exist, except the optional top-30 one.
    available = get(source, *prefix)
    kept = []
    for method in methods:
        if method in available:
            kept.append(method)
        elif method == OPTIONAL_METHOD:
            print(f"skipped {method}: not in {source['path']}")
        else:
            get(source, *prefix, method)
    return kept


def pair_name(key):
    # "rank_max - set_density (worst AUROC)" becomes "RankMaxVsSetDensity".
    first, second = key.removesuffix(" (worst AUROC)").split(" - ")
    return f"{camel(first)}Vs{camel(second)}"


def robustness(macros, tables):
    source = load(TRUST_DIR / "robustness_results.json")
    methods = present(ROBUSTNESS_METHODS, source, "methods")
    for method in methods:
        name = camel(method)
        summary = ("methods", method)
        low, high = get(source, *summary, "worst_auroc_ci_95")
        macros.add(
            f"WorstAuroc{name}", three(get(source, *summary, "worst_auroc"))
        )
        macros.add(f"WorstAurocLow{name}", three(low))
        macros.add(f"WorstAurocHigh{name}", three(high))
        macros.add(
            f"MeanAuroc{name}", three(get(source, *summary, "mean_auroc"))
        )
        macros.add(
            f"WorstSetting{name}",
            SETTING_LABELS[get(source, "methods", method, "worst_setting")],
        )
        for setting in SETTINGS:
            macros.add(
                f"Auroc{camel(setting)}{name}",
                three(get(source, "settings", setting, method, "auroc")),
            )
            macros.add(
                f"Tpr{camel(setting)}{name}",
                three(get(source, "settings", setting, method,
                          "tpr_at_5pct_fpr")),
            )

    bootstraps = get(source, "setup", "bootstraps")
    for key in sorted(get(source, "comparisons")):
        if OPTIONAL_METHOD not in methods and OPTIONAL_METHOD in key:
            continue
        name = pair_name(key)
        result = ("comparisons", key)
        low, high = get(source, *result, "ci_95")
        macros.add(
            f"WorstDiff{name}", three(get(source, *result, "auroc_difference"))
        )
        macros.add(f"WorstDiffLow{name}", three(low))
        macros.add(f"WorstDiffHigh{name}", three(high))
        macros.add(
            f"WorstP{name}",
            p_value(get(source, *result, "p_value"), bootstraps),
        )

    holm = load(TRUST_DIR / "holm_corrected_pvalues.json")
    for test in get(holm, "families", "robustness"):
        test = {"path": holm["path"], "data": test}
        if not get(test, "test").startswith("comparisons/"):
            continue
        macros.add(
            f"WorstHolmP{pair_name(get(test, 'test').removeprefix('comparisons/'))}",
            p_value(get(test, "p_holm")),
        )

    macros.add("Bootstraps", f"{bootstraps:,}")
    macros.add(
        "OnePoisonedDraws", str(get(source, "setup", "one_poisoned_draws"))
    )

    ordered = sorted(
        methods, key=lambda m: get(source, "methods", m, "worst_auroc")
    )
    rows = [
        [
            LABELS[method],
            *(three(get(source, "settings", s, method, "auroc"))
              for s in SETTINGS),
            three(get(source, "methods", method, "worst_auroc")),
            three(get(source, "methods", method, "mean_auroc")),
        ]
        for method in ordered
    ]
    header = ["Method", *(SETTING_LABELS[s] for s in SETTINGS), "Worst", "Mean"]
    tables["robustness"] = tabular("l" + "r" * 8, header, rows)


def detection(macros, tables):
    source = load(TRUST_DIR / "detection_results.json")
    kinds = {"Prefix": "prefix/all_benign", "Noprefix": "noprefix/all_benign"}
    for kind, key in kinds.items():
        macros.add(f"NumPoisoned{kind}", str(get(source, key, "poisoned")))
        macros.add(f"NumBenign{kind}", str(get(source, key, "benign")))
        for method in DETECTION_METHODS:
            name = f"{kind}{camel(method)}"
            for metric, field in (
                ("Auroc", "auroc"), ("Auprc", "auprc"),
                ("Tpr", "tpr_at_5pct_fpr"),
            ):
                macros.add(
                    f"Detect{metric}{name}",
                    three(get(source, key, "methods", method, field)),
                )
            low, high = get(source, key, "methods", method, "auroc_ci_95")
            macros.add(f"DetectAurocLow{name}", three(low))
            macros.add(f"DetectAurocHigh{name}", three(high))

    rows = [
        [
            LABELS[method],
            *(
                three(get(source, key, "methods", method, field))
                for key in kinds.values()
                for field in ("auroc", "auprc", "tpr_at_5pct_fpr")
            ),
        ]
        for method in DETECTION_METHODS
    ]
    header = [
        "Signal", "AUROC", "AUPRC", "TPR", "AUROC", "AUPRC", "TPR",
    ]
    tables["detection"] = tabular("l" + "r" * 6, header, rows, KIND_GROUPS)


def end_to_end(macros, tables):
    source = load(TRUST_DIR / "reverse_end_to_end_results.json")
    kinds = {"Prefix": "prefix", "Noprefix": "noprefix"}
    methods = present(END_TO_END_METHODS, source, READERS["Qwen"], "prefix")
    for reader, reader_key in READERS.items():
        for kind, kind_key in kinds.items():
            for method in methods:
                name = f"{reader}{kind}{camel(method)}"
                attacked = (reader_key, kind_key, method, "attacked")
                clean = (reader_key, kind_key, method, "clean")
                macros.add(
                    f"Asr{name}",
                    rate(get(source, *attacked, "attack_success", "rate")),
                )
                macros.add(
                    f"Acc{name}",
                    rate(get(source, *attacked, "correct", "rate")),
                )
                macros.add(
                    f"CleanAcc{name}",
                    rate(get(source, *clean, "correct", "rate")),
                )

    setup = ("setup",)
    macros.add("RetrieveK", str(get(source, *setup, "retrieve_k")))
    macros.add("ReadK", str(get(source, *setup, "read_k")))
    macros.add("TargetFpr", rate(get(source, *setup, "target_fpr")))
    macros.add(
        "GoldInCleanTopTen", str(get(source, *setup, "gold_in_clean_top10"))
    )
    macros.add("NumQuestions", str(get(source, *setup, "questions")))

    qwen = READERS["Qwen"]
    extractive = READERS["Extractive"]
    rows = [
        [
            LABELS[method],
            *(
                rate(get(source, reader, kind, method, "attacked",
                         field, "rate"))
                for kind in kinds.values()
                for reader, field in (
                    (qwen, "attack_success"), (qwen, "correct"),
                    (extractive, "attack_success"),
                )
            ),
        ]
        for method in methods
    ]
    header = ["Defence", "ASR", "Acc.", "ASR (ext.)",
              "ASR", "Acc.", "ASR (ext.)"]
    tables["end_to_end"] = tabular("l" + "r" * 6, header, rows, KIND_GROUPS)


def cost(macros, tables):
    source = load(TRUST_DIR / "cost_results.json")
    macros.add(
        "PlainRetrievalMs",
        milliseconds(get(source, "plain_retrieval", "median_ms")),
    )
    macros.add("PoolSize", f"{get(source, 'pool_size'):,}")
    macros.add(
        "CandidatesPerQuestion", f"{get(source, 'candidates_per_question')}"
    )
    rows = []
    for method in COST_METHODS:
        name = camel(method)
        median = milliseconds(get(source, "extra_cost", method, "median_ms"))
        times = f"{get(source, 'extra_cost', method, 'times_plain_retrieval'):.2f}"
        macros.add(f"CostMs{name}", median)
        macros.add(f"CostTimesPlain{name}", times)
        rows.append([LABELS[method], median, times])
    offline = milliseconds(
        get(source, "offline_cost", "question_focus", "median_ms")
    )
    macros.add("OfflineCostMsQuestionFocus", offline)
    rows.append(["Question focus (offline generation)", offline, "--"])
    tables["cost"] = tabular(
        "lrr", ["Signal", "Median ms", "$\\times$ retrieval"], rows
    )


def paraphrase(macros, tables):
    source = load(DATA_DIR / "experiment_retrieval_metrics.json")
    baseline = load(DATA_DIR / "baseline_retrieval_metrics.json")
    significance = load(DATA_DIR / "paraphrase_significance_results.json")
    metrics = ("hit@1", "hit@5", "hit@10", "hit@100", "mrr@10")
    for group in ("original", "paraphrase"):
        for metric in metrics:
            value = get(source, "metrics", group, metric)
            text = percent(value) if metric.startswith("hit") else three(value)
            macros.add(f"{camel(metric)}{camel(group)}", text)
    for metric in metrics:
        value = get(baseline, "metrics", metric)
        text = percent(value) if metric.startswith("hit") else three(value)
        macros.add(f"Baseline{camel(metric)}", text)
    macros.add(
        "NumParaphraseQueries",
        str(get(source, "group_counts", "paraphrase")),
    )
    for key in ("hit10", "hit100", "mrr10"):
        name = camel(key.replace("hit", "hit_").replace("mrr", "mrr_"))
        result = percent if key.startswith("hit") else three
        difference = get(significance, key, "mean_difference")
        low, high = get(significance, key, "ci_95")
        macros.add(f"ParaphraseDiff{name}", result(difference))
        macros.add(f"ParaphraseDiffLow{name}", result(low))
        macros.add(f"ParaphraseDiffHigh{name}", result(high))
        macros.add(f"ParaphraseP{name}", p_value(get(significance, key,
                                                      "p_value")))

    rows = []
    for metric, key, label in (
        ("hit@10", "hit10", "Hit@10 (\\%)"),
        ("hit@100", "hit100", "Hit@100 (\\%)"),
        ("mrr@10", "mrr10", "MRR@10"),
    ):
        fmt = percent if metric.startswith("hit") else three
        rows.append([
            label,
            fmt(get(source, "metrics", "original", metric)),
            fmt(get(source, "metrics", "paraphrase", metric)),
            p_value(get(significance, key, "p_value")),
        ])
    tables["paraphrase"] = tabular(
        "lrrr", ["Metric", "Original", "Paraphrased", "$p$"], rows
    )


def self_check(macros, tables):
    # Headline values checked by hand against the results files on 17 Oct.
    expected = {
        "WorstAurocZDrop": "0.389", "WorstAurocSetDensity": "0.389",
        "WorstAurocReaderSupport": "0.527",
        "WorstAurocRetrievalScore": "0.570", "WorstAurocShortness": "0.626",
        "WorstAurocPerplexity": "0.680", "WorstAurocQuestionFocus": "0.703",
        "WorstAurocQuestionFocusRelative": "0.774",
        "WorstAurocRankMaxNoFocus": "0.870", "WorstAurocRankMax": "0.871",
        "WorstAurocRankMaxRelative": "0.874",
        "MeanAurocZDrop": "0.518", "MeanAurocSetDensity": "0.761",
        "MeanAurocReaderSupport": "0.729",
        "MeanAurocRetrievalScore": "0.834", "MeanAurocShortness": "0.809",
        "MeanAurocPerplexity": "0.718", "MeanAurocQuestionFocus": "0.840",
        "MeanAurocQuestionFocusRelative": "0.902",
        "MeanAurocRankMaxNoFocus": "0.947", "MeanAurocRankMax": "0.952",
        "MeanAurocRankMaxRelative": "0.954",
        "WorstSettingSetDensity": "No prefix, 1",
        "AsrQwenPrefixNoDefence": "0.96",
        "AsrQwenPrefixRetrievalScore": "0.20",
        "AsrQwenPrefixSetDensity": "0.08", "AsrQwenPrefixRankMax": "0.14",
        "AsrQwenPrefixQuestionFocus": "0.35",
        "AsrQwenPrefixRetrievalPlusFocus": "0.10",
        "AsrQwenPrefixQuestionFocusRelative": "0.89",
        "AsrQwenNoprefixNoDefence": "0.76",
        "AsrQwenNoprefixReaderSupport": "0.39",
        "AsrQwenNoprefixRankMax": "0.43",
        "AsrQwenNoprefixSetDensity": "0.51",
        "AsrExtractivePrefixQuestionFocusRelative": "0.73",
        "AurocKindOneSetDensity": "0.990",
        "HitTenOriginal": "31.00", "HitTenParaphrase": "30.67",
        "HitHundredOriginal": "79.00", "HitHundredParaphrase": "89.00",
        "ParaphrasePHitHundred": "0.0016",
        "NumQuestions": "100", "PoolSize": "9,872",
        # A bootstrap p of 0 is reported as a bound, never as 0.
        "WorstPRankMaxVsSetDensity": "\\ensuremath{<}0.0001",
        "WorstHolmPRankMaxVsRankMaxNoFocus": "1.0",
    }
    if f"WorstAuroc{camel(OPTIONAL_METHOD)}" in macros.values:
        expected |= {
            "WorstAurocQuestionFocusRelativeTopThirty": "0.760",
            "MeanAurocQuestionFocusRelativeTopThirty": "0.891",
            "WorstAurocLowQuestionFocusRelativeTopThirty": "0.719",
            "WorstAurocHighQuestionFocusRelativeTopThirty": "0.800",
            "AsrQwenPrefixQuestionFocusRelativeTopThirty": "0.43",
            "AsrQwenNoprefixQuestionFocusRelativeTopThirty": "0.69",
            "AsrExtractivePrefixQuestionFocusRelativeTopThirty": "0.25",
            "AsrExtractiveNoprefixQuestionFocusRelativeTopThirty": "0.66",
        }
    for name, value in expected.items():
        assert macros.values.get(name) == value, (
            f"\\{name} is {macros.values.get(name)!r}, expected {value!r}"
        )
    # Table cells come from separate code, so check a row of each against
    # its macros to catch swapped columns.
    m = macros.values
    assert (
        f"No defence & {m['AsrQwenPrefixNoDefence']} & "
        f"{m['AccQwenPrefixNoDefence']} & {m['AsrExtractivePrefixNoDefence']}"
        f" & {m['AsrQwenNoprefixNoDefence']}"
    ) in tables["end_to_end"]
    assert (
        f"Rank max (with focus) & {m['AurocKindOneRankMax']} & "
        f"{m['AurocKindTwoRankMax']}"
    ) in tables["robustness"]
    assert (
        f"Set density & {m['DetectAurocPrefixSetDensity']} & "
        f"{m['DetectAuprcPrefixSetDensity']} & {m['DetectTprPrefixSetDensity']}"
        f" & {m['DetectAurocNoprefixSetDensity']}"
    ) in tables["detection"]
    try:
        get({"path": "test.json", "data": {"a": {}}}, "a", "b")
    except SystemExit as error:
        assert str(error) == "missing key 'a/b' in test.json", error
    else:
        raise AssertionError("a missing key did not stop the script")
    assert camel("kind2_one_poisoned") == "KindTwoOnePoisoned"


def main():
    macros = Macros()
    tables = {}
    for build in (robustness, detection, end_to_end, cost, paraphrase):
        build(macros, tables)
    self_check(macros, tables)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    macros.write(REPORT_DIR / "numbers.tex")
    for name, text in tables.items():
        (TABLE_DIR / f"{name}.tex").write_text(text)
    print(f"wrote {len(macros.values)} macros and {len(tables)} tables")


if __name__ == "__main__":
    main()
