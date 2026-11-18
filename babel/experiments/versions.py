"""比較実験で使う各版の関門の設定。後続の版で追加された関門は無効にする。"""

VARIANTS = {}
for minor in range(2, 9):
    VARIANTS[f"v1.{minor}"] = {
        "null_gates": minor >= 3,
        "affix_null_alpha": 0.05 if minor <= 3 else 0.025,
        "v15_gates": minor >= 5,
        "v16_gates": minor >= 6,
        "v17_gates": minor >= 7,
        "v18_gates": minor >= 8,
    }
