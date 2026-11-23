# 凍結済み実験の再現

推論・採点・記事分割・採用基準は `results/role_recovery/development-freeze.json` のハッシュで固定されている。
`python -m babel.experiments.role_recovery_report` はその固定を検査して報告書を再生成する。

最初の結果を置き換えずに推論を再実行するには、検証専用の次のコマンドを使う。

```sh
PYTHONHASHSEED=0 python -m babel.experiments.reproduce_role_study --phase dev --out /tmp/reproduce-dev.json
PYTHONHASHSEED=0 python -m babel.experiments.reproduce_role_study --phase final --out /tmp/reproduce-final.json
```

既定では日本語seed0の通常対訳、ガリシア語seed2の誤対応を再実行する。
`--all` を付けると、その分割の全120条件を再実行する。
最初の入力・予測ハッシュと、時間以外の**全集計値**が一致しなければ失敗する。既存の予測と異なる内容の上書きも拒否する。
詳細予測は `.cache/role_recovery/locked/` に再生成される。最初の結果ファイルは変更しない。

この検証ユーティリティは開発凍結後に追加したもので、凍結済みの推論と採点を呼び出して比較するだけである。
実験の方式・評価データ・閾値・採用判定は変更しない。GitHubの集計JSONは生成物として折りたたまれるが、全件を収録している。
