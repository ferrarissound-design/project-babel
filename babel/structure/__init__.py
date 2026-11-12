"""意味の情報を使わない、未知の記号列の構造発見（PROJECT BABEL の独立モジュール）。

解読器（babel.decipher）とは独立している。翻訳・意味の推定はしない。
観測された規則（統計的に偶然では説明しにくい並び）と、そこから推測できる文法的機能
（未検証）を必ず区別して出力する。

このパッケージは babel.generator / babel.evaluate / babel.structure_eval を import しない
（tests/test_structure.py で強制）。正解は評価側だけが持つ。
"""

VERSION = "0.1.4"
