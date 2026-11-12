"""構造解析（babel.structure）の評価。正解（設計した規則・単語境界・母集団）はこのパッケージだけが持つ。

babel.structure はこのパッケージを import しない（tests/test_structure_isolation.py で強制）。
解析器に渡すのは記号列だけで、単語境界・規則・生成器の設定は渡さない。
"""
