"""
ベイジアンA/Bテストの実装検証スクリプト

モンテカルロサンプリングと解析的計算の結果を比較して、
実装が正確かどうかを検証します。
"""

import numpy as np
from src.test_data import TestData
from src.bayesian import BayesianABTest

# シード固定
np.random.seed(42)

print("=" * 60)
print("ベイジアンA/Bテスト実装の検証")
print("=" * 60)

# テストケース1: 明確な差がある場合
print("\n[テストケース1: 明確な差がある場合]")
data1 = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150)
print(f"グループA: {data1.n_a}訪問, {data1.conv_a}コンバージョン (CVR: {data1.cvr_a:.2%})")
print(f"グループB: {data1.n_b}訪問, {data1.conv_b}コンバージョン (CVR: {data1.cvr_b:.2%})")

test1 = BayesianABTest(data1, n_samples=100000)

# モンテカルロサンプリングによる確率計算
samples_a, samples_b = test1.sample_posterior()
prob_b_better_mc, prob_a_better_mc = test1.calculate_probability(samples_a, samples_b)

# 解析的な確率計算
prob_b_better_analytical = test1.probability_analytical()

print(f"\nモンテカルロサンプリング: P(B > A) = {prob_b_better_mc:.6f}")
print(f"解析的計算: P(B > A) = {prob_b_better_analytical:.6f}")
print(f"差の絶対値: {abs(prob_b_better_mc - prob_b_better_analytical):.6f}")

if abs(prob_b_better_mc - prob_b_better_analytical) < 0.01:
    print("✅ 検証OK: モンテカルロと解析的計算がほぼ一致")
else:
    print("❌ 検証NG: 差が大きすぎる")

# テストケース2: 微妙な差がある場合
print("\n[テストケース2: 微妙な差がある場合]")
data2 = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=115)
print(f"グループA: {data2.n_a}訪問, {data2.conv_a}コンバージョン (CVR: {data2.cvr_a:.2%})")
print(f"グループB: {data2.n_b}訪問, {data2.conv_b}コンバージョン (CVR: {data2.cvr_b:.2%})")

test2 = BayesianABTest(data2, n_samples=100000)
samples_a, samples_b = test2.sample_posterior()
prob_b_better_mc, prob_a_better_mc = test2.calculate_probability(samples_a, samples_b)
prob_b_better_analytical = test2.probability_analytical()

print(f"\nモンテカルロサンプリング: P(B > A) = {prob_b_better_mc:.6f}")
print(f"解析的計算: P(B > A) = {prob_b_better_analytical:.6f}")
print(f"差の絶対値: {abs(prob_b_better_mc - prob_b_better_analytical):.6f}")

if abs(prob_b_better_mc - prob_b_better_analytical) < 0.01:
    print("✅ 検証OK: モンテカルロと解析的計算がほぼ一致")
else:
    print("❌ 検証NG: 差が大きすぎる")

# テストケース3: 差がない場合
print("\n[テストケース3: 差がほぼない場合]")
data3 = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=105)
print(f"グループA: {data3.n_a}訪問, {data3.conv_a}コンバージョン (CVR: {data3.cvr_a:.2%})")
print(f"グループB: {data3.n_b}訪問, {data3.conv_b}コンバージョン (CVR: {data3.cvr_b:.2%})")

test3 = BayesianABTest(data3, n_samples=100000)
samples_a, samples_b = test3.sample_posterior()
prob_b_better_mc, prob_a_better_mc = test3.calculate_probability(samples_a, samples_b)
prob_b_better_analytical = test3.probability_analytical()

print(f"\nモンテカルロサンプリング: P(B > A) = {prob_b_better_mc:.6f}")
print(f"解析的計算: P(B > A) = {prob_b_better_analytical:.6f}")
print(f"差の絶対値: {abs(prob_b_better_mc - prob_b_better_analytical):.6f}")

if abs(prob_b_better_mc - prob_b_better_analytical) < 0.01:
    print("✅ 検証OK: モンテカルロと解析的計算がほぼ一致")
else:
    print("❌ 検証NG: 差が大きすぎる")

# 期待損失の検証
print("\n[期待損失の検証]")
result = test1.run()
print(f"Aを選択した場合の期待損失: {result.expected_loss_a:.6f}")
print(f"Bを選択した場合の期待損失: {result.expected_loss_b:.6f}")

# 期待損失は常に非負であるべき
if result.expected_loss_a >= 0 and result.expected_loss_b >= 0:
    print("✅ 期待損失は非負")
else:
    print("❌ 期待損失が負の値")

# 事後分布のパラメータ検証
print("\n[事後分布のパラメータ検証]")
print(f"事後分布A: Beta({test1.alpha_post_a}, {test1.beta_post_a})")
print(f"事後分布B: Beta({test1.alpha_post_b}, {test1.beta_post_b})")

# ベイズ更新の正確性を確認
expected_alpha_a = test1.alpha_prior + data1.conv_a
expected_beta_a = test1.beta_prior + (data1.n_a - data1.conv_a)
expected_alpha_b = test1.alpha_prior + data1.conv_b
expected_beta_b = test1.beta_prior + (data1.n_b - data1.conv_b)

if (test1.alpha_post_a == expected_alpha_a and
    test1.beta_post_a == expected_beta_a and
    test1.alpha_post_b == expected_alpha_b and
    test1.beta_post_b == expected_beta_b):
    print("✅ 事後分布のパラメータが正しく計算されている")
else:
    print("❌ 事後分布のパラメータに誤りがある")

print("\n" + "=" * 60)
print("検証完了")
print("=" * 60)
