# ベイジアンA/Bテストの実装詳細

## 概要

このプロジェクトのベイジアンA/Bテストは、**Beta-Binomial共役モデル**を使用して実装されています。
この実装は**数学的に厳密で正確**であり、MCMCなどの近似手法は不要です。

## なぜMCMCが不要なのか？

### Beta-Binomial共役性

ベイジアン統計では、事前分布と尤度の組み合わせによっては、事後分布が解析的に（閉じた形で）求まる場合があります。
これを**共役事前分布**と呼びます。

A/Bテストの場合：
- **事前分布**: Beta分布 `p ~ Beta(α, β)`
- **尤度**: 二項分布 `conversions ~ Binomial(n, p)`
- **事後分布**: Beta分布 `p | data ~ Beta(α + conversions, β + n - conversions)`

### 数学的背景

ベイズの定理により：
```
p(θ | data) ∝ p(data | θ) × p(θ)
```

Beta-Binomial共役の場合：
```
Beta(α, β) × Binomial(k | n, p) = Beta(α + k, β + n - k)
```

この性質により、**事後分布が解析的に求まる**ため、MCMCは不要です。

## 実装の詳細

### 事後分布のパラメータ計算

```python
# 事前分布のパラメータ
alpha_prior = 1.0  # 無情報事前分布
beta_prior = 1.0   # 無情報事前分布

# 事後分布のパラメータ
alpha_post_a = alpha_prior + conv_a
beta_post_a = beta_prior + (n_a - conv_a)
alpha_post_b = alpha_prior + conv_b
beta_post_b = beta_prior + (n_b - conv_b)
```

### サンプリング

MCMCの代わりに、**Beta分布から直接サンプリング**します：
```python
samples_a = np.random.beta(alpha_post_a, beta_post_a, n_samples)
samples_b = np.random.beta(alpha_post_b, beta_post_b, n_samples)
```

これは以下の理由で正確かつ効率的です：
- ✅ Beta分布のサンプリングは高度に最適化されている
- ✅ MCMCのような収束判定が不要
- ✅ 計算が高速（100,000サンプルでも瞬時）
- ✅ 理論的に正確（近似ではない）

### 確率計算

P(B > A)の計算には2つの方法を実装しています：

#### 1. モンテカルロサンプリング（デフォルト）
```python
prob_b_better = np.mean(samples_b > samples_a)
```

#### 2. 解析的計算（より正確）
```python
def probability_analytical(self):
    def integrand(x):
        return (
            stats.beta.pdf(x, alpha_post_a, beta_post_a) *
            stats.beta.cdf(x, alpha_post_b, beta_post_b)
        )
    result, _ = integrate.quad(integrand, 0, 1)
    return 1 - result
```

数式：
```
P(p_B > p_A) = ∫₀¹ f_A(x) × [1 - F_B(x)] dx
             = 1 - ∫₀¹ f_A(x) × F_B(x) dx
```

ここで：
- f_A(x): グループAの事後分布の確率密度関数
- F_B(x): グループBの事後分布の累積分布関数

## 検証結果

`verify_bayesian.py`スクリプトを実行した結果：

### テストケース1: 明確な差がある場合
- グループA: 1000訪問, 100コンバージョン (CVR: 10.00%)
- グループB: 1000訪問, 150コンバージョン (CVR: 15.00%)

```
モンテカルロサンプリング: P(B > A) = 0.999700
解析的計算: P(B > A) = 0.999642
差の絶対値: 0.000058
✅ 検証OK
```

### テストケース2: 微妙な差がある場合
- グループA: 1000訪問, 100コンバージョン (CVR: 10.00%)
- グループB: 1000訪問, 115コンバージョン (CVR: 11.50%)

```
モンテカルロサンプリング: P(B > A) = 0.860460
解析的計算: P(B > A) = 0.860069
差の絶対値: 0.000391
✅ 検証OK
```

### テストケース3: 差がほぼない場合
- グループA: 1000訪問, 100コンバージョン (CVR: 10.00%)
- グループB: 1000訪問, 105コンバージョン (CVR: 10.50%)

```
モンテカルロサンプリング: P(B > A) = 0.644380
解析的計算: P(B > A) = 0.643451
差の絶対値: 0.000929
✅ 検証OK
```

### 期待損失の検証
```
Aを選択した場合の期待損失: 0.049910
Bを選択した場合の期待損失: 0.000001
✅ 期待損失は非負
```

### 事後分布のパラメータ検証
```
事後分布A: Beta(101.0, 901.0)
事後分布B: Beta(151.0, 851.0)
✅ 事後分布のパラメータが正しく計算されている
```

## ベイズファクター（オッズ比）

現在の実装では、以下の簡易版のベイズファクター（オッズ比）を計算しています：

```
BF = P(B > A) / P(A > B)
```

### 注意点
これは厳密なベイズファクター `P(Data | H1) / P(Data | H0)` ではなく、オッズ比です。
しかし、実用上は「BがAより優れている」証拠の強さの指標として十分有用です。

### 解釈の目安
- BF < 1: Aが優れている証拠
- BF = 1: どちらとも言えない
- 1 < BF < 3: Bが優れている弱い証拠
- 3 < BF < 10: Bが優れている中程度の証拠
- BF > 10: Bが優れている強い証拠

## MCMCが必要になる場合

以下の場合にはMCMCが必要になります：

1. **非共役事前分布を使用する場合**
   - 例: 正規分布の事前分布と二項分布の尤度

2. **階層モデル（Hierarchical Model）**
   - 例: 複数の実験結果を同時にモデル化

3. **複雑なモデル**
   - 例: 時系列依存性、相関構造など

4. **カスタムモデル**
   - 例: ドメイン知識に基づく複雑な事前分布

現在のシンプルなA/Bテストでは、**Beta-Binomial共役モデルで十分**であり、
MCMCを使用すると計算コストが増えるだけで、精度は向上しません。

## 参考文献

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
2. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
3. Cameron Davidson-Pilon (2015). *Bayesian Methods for Hackers*. Addison-Wesley.
4. [Conjugate prior - Wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior)
5. [Beta-Binomial distribution - Wikipedia](https://en.wikipedia.org/wiki/Beta-binomial_distribution)

## まとめ

✅ **現在の実装は数学的に正確です**
✅ **MCMCは不要です**（Beta-Binomial共役性により事後分布が解析的に求まるため）
✅ **検証済みです**（モンテカルロと解析的計算が一致）
✅ **高速です**（MCMCより圧倒的に速い）
✅ **理論的に正しいです**（近似ではなく、厳密な計算）

もしMCMCの実装が必要になる場合（非共役事前分布、階層モデルなど）は、
PyMC3やStanなどのライブラリを使用することを推奨します。
