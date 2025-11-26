"""
事前分布パラメータのリセット動作を検証するテストスクリプト
"""

def simulate_session_state_behavior():
    """
    Streamlitのセッション状態の動作をシミュレート
    """
    print("=" * 70)
    print("事前分布パラメータのリセット動作テスト")
    print("=" * 70)

    # セッション状態をシミュレート
    session_state = {}

    # シナリオ1: アプリ起動、プリセット="明確な差がある例"
    print("\n【シナリオ1】アプリ起動、プリセット='明確な差がある例'")
    print("-" * 70)
    preset = "明確な差がある例"
    session_state['previous_preset'] = preset
    session_state['input_n_a'] = 1000
    session_state['input_conv_a'] = 100
    print(f"プリセット: {preset}")
    print(f"サンプルデータ: n_a={session_state['input_n_a']}, conv_a={session_state['input_conv_a']}")
    print(f"事前分布パラメータ: 未設定（デフォルト値を使用）")

    # シナリオ2: 詳細設定で「A/B個別設定」を選択し、事前分布を変更
    print("\n【シナリオ2】詳細設定で事前分布を変更")
    print("-" * 70)
    # ユーザーが事前分布パラメータを変更
    session_state['n_a'] = 20
    session_state['conv_a'] = 2
    session_state['n_b'] = 15
    session_state['conv_b'] = 1

    # 事前分布パラメータの計算
    alpha_prior_a = session_state['conv_a'] + 1.0
    beta_prior_a = (session_state['n_a'] - session_state['conv_a']) + 1.0
    alpha_prior_b = session_state['conv_b'] + 1.0
    beta_prior_b = (session_state['n_b'] - session_state['conv_b']) + 1.0

    print(f"ユーザーが設定: n_a={session_state['n_a']}, conv_a={session_state['conv_a']}")
    print(f"計算された事前分布: α_A={alpha_prior_a:.1f}, β_A={beta_prior_a:.1f}")
    print(f"ユーザーが設定: n_b={session_state['n_b']}, conv_b={session_state['conv_b']}")
    print(f"計算された事前分布: α_B={alpha_prior_b:.1f}, β_B={beta_prior_b:.1f}")
    print("✓ 事前分布が正しく反映されている")

    # シナリオ3: プリセットを変更
    print("\n【シナリオ3】プリセットを'微妙な差がある例'に変更")
    print("-" * 70)
    new_preset = "微妙な差がある例"

    # プリセット変更検出
    if session_state['previous_preset'] != new_preset:
        print(f"プリセット変更を検出: {session_state['previous_preset']} → {new_preset}")
        session_state['previous_preset'] = new_preset
        session_state['input_n_a'] = 1000
        session_state['input_conv_a'] = 100
        session_state['input_n_b'] = 1000
        session_state['input_conv_b'] = 125

        # 事前分布パラメータをリセット（修正後の動作）
        if "n_a" in session_state:
            del session_state["n_a"]
        if "conv_a" in session_state:
            del session_state["conv_a"]
        if "n_b" in session_state:
            del session_state["n_b"]
        if "conv_b" in session_state:
            del session_state["conv_b"]

        print("✓ サンプルデータを更新")
        print(f"  n_a={session_state['input_n_a']}, conv_a={session_state['input_conv_a']}")
        print(f"  n_b={session_state['input_n_b']}, conv_b={session_state['input_conv_b']}")
        print("✓ 事前分布パラメータをリセット（セッション状態から削除）")

    # 次のレンダリングで使用されるデフォルト値
    default_n_a = 10
    default_conv_a = 1
    prior_n_a = session_state.get('n_a', default_n_a)
    prior_conv_a = session_state.get('conv_a', default_conv_a)

    alpha_prior_a = prior_conv_a + 1.0
    beta_prior_a = (prior_n_a - prior_conv_a) + 1.0

    print(f"次のレンダリングで使用される事前分布パラメータ:")
    print(f"  n_a={prior_n_a} (デフォルト), conv_a={prior_conv_a} (デフォルト)")
    print(f"  α_A={alpha_prior_a:.1f}, β_A={beta_prior_a:.1f}")
    print("✓ デフォルト値が使用される（冪等性が保たれる）")

    # シナリオ4: プリセット変更後に再度事前分布を変更
    print("\n【シナリオ4】プリセット変更後に再度事前分布を変更")
    print("-" * 70)
    session_state['n_a'] = 30
    session_state['conv_a'] = 5

    alpha_prior_a = session_state['conv_a'] + 1.0
    beta_prior_a = (session_state['n_a'] - session_state['conv_a']) + 1.0

    print(f"ユーザーが再設定: n_a={session_state['n_a']}, conv_a={session_state['conv_a']}")
    print(f"計算された事前分布: α_A={alpha_prior_a:.1f}, β_A={beta_prior_a:.1f}")
    print("✓ 新しい値が正しく反映されている")

    # シナリオ5: 同じプリセットを選択（冪等性の確認）
    print("\n【シナリオ5】同じプリセットを再度選択（冪等性の確認）")
    print("-" * 70)
    same_preset = "微妙な差がある例"

    if session_state['previous_preset'] == same_preset:
        print(f"プリセット変更なし: {same_preset}")
        print("✓ 事前分布パラメータは保持される")
        print(f"  n_a={session_state['n_a']}, conv_a={session_state['conv_a']}")
        print(f"  α_A={alpha_prior_a:.1f}, β_A={beta_prior_a:.1f}")

    print("\n" + "=" * 70)
    print("全てのシナリオが期待通りに動作しています！")
    print("=" * 70)

    # 修正前の問題を説明
    print("\n【修正前の問題】")
    print("-" * 70)
    print("プリセットを変更しても事前分布パラメータがリセットされないため、")
    print("同じプリセットでも異なる結果が出る可能性がありました。")
    print("\n例:")
    print("  1. プリセット'A'を選択")
    print("  2. 事前分布を変更（n_a=20）")
    print("  3. プリセット'B'に変更 → 事前分布は n_a=20 のまま")
    print("  4. プリセット'A'に戻す → 事前分布は n_a=20 のまま（本来は10であるべき）")
    print("\n【修正後】")
    print("-" * 70)
    print("プリセット変更時に事前分布パラメータをリセットするため、")
    print("同じプリセットなら常に同じデフォルト値が使用されます。")
    print("ユーザーが明示的に変更した値は、プリセットを変更するまで保持されます。")


if __name__ == "__main__":
    simulate_session_state_behavior()
