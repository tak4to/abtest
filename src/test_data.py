from dataclasses import dataclass
from enum import Enum


class TestMethod(Enum):
    """統計検定の方法"""
    Z_TEST = "z_test"
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"


@dataclass
class TestData:
    """
    A/Bテストのデータ
    
    Attributes
    ----------
    n_a : int
        グループAのサンプルサイズ
    conv_a : int
        グループAのコンバージョン数
    n_b : int
        グループBのサンプルサイズ
    conv_b : int
        グループBのコンバージョン数
    """
    n_a: int
    conv_a: int
    n_b: int
    conv_b: int
    
    def __post_init__(self):
        """データの妥当性チェック"""
        if self.n_a <= 0 or self.n_b <= 0:
            raise ValueError("サンプルサイズは正の整数である必要があります")
        if self.conv_a < 0 or self.conv_b < 0:
            raise ValueError("コンバージョン数は非負整数である必要があります")
        if self.conv_a > self.n_a:
            raise ValueError(
                f"コンバージョン数({self.conv_a})はサンプルサイズ({self.n_a})を超えることはできません"
            )
        if self.conv_b > self.n_b:
            raise ValueError(
                f"コンバージョン数({self.conv_b})はサンプルサイズ({self.n_b})を超えることはできません"
            )
    
    @property
    def cvr_a(self) -> float:
        """グループAのコンバージョン率"""
        return self.conv_a / self.n_a
    
    @property
    def cvr_b(self) -> float:
        """グループBのコンバージョン率"""
        return self.conv_b / self.n_b
    
    @property
    def cvr_diff(self) -> float:
        """コンバージョン率の差 (B - A)"""
        return self.cvr_b - self.cvr_a
