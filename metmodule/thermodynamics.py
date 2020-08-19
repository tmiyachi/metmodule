"""
熱力学関連計算モジュール
------------------------

.. autosummary::

    potential_temperature
    vapor_pressure
    saturation_vapor_pressure
    mixing_ratio
    dewpoint_from_e
    dewpoint_from_rh
    wetbulb
    dry_lapse
    moist_lapse
"""
import numpy as np
from scipy.integrate import odeint

from constants import kappa, epsilon, water_es_0c, air_Rd, water_Lv_0c, air_Cp_d


def potential_temperature(p, t, base_p=1.e5):
    r"""
    ポテンシャル温度（温位）を計算する．

    Parameters
    ----------
    p : array_like
        気圧 [Pa]
    t : array_like
        気温 [K]
    base_p : float, optional
        基準気圧．デフォルトは1000hPa．

    Returns
    -------
    array_like
        ポテンシャル温度 [K]

    Notes
    -----
    計算式:

    .. math:: \theta = T \left(\frac{p_0}{p}\right)^\kappa
    """

    return t * (base_p / p) ** kappa


def vapor_pressure(p, r):
    """
    水蒸気圧を計算する．

    Parameters
    ----------
    p : array_like
        気圧 [Pa]
    r : array_like
        混合比 [kg/kg]

    Returns
    -------
    array_like
        水蒸気圧 [Pa]

    See Also
    --------
    saturation_vapor_pressure
    """

    return p * r / (epsilon + r)


def saturation_vapor_pressure(t):
    r"""
    飽和水蒸気圧を計算する．

    気温が0度以上の場合は水に対する飽和水蒸気圧を，0度未満の場合は氷に対する水蒸気圧を求める．

    Parameters
    ----------
    t : array_like
        気温 [K]

    Returns
    -------
    array_like
        飽和水蒸気圧 [Pa]

    Notes
    -----
    計算式:

    .. math:: e_s = e_0 \exp\left[ \frac{17.67 T[deg] }{T[deg] + 243.5} \right] ~~ (T \le 0)

    .. math:: e_s = e_0 \exp\left[ \frac{22.46 T[deg]) }{T + 272.62} \right] ~~ (T < 0)

    References
    ----------
    Bolton, D., 1980: The Computation of Equivalent Potential Temperature. *Mon. Wea. Rev.*, 108, 1046-1053
    """
    over_liquid = water_es_0c * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    over_ice = water_es_0c * np.exp(22.46 * (t - 273.15) / (t - 0.53))

    if np.isscalar(t):
        if t >= 273.15:
            return over_liquid
        else:
            return over_ice
    else:
        return np.where(t < 273.15, over_ice, over_liquid)


def mixing_ratio(e, p):
    r"""
    混合比を計算する．

    Parameters
    ----------
    e : array_like
        水蒸気圧 [Pa]
    p : array_like
        気圧 [Pa]

    Returns
    -------
    array_like
        混合比 [kg/kg]

    Notes
    -----
    計算式

    .. math:: r = \frac{\epsilon e}{p - e}
    """

    return epsilon * e / (p - e)


def dewpoint_from_e(e):
    r"""
    露点温度を水蒸気圧から計算する．

    Parameters
    ----------
    e : array_like
        水蒸気圧 [Pa]

    Returns
    -------
    array_like
        露点温度 [K]

    Notes
    -----
    Boltonの式で `e=es` としたときの `T` が `Td` となる:

    .. math:: Td = \frac{243.5 \ln(e / e_0)}{17.67 - \ln(e / e_0)} [deg]

    See Also
    --------
    saturation_vapor_pressure, dewpoint_from_rh
    """
    val = np.log(e / water_es_0c)
    return 243.5 * val / (17.67 - val) + 273.15


def dewpoint_from_rh(t, rh):
    """
    露点温度を湿度から計算する．

    Parameters
    ----------
    t : array_like
        気温 [K]
    rh : array_like
        相対湿度 [%]

    Returns
    -------
    array_like
        露点温度 [K]

    See Also
    --------
    dewpoint_from_e
    """
    return dewpoint_from_e(rh * saturation_vapor_pressure(t) / 100.)


def dry_lapse(p, t):
    """
    乾燥断熱過程で気塊を上昇下降させたときの気温を求める．

    気塊は `p[0]` , `t` から移動させて計算し `p` における各気温，
    すなわち `p[0]` , `t` を通る乾燥断熱線を求める．

    Parameters
    ----------
    p : array_like
        求める気圧の座標．先頭は `t` における気圧． [Pa]
    t : array_like
        持ち上げる気塊の気温 [K]

    Returns
    -------
    array_like
        気塊を持ち上げたときの `p` における気温.
    """
    return t * (p / p[0]) ** kappa


def moist_lapse(p, t):
    r"""
    湿潤断熱過程で気塊を上昇下降させたときの気温を求める．

    気塊は `p[0]` , `t` から移動させて計算し全過程で飽和しているとみなし `p` における各気温，
    すなわち `p[0]` , `t` を通る湿潤断熱線を求める．

    Parameters
    ----------
    p : array_like
        求める気圧の座標．先頭は `t` における気圧． [Pa]
    t : array_like
        持ち上げる気塊の気温 [K]

    Returns
    -------
    array_like
        気塊を持ち上げたときの `p` における気温.

    Notes
    -----
    次の湿潤断熱過程における微分方程式から求める．

    .. math:: \frac{dT}{dp} = \frac{b}{p} \frac{R_d T + L_v r_s}
                              {C_{pd} + \frac{L_v^2 r_s \epsilon b}{R_d T^2}}

    .. math:: b = 1 - 0.24r_{s} \simeq 1

    References
    ----------
    Bakhshaii, A. and R. Stull, 2013: Saturated Pseudoadiabats--A Noniterative Approximation.
    *J. Appl. Meteor. Clim.*, 52, 5-15.
    """
    def dT_dP(T, P):
        rs = mixing_ratio(saturation_vapor_pressure(T), P)
        # b  = 1. - 0.24 * rs
        b = 1
        return b / P * (air_Rd * T + water_Lv_0c * rs) / (air_Cp_d +
                                                          (water_Lv_0c * water_Lv_0c * rs * epsilon * b) / air_Rd / T / T)

    return odeint(dT_dP, np.atleast_1d(t).squeeze(), p.squeeze()).T.squeeze()


def lifted_condensation_level(p, t, td, max_iters=50, eps=1.e-2):
    r"""
    持ち上げ凝結高度（LCL）を求める．

    Parameters
    ----------
    p : array_like
        持ち上げる気塊の気圧
    t : array_like
        持ち上げる気塊の気温
    td : array_like
        持ち上げる気塊の露点温度
    max_iters : int, optional
        計算する反復回数の最大値．デフォルトは50.
    eps : float, optional
        計算を打ち切る誤差．デフォルトは0.01 [Pa].

    Returns
    -------
    array_like
        持ち上げ凝結高度 [Pa]

    Notes
    -----
    持ち上げる気塊 `(p0, T0)` のLCLは `(p, Td0)` を通る等飽和混合比線と `(p, T)` を通る乾燥断熱線の交点として
    求めることができる．すなわちLCLを求めることは次の連立非線形方程式を解くことに等しい．

    .. math:: ws(p_0, Td_0) = ws(p, T) ~~~~(a)

    .. math:: \theta(p_0, T_0) = \theta(p, T) ~~~~(b)

    これは次のようにして数値的反復法により解くことができる．

    1. (a)を満たす `(p_n, T_n) = (p_0, Td_0)` を初期値とする．
    2. `T_n` と(b)式から(b)を満たす `(p_n+1, T_n)` を求める．
    3. `p_n+1` と(a)式から(a)を満たす `(p_n+1, T_n+1)` を求める．
    4. これを繰り返して `(p_n, T_n)` を更新していくと値は(a),(b)の解 `(p_lcl, T_lcl)` に収束する．
    """
    max_iters = 100
    # 初期値
    p_n = p
    t_n = td
    # (p, td)を通る等飽和混合比線
    ws = mixing_ratio(saturation_vapor_pressure(td), p)
    while max_iters:
        # p_nを更新
        p_new = p * (t_n / t) ** (1. / kappa)
        # t_nを更新
        t_new = dewpoint_from_e(vapor_pressure(p_new, ws))
        # 解が収束したらループをbreak
        if np.abs(p_new - p_n).max() < eps:
            break
        # 収束しない場合は繰り返す
        max_iters -= 1
        p_n = p_new
        t_n = t_new
    return p_new


def wetbulb(p, t, td):
    """
    湿球温度を求める．

    Parameters
    ----------
    p : array_like
        気圧 [Pa]
    t : array_like
        気温 [K]
    td : array_like
        露点温度 [K]

    Returns
    -------
    array_like
        湿球温度 [K]

    Notes
    -----
    与えられた気塊についてLCLを求め，LCLを通る湿潤断熱線が気塊の気圧 `p` と交わる点の気温を湿球温度として求める．
    """
    p_lcl = lifted_condensation_level(p, t, td)
    # T_lclは (p,T) からの温位の保存から求める
    t_lcl = t * (p_lcl / p) ** kappa

    return np.atleast_1d([moist_lapse(np.asarray([p_lcl, p]), t_lcl)[-1]
                          for p, p_lcl, t_lcl in zip(p, p_lcl, t_lcl)]).squeeze()
