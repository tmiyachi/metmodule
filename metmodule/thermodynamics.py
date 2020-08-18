"""
熱力学関連計算モジュール

.. autosummary::

    potential_temperature
    vapor_pressure
    saturation_vapor_pressure
    mixing_ratio
    dewpoint
    wetbulb
    dry_lapse
    moist_lapse
"""
import numpy as np
from scipy.integrate import odeint

from constants import kappa, epsilon, water_es_0cm air_Rd, water_Lv, air_Cp_d


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

    .. math:: e_s = e_0 \exp\left[ \frac{17.67 (T - 273.15) }{T - 29.65} \right] ~~ (T \le 0)

    .. math:: e_s = e_0 \exp\left[ \frac{22.46 (T - 273.15) }{T - 0.53} \right] ~~ (T < 0)

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
    計算式:

    .. math:: r = \frac{\epsilon e}{p - e}
    """

    return epsilon * e / (p - e)


def dewpoint(t, rh):
    """
    露点温度を計算する．

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
    """
    return dewpoint(rh * saturation_vapor_pressure(t) / 100.)


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

    .. Bakhshaii, A. and R. Stull, 2013: Saturated Pseudoadiabats--A
       Noniterative Approximation. J. Appl. Meteor. Clim., 52, 5-15.
    """
    def dT_dP(T, P):
        rs = mixing_ratio(saturation_vapor_pressure(T), P)
        # b  = 1. - 0.24 * rs
        b = 1
        return b / P * (air_Rd * T + water_Lv * rs) / (air_Cp_d +
                                                       (water_Lv * water_Lv * rs * epsilon * b) / air_Rd / T / T)

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
    1. Find the dew point from the LCL pressure and starting mixing ratio


    一方，乾燥断熱過程での保存則から `(T_0, p_0)` と `(T_lcl, p_lcl)` は以下の関係を満たすことを利用して

    .. math:: T_lcl = T \left(\frac{p_lcl}{p}\right)^\kappa

    .. math:: p_lcl = p_0 * \left\frac{T_lcl}{T}\right^{1/\kappa}

    から求まる．


    3. Iterate until convergence
    """
    max_iters = 100
    # (p, td)を通る飽和混合比線 -> LCLにおける（飽和）混合比
    ws = mixing_ratio(saturation_vapor_pressure(td), p)
    p_lcl = p
    while max_iters:
        # LCLでの温度=露点温度
        t_lcl = dewpoint(vapor_pressure(p_lcl, ws))
        new_p = p * (t_lcl / t) ** (1. / kappa)
        if np.abs(new_p - p_lcl).max() < eps:
            break
        p_i = new_p
        max_iters -= 1
    return new_p


def wetbulb(pressure, temperature, dewpt):
    """
    湿球温度を求める．

    Parameters
    ----------
    pressure : array_like
        air pressure [Pa]
    temperature : array_like
        air temperature [K]
    dewpt : array_like
        dew point temperature [K]

    Returns
    -------
    array_like
        wet-bulb temperature [K]

    Notes
    -----
    This function is implemented using an iterative approach.
    The basic algorithm is:
    1. Calculate LCL level from a given pressure, temperature and dewpoint using iterative approach.
    2. Calculate LCL temperature.
    3. Calculate the temperature at a level assuming liquid saturation processes operating from the LCL point.
    """
    p_lcl = lifted_condensation_level(p, t, td)
    t_lcl = t * (p_lcl / p) ** kappa

    return np.atleast_1d([moist_lapse(np.asarray([p_lcl, p]), t_lcl)[-1]
                          for p, p_lcl, t_lcl in zip(p, p_lcl, t_lcl)]).squeeze()
