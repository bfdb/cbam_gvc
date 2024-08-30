# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:15:07 2023

@author: bertr
"""

import cfg
import numpy as np
import pandas as pd
import utils as ut

# Parse EXIOBASE.
d_eb = ut.parse_eb_proc_pkl()

# Get EXIOBASE elements.
df_a = d_eb["a"]
df_y = d_eb["y"]
df_z = d_eb["z"]
df_l = d_eb["li"]
df_x = d_eb["x"].squeeze()

df_sat_f_z = d_eb[cfg.s_sat_f_z]
df_sat_s_z = d_eb[cfg.s_sat_s_z]
df_sat_f_y = d_eb[cfg.s_sat_f_y]
df_sat_s_y = d_eb[cfg.s_sat_s_y]

df_ghg = pd.read_csv(cfg.eb_ghg_file_path, sep="\t")
l_ghg = df_ghg.squeeze().to_list()

df_f_z = df_sat_f_z.loc[l_ghg]
df_s_z = df_sat_s_z.loc[l_ghg]
df_f_y = df_sat_f_y.loc[l_ghg]
df_s_y = df_sat_s_y.loc[l_ghg]

# Parse characterization factors.
df_q_gwp100 = pd.read_csv(cfg.q_gwp100_co2_file_path, sep="\t", index_col=0, header=0)
df_q_gwp100_d = pd.DataFrame(
    np.diag(df_q_gwp100.squeeze()),
    index=df_q_gwp100.index,
    columns=df_q_gwp100.index,
)

# Calculate characterized GHG emissions.
df_qf_z = df_q_gwp100_d.dot(df_f_z)
df_qf_y = df_q_gwp100_d.dot(df_f_y)

# Calculate characterized GHG emission intensities.
df_qs_z = df_q_gwp100_d.dot(df_s_z)
df_qs_y = df_q_gwp100_d.dot(df_s_y)

# Parse industry aggregations.
df_eb_ind_agg = pd.read_csv(cfg.eb_ind_agg_file_path, sep="\t", index_col=0)
df_eb_ind_agg_n_acefis = df_eb_ind_agg["nonACEFIS"]
df_eb_ind_agg_acefis = df_eb_ind_agg["ACEFIS"]
l_i_non_acefis = list(df_eb_ind_agg_n_acefis[df_eb_ind_agg_n_acefis > 0].index)
l_i_acefis = list(df_eb_ind_agg_acefis[df_eb_ind_agg_acefis > 0].index)

# Parse GHG aggregations.
df_ghg_agg = pd.read_csv(cfg.ghg_agg_file_path, sep="\t", index_col=0)

# Parse country aggregations.
# Liechtenstein and Iceland part of RoE
df_eb_cntr_agg, d_eb_cntr_agg = ut.parse_eb_cntr_agg()
l_c_all = d_eb_cntr_agg["ALL"]
l_c_eu27 = d_eb_cntr_agg["EU27"]
l_c_eu27_nor_che = d_eb_cntr_agg["EU27_NOR_CHE"]
l_c_non_eu27_nor_che = d_eb_cntr_agg["non_EU27_NOR_CHE"]

l_c_all_acefis = []
for cntr in l_c_all:
    for ind in l_i_acefis:
        t_cntr_ind = (cntr, ind)
        l_c_all_acefis.append(t_cntr_ind)

l_c_eu27_nor_che_i_non_acefis = []
for cntr in l_c_eu27_nor_che:
    for ind in l_i_non_acefis:
        t_cntr_ind = (cntr, ind)
        l_c_eu27_nor_che_i_non_acefis.append(t_cntr_ind)

l_c_eu27_nor_che_i_acefis = []
for cntr in l_c_eu27_nor_che:
    for ind in l_i_acefis:
        t_cntr_ind = (cntr, ind)
        l_c_eu27_nor_che_i_acefis.append(t_cntr_ind)

l_c_non_eu27_nor_che_i_non_acefis = []
for cntr in l_c_non_eu27_nor_che:
    for ind in l_i_non_acefis:
        t_cntr_ind = (cntr, ind)
        l_c_non_eu27_nor_che_i_non_acefis.append(t_cntr_ind)

l_c_non_eu27_nor_che_i_acefis = []
for cntr in l_c_non_eu27_nor_che:
    for ind in l_i_acefis:
        t_cntr_ind = (cntr, ind)
        l_c_non_eu27_nor_che_i_acefis.append(t_cntr_ind)

# Parse PPP.
df_ppp = pd.read_csv(cfg.ppp_file_path, sep="\t", index_col=0)
df_ppp_eu27 = df_ppp["EU27"]


# Calculate aggregated characterized GHG emission intensities.
def calc_qs_z_s_d():
    df_qf_z_s = df_qf_z.sum(axis=0)
    df_qs_z_s = df_qf_z_s / df_x
    df_qs_z_s = df_qs_z_s.fillna(0)
    df_qs_z_s_d = pd.DataFrame(
        np.diag(df_qs_z_s), index=df_qs_z_s.index, columns=df_qs_z_s.index
    )
    return df_qs_z_s_d


df_qs_z_s_d = calc_qs_z_s_d()


def sel_reg2reg(df, s_reg2reg, sel_col, sel_row):
    df_reg2reg = df.copy()
    df_reg2reg = df_reg2reg[sel_col]
    df_reg2reg = df_reg2reg.loc[sel_row]
    df_reg2reg = df_reg2reg.sum(axis=1)
    df_reg2reg_agg = df_reg2reg.groupby(level=1).sum()
    df_reg2reg_agg = pd.concat({s_reg2reg: df_reg2reg_agg})
    df_reg2reg_agg = ut.agg_ind(pd.DataFrame(df_reg2reg_agg).T, df_eb_ind_agg).squeeze()
    return df_reg2reg, df_reg2reg_agg


def get_df_reg2reg(df, d_reg2reg_0):
    df_reg2reg = df.copy()
    df_reg2reg[d_reg2reg_0["col"]] = 0
    df_reg2reg.loc[d_reg2reg_0["idx"]] = 0
    return df_reg2reg


def calc_ghg_base(df, df_qs_z, d_reg2reg_0, df_eb_ind_agg, df_ghg_agg):
    df_reg2reg = df.copy()
    df_reg2reg[d_reg2reg_0["col"]] = 0
    df_reg2reg.loc[d_reg2reg_0["idx"]] = 0
    df_reg2reg_s = df_reg2reg.sum(axis=1)
    df_reg2reg_s_d = pd.DataFrame(
        np.diag(df_reg2reg_s),
        index=df_reg2reg_s.index,
        columns=df_reg2reg_s.index,
    )
    df_q_reg2reg_s_d = df_ghg_agg.dot(df_qs_z.dot(df_reg2reg_s_d))

    df_q_reg2reg_s_d_agg_ind = ut.agg_ind(df_q_reg2reg_s_d, df_eb_ind_agg)

    return df_q_reg2reg_s_d, df_q_reg2reg_s_d_agg_ind, df_reg2reg_s


d_reg2reg_0 = {}
reg2reg = "eu2eu"
d_reg2reg_0[reg2reg] = {}
d_reg2reg_0[reg2reg]["col"] = l_c_non_eu27_nor_che
d_reg2reg_0[reg2reg]["idx"] = l_c_non_eu27_nor_che

reg2reg = "row2row"
d_reg2reg_0[reg2reg] = {}
d_reg2reg_0[reg2reg]["col"] = l_c_eu27_nor_che
d_reg2reg_0[reg2reg]["idx"] = l_c_eu27_nor_che

reg2reg = "row2eu"
d_reg2reg_0[reg2reg] = {}
d_reg2reg_0[reg2reg]["col"] = l_c_non_eu27_nor_che
d_reg2reg_0[reg2reg]["idx"] = l_c_eu27_nor_che

reg2reg = "eu2row"
d_reg2reg_0[reg2reg] = {}
d_reg2reg_0[reg2reg]["col"] = l_c_eu27_nor_che
d_reg2reg_0[reg2reg]["idx"] = l_c_non_eu27_nor_che

kg2gt = 1e12
kg2mt = 1e9
kg2kt = 1e6

explore_global = True
if explore_global:
    """Total global CO2."""
    df_qf_z_s = df_qf_z.sum(axis=0) / kg2gt
    df_qf_y_s = df_qf_y.sum(axis=1) / kg2gt
    df_qf_x_s = df_qf_z_s.sum() + df_qf_y_s.sum()

    """ Global intermediate CO2."""
    df_qf_z_ind_agg = df_qf_z.sum(axis=0).unstack().dot(df_eb_ind_agg) / kg2gt
    df_qf_z_ind_agg_eu27_nor_che = df_qf_z_ind_agg.loc[l_c_eu27_nor_che]
    df_qf_z_ind_agg_eu27_nor_che_s = (
        df_qf_z_ind_agg_eu27_nor_che.sum(axis=0) * kg2gt / kg2mt
    )

    """ Total EU import CO2"""
    df_z_row2eu = df_z.loc[l_c_non_eu27_nor_che, l_c_eu27_nor_che]

    """ explore global international trade of ACEFIS."""
    # aggregated emission intensity
    df_qfi_z = df_qf_z_s / df_x
    df_qfi_z = df_qfi_z.fillna(0)
    df_qfi_z_d = pd.DataFrame(
        np.diag(df_qfi_z), index=df_qfi_z.index, columns=df_qfi_z.index
    )
    df_qz = df_qfi_z_d.dot(df_z)
    df_qz = df_qz.fillna(0)
    l_qz_ind_agg = []
    for cntr_orig in l_c_all:
        l_qz_row = []
        for cntr_dest in l_c_all:
            df_ghg_cntr_o2d = df_qz.loc[cntr_orig, cntr_dest]
            df_ghg_cntr_o2d_ind_agg = df_eb_ind_agg.T.dot(
                df_ghg_cntr_o2d.dot(df_eb_ind_agg)
            )
            df_ghg_cntr_o2d_ind_agg = pd.concat({cntr_orig: df_ghg_cntr_o2d_ind_agg})
            df_ghg_cntr_o2d_ind_agg = pd.concat(
                {cntr_dest: df_ghg_cntr_o2d_ind_agg}, axis=1
            )
            l_qz_row.append(df_ghg_cntr_o2d_ind_agg)
        df_qz_rw = pd.concat(l_qz_row, axis=1)
        l_qz_ind_agg.append(df_qz_rw)
    df_qz_ind_agg = pd.concat(l_qz_ind_agg)

    df_qz_ind_agg_agg = (
        df_qz_ind_agg.groupby(level=1, axis=0).sum().groupby(level=1, axis=1).sum()
    )

    l_c_all_acefis = []
    l_c_all_acefis_n_acefis = []
    for cntr in l_c_all:
        l_c_all_acefis.append((cntr, "ACEFIS"))
        l_c_all_acefis_n_acefis.append((cntr, "ACEFIS"))
        l_c_all_acefis_n_acefis.append((cntr, "nonACEFIS"))

    df_qz_ind_agg_acefis = df_qz_ind_agg.loc[l_c_all_acefis]
    df_qz_ind_agg_acefis = df_qz_ind_agg_acefis[l_c_all_acefis_n_acefis]
    df_qz_ind_agg_acefis = df_qz_ind_agg_acefis.groupby(axis=1, level=0).sum()

    df_qz_ind_agg_acefis_dom = df_qz_ind_agg_acefis.copy() * 0
    for cntr in l_c_all:
        df_qz_ind_agg_acefis_dom.loc[cntr, cntr] = df_qz_ind_agg_acefis
    df_qz_ind_agg_acefis_dom_s = df_qz_ind_agg_acefis_dom.sum(axis=1)

    df_qz_ind_agg_acefis_trade = df_qz_ind_agg_acefis.copy()
    for cntr in l_c_all:
        df_qz_ind_agg_acefis_trade.loc[cntr, cntr] = 0
    df_qz_ind_agg_acefis_export_s = df_qz_ind_agg_acefis_trade.sum(axis=1)
    df_qz_ind_agg_acefis_import_s = df_qz_ind_agg_acefis_trade.sum(axis=0)

    df_qy = df_qfi_z_d.dot(df_y)
    df_qy = df_qy.groupby(axis=1, level=0).sum()
    l_qy_ind_agg = []
    for cntr_orig in l_c_all:
        df_qy_row = df_qy.loc[cntr_orig]
        df_qy_row_ind_agg = df_eb_ind_agg.T.dot(df_qy_row)
        df_qy_row_ind_agg = pd.concat({cntr_orig: df_qy_row_ind_agg})

        l_qy_ind_agg.append(df_qy_row_ind_agg)
    df_qy_ind_agg = pd.concat(l_qy_ind_agg)
    df_qy_ind_agg_acefis = df_qy_ind_agg.loc[l_c_all_acefis]

    df_qy_ind_agg_acefis_dom = df_qy_ind_agg_acefis.copy() * 0
    for cntr in l_c_all:
        df_qy_ind_agg_acefis_dom.loc[cntr, cntr] = df_qy_ind_agg_acefis
    df_qy_ind_agg_acefis_dom_s = df_qy_ind_agg_acefis_dom.sum(axis=1)

    df_qy_ind_agg_acefis_trade = df_qy_ind_agg_acefis.copy()
    for cntr in l_c_all:
        df_qy_ind_agg_acefis_trade.loc[cntr, cntr] = 0
    df_qy_ind_agg_acefis_export_s = df_qy_ind_agg_acefis_trade.sum(axis=1)
    df_qy_ind_agg_acefis_import_s = df_qy_ind_agg_acefis_trade.sum(axis=0)

    df_qx_ind_agg_acefis_dom_s = df_qz_ind_agg_acefis_dom_s + df_qy_ind_agg_acefis_dom_s
    df_qx_ind_agg_acefis_export_s = (
        df_qz_ind_agg_acefis_export_s + df_qy_ind_agg_acefis_export_s
    )
    df_qx_ind_agg_acefis_import_s = (
        df_qz_ind_agg_acefis_import_s + df_qy_ind_agg_acefis_import_s
    )

    df_qx_ind_agg_acefis_trade = df_qz_ind_agg_acefis_trade + df_qy_ind_agg_acefis_trade
    df_qx_ind_agg_acefis_trade_u = df_qx_ind_agg_acefis_trade.stack()
    df_qx_ind_agg_acefis_trade_s = df_qx_ind_agg_acefis_trade_u.sum()

    df_qx_ind_agg = df_qz_ind_agg.sum(axis=1) + df_qy_ind_agg.sum(axis=1)
    df_qx_ind_agg_ind = df_qx_ind_agg.groupby(level=1).sum()

    df_qz_ind_agg_acefis_n_acefis = df_qz_ind_agg[l_c_all_acefis_n_acefis]

    df_qx_ind_agg = (
        df_qz_ind_agg_acefis_n_acefis.groupby(axis=1, level=0).sum()
        + df_qy_ind_agg.groupby(axis=1, level=0).sum()
    )

    df_qx_ind_agg_s = df_qx_ind_agg.sum(axis=1)
    df_qx_ind_agg_u = df_qx_ind_agg_s.unstack()
    df_qx_ind_agg_us = df_qx_ind_agg_u.sum(axis=0)

    df_qx_ind_agg_dom = df_qx_ind_agg.copy() * 0
    df_qx_ind_agg_export = df_qx_ind_agg.copy()

    for cntr in l_c_all:
        df_qx_ind_agg_dom.loc[cntr, cntr] = df_qx_ind_agg.loc[cntr, cntr].values
        df_qx_ind_agg_export.loc[cntr, cntr] = 0

    df_qx_ind_agg_dom_s = df_qx_ind_agg_dom.sum(axis=1).groupby(level=1).sum()
    df_qx_ind_agg_export_s = df_qx_ind_agg_export.sum(axis=1).groupby(level=1).sum()

    df_qx_ind_agg_acefis = df_qz_ind_agg_acefis + df_qy_ind_agg_acefis
    df_qx_ind_agg_acefis_row2eu = df_qx_ind_agg_acefis[l_c_eu27_nor_che]
    df_qx_ind_agg_acefis_row2eu = df_qx_ind_agg_acefis_row2eu.loc[l_c_non_eu27_nor_che]
    df_qx_ind_agg_acefis_row2eu_s = df_qx_ind_agg_acefis_row2eu.sum().sum()

    df_qx_ind_agg_acefis_eu2eu = df_qx_ind_agg_acefis[l_c_eu27_nor_che]
    df_qx_ind_agg_acefis_eu2eu = df_qx_ind_agg_acefis_eu2eu.loc[l_c_eu27_nor_che]
    df_qx_ind_agg_acefis_eu2eu_s = df_qx_ind_agg_acefis_eu2eu.sum().sum()

    df_qx_ind_agg_acefis_eu2eu_dom = df_qx_ind_agg_acefis_eu2eu.copy() * 0
    for cntr in l_c_eu27_nor_che:
        df_qx_ind_agg_acefis_eu2eu_dom.loc[cntr, cntr] = df_qx_ind_agg_acefis_eu2eu.loc[
            cntr, cntr
        ].values[0]
    df_qx_ind_agg_acefis_eu2eu_dom_s = df_qx_ind_agg_acefis_eu2eu_dom.sum().sum()

    df_qx_ind_agg_acefis_eu2eu_trade = df_qx_ind_agg_acefis_eu2eu.copy()
    for cntr in l_c_eu27_nor_che:
        df_qx_ind_agg_acefis_eu2eu_trade.loc[cntr, cntr] = 0
    df_qx_ind_agg_acefis_eu2eu_trade_s = df_qx_ind_agg_acefis_eu2eu_trade.sum().sum()

    df_qx_ind_agg_acefis_eu2row = df_qx_ind_agg_acefis[l_c_non_eu27_nor_che]
    df_qx_ind_agg_acefis_eu2row = df_qx_ind_agg_acefis_eu2row.loc[l_c_eu27_nor_che]
    df_qx_ind_agg_acefis_eu2row_s = df_qx_ind_agg_acefis_eu2row.sum().sum()

    df_qx_ind_agg_acefis_row2row = df_qx_ind_agg_acefis[l_c_non_eu27_nor_che]
    df_qx_ind_agg_acefis_row2row = df_qx_ind_agg_acefis_row2row.loc[
        l_c_non_eu27_nor_che
    ]
    df_qx_ind_agg_acefis_row2row_s = df_qx_ind_agg_acefis_row2row.sum().sum()

    df_qx_ind_agg_acefis_u = df_qx_ind_agg_acefis.stack()


def calc_qs_z_lte_eu27_avg_all():
    # Get EU27 GHG emissions per industry
    df_f_z_eu27 = df_f_z[l_c_eu27]
    df_f_z_eu27_ind = df_f_z_eu27.groupby(level=1, axis=1).sum()

    # Get EU27 total output per industry
    df_x_eu27 = df_x[l_c_eu27]
    df_x_eu27_ind = df_x_eu27.groupby(level=1).sum()

    # Calculate EU average GHG intensities
    df_s_z_eu27_ind = df_f_z_eu27_ind / df_x_eu27_ind
    df_s_z_eu27_ind.fillna(0, inplace=True)
    df_s_z_eu27_ind.replace(np.inf, 0, inplace=True)
    df_s_z_eu27_ind.replace(-1 * np.inf, 0, inplace=True)

    # If higher, replace global GHG intensities for all sectors by EU27 averages
    l_i_all = l_i_acefis + l_i_non_acefis
    df_s_z_lte_eu27_avg_all = df_s_z.copy()
    for ghg in l_ghg:
        for cntr in l_c_non_eu27_nor_che:
            for i_all in l_i_all:
                val_base = df_s_z.loc[ghg, (cntr, i_all)]
                val_eu27 = df_s_z_eu27_ind.loc[ghg, i_all]
                val_eu27_ppp = val_eu27 / df_ppp_eu27[cntr]
                if val_base > val_eu27_ppp:
                    df_s_z_lte_eu27_avg_all.loc[ghg, (cntr, i_all)] = val_eu27_ppp

    df_qs_z_lte_eu27_avg_all = df_q_gwp100_d.dot(df_s_z_lte_eu27_avg_all)

    return df_qs_z_lte_eu27_avg_all, df_s_z_eu27_ind


def calc_qs_z_ind_agg(l_c_reg):
    # Get reg GHG emissions per industry
    df_f_z_reg = df_f_z[l_c_reg]
    df_f_z_reg_ind = df_f_z_reg.groupby(level=1, axis=1).sum()
    df_f_z_reg_ind_agg = df_f_z_reg_ind.dot(df_eb_ind_agg)

    # Get reg total output per industry
    df_x_reg = df_x[l_c_reg]
    df_x_reg_ind = df_x_reg.groupby(level=1).sum()
    df_x_reg_ind_agg = df_x_reg_ind.dot(df_eb_ind_agg)

    # Calculate EU average GHG intensities
    df_s_z_reg_ind = df_f_z_reg_ind / df_x_reg_ind
    df_s_z_reg_ind.fillna(0, inplace=True)
    df_s_z_reg_ind.replace(np.inf, 0, inplace=True)
    df_s_z_reg_ind.replace(-1 * np.inf, 0, inplace=True)

    # Calculate EU average GHG intensities per industry aggregates
    df_s_z_reg_ind_agg = df_f_z_reg_ind_agg / df_x_reg_ind_agg
    df_s_z_reg_ind_agg.fillna(0, inplace=True)
    df_s_z_reg_ind_agg.replace(np.inf, 0, inplace=True)
    df_s_z_reg_ind_agg.replace(-1 * np.inf, 0, inplace=True)
    df_qs_z_reg_ind_agg = df_q_gwp100.T.dot(df_s_z_reg_ind_agg).squeeze()

    return df_qs_z_reg_ind_agg, df_x_reg_ind_agg


explore_ghgi = True
if explore_ghgi:
    df_qs_z_eu27_ind_agg, df_x_eu27_ind_agg = calc_qs_z_ind_agg(l_c_eu27)
    df_qs_z_eu27_ind_agg = df_qs_z_eu27_ind_agg / kg2kt
    df_qs_zx_eu27_ind_agg = pd.concat([df_qs_z_eu27_ind_agg, df_x_eu27_ind_agg], axis=1)
    df_qs_z_non_eu27_nor_che_ind_agg, df_x_non_eu27_nor_che_ind_agg = calc_qs_z_ind_agg(
        l_c_non_eu27_nor_che
    )
    df_qs_z_non_eu27_nor_che_ind_agg = df_qs_z_non_eu27_nor_che_ind_agg / kg2kt

    df_qs_zx_non_eu27_nor_che_ind_agg = pd.concat(
        [df_qs_z_non_eu27_nor_che_ind_agg, df_x_non_eu27_nor_che_ind_agg], axis=1
    )

    l_c_reg = l_c_eu27

    # Get reg GHG emissions per industry
    df_f_z_reg = df_f_z[l_c_reg]
    df_qf_z_reg = df_q_gwp100.T.dot(df_f_z_reg).squeeze()
    df_qf_z_reg_u = df_qf_z_reg.unstack()
    df_qf_z_reg_u_ind_agg = df_qf_z_reg_u.dot(df_eb_ind_agg)

    # Get reg total output per industry
    df_x_reg = df_x[l_c_reg]
    df_x_reg_u = df_x_reg.unstack()
    df_x_reg_u_ind_agg = df_x_reg_u.dot(df_eb_ind_agg)

    df_qs_z_reg_u_ind_agg = df_qf_z_reg_u_ind_agg / df_x_reg_u_ind_agg / 1e3 / 1e3

    l_c_reg = l_c_non_eu27_nor_che

    # Get reg GHG emissions per industry
    df_f_z_reg = df_f_z[l_c_reg]
    df_qf_z_reg = df_q_gwp100.T.dot(df_f_z_reg).squeeze()
    df_qf_z_reg_u = df_qf_z_reg.unstack()
    df_qf_z_reg_u_ind_agg = df_qf_z_reg_u.dot(df_eb_ind_agg)

    # Get reg total output per industry
    df_x_reg = df_x[l_c_reg]
    df_x_reg_u = df_x_reg.unstack()
    df_x_reg_u_ind_agg = df_x_reg_u.dot(df_eb_ind_agg)

    df_qs_z_reg_u_ind_agg = df_qf_z_reg_u_ind_agg / df_x_reg_u_ind_agg / 1e3 / 1e3

"""
Baseline: Direct
"""


def calc_base_direct():
    df_qz_base_all_row2eu, df_qz_base_all_row2eu_ind_agg, df_z_row2eu = calc_ghg_base(
        df_z,
        df_qs_z,
        d_reg2reg_0["row2eu"],
        df_eb_ind_agg,
        df_ghg_agg,
    )

    df_qy_base_all_row2eu, df_qy_base_all_row2eu_ind_agg, df_y_row2eu = calc_ghg_base(
        df_y,
        df_qs_z,
        d_reg2reg_0["row2eu"],
        df_eb_ind_agg,
        df_ghg_agg,
    )

    df_x_row2eu = df_z_row2eu + df_y_row2eu

    df_qx_base_all_row2eu = df_qz_base_all_row2eu + df_qy_base_all_row2eu
    df_qx_base_all_row2eu = df_qx_base_all_row2eu / kg2kt

    df_qx_base_all_row2eu_s = df_qx_base_all_row2eu.sum(axis=0)
    df_qx_base_all_row2eu_u = df_qx_base_all_row2eu_s.unstack()
    df_qx_base_all_row2eu_u = df_qx_base_all_row2eu_u[df_eb_ind_agg.index]

    df_qx_base_all_row2eu_u_ind_agg = df_qx_base_all_row2eu_u.dot(df_eb_ind_agg)

    return df_qx_base_all_row2eu_u, df_qx_base_all_row2eu_u_ind_agg, df_x_row2eu


(
    df_qx_base_all_row2eu_u,
    df_qx_base_all_row2eu_u_ind_agg,
    df_x_row2eu,
) = calc_base_direct()
df_qx_base_all_row2eu_ind_agg = (
    df_qx_base_all_row2eu_u_ind_agg.sum(axis=0) * kg2kt / kg2mt
)

explore_x = True
if explore_x:
    Mt2kt = 1e3
    df_qx_base_all_row2eu_u_ind_agg_non_eu27_nor_che = (
        df_qx_base_all_row2eu_u_ind_agg.loc[l_c_non_eu27_nor_che]
    )
    df_qx_base_all_row2eu_ind_agg_non_eu27_nor_che = (
        df_qx_base_all_row2eu_u_ind_agg_non_eu27_nor_che.stack()
    )
    df_qx_base_all_row2eu_ind_agg_non_eu27_nor_che = (
        df_qx_base_all_row2eu_ind_agg_non_eu27_nor_che * Mt2kt
    )

"""
Baseline: Embodied
"""


def calc_base_supply():
    # Calculate embodied flows of all sectors driven by EU2EU final demand.
    df_y_eu2eu = get_df_reg2reg(df_y, d_reg2reg_0["eu2eu"])
    df_ly_all_eu2eu = df_l.dot(df_y_eu2eu)
    df_ly_all_eu2eu.loc[l_c_eu27_nor_che] = 0
    df_ly_all_eu2eu = df_ly_all_eu2eu.sum(axis=1)
    df_qsly_base_all_eu2eu = df_ghg_agg.dot(df_qs_z.dot(ut.df2df_d(df_ly_all_eu2eu)))

    # Calculate embodied flows of all sectors driven by ROW2EU final demand.
    df_y_row2eu = get_df_reg2reg(df_y, d_reg2reg_0["row2eu"])
    df_ly_all_row2eu = df_l.dot(df_y_row2eu)
    df_ly_all_row2eu.loc[l_c_eu27_nor_che] = 0
    df_ly_all_row2eu = df_ly_all_row2eu.sum(axis=1)
    df_qsly_base_all_row2eu = df_ghg_agg.dot(df_qs_z.dot(ut.df2df_d(df_ly_all_row2eu)))

    # Calculate embodied flows of all sectors driven by EU2ROW final demand.
    df_y_eu2row = get_df_reg2reg(df_y, d_reg2reg_0["eu2row"])
    df_ly_all_eu2row = df_l.dot(df_y_eu2row)
    df_ly_all_eu2row.loc[l_c_eu27_nor_che] = 0
    df_ly_all_eu2row = df_ly_all_eu2row.sum(axis=1)
    df_qsly_base_all_eu2row = df_ghg_agg.dot(df_qs_z.dot(ut.df2df_d(df_ly_all_eu2row)))

    # Calculate embodied flows of all sectors driven by ROW2ROW final demand.
    df_y_row2row = get_df_reg2reg(df_y, d_reg2reg_0["row2row"])
    df_ay_row2row = df_a.dot(df_y_row2row)
    df_ay_row2row.loc[l_c_non_eu27_nor_che] = 0
    df_lay_all_row2row = df_l.dot(df_ay_row2row)
    df_lay_all_row2row.loc[l_c_eu27_nor_che] = 0
    df_lay_all_row2row = df_lay_all_row2row.sum(axis=1)
    df_qslay_base_all_row2row = df_ghg_agg.dot(
        df_qs_z.dot(ut.df2df_d(df_lay_all_row2row))
    )

    df_qsly_base_all = (
        df_qsly_base_all_eu2eu
        + df_qsly_base_all_row2eu
        + df_qsly_base_all_eu2row
        + df_qslay_base_all_row2row
    )
    # df_qsly_base_all = df_qsly_base_all / kg2mt
    df_qsly_base_all = df_qsly_base_all / kg2kt

    df_qsly_base_all_s = df_qsly_base_all.sum(axis=0)
    df_qsly_base_all_u = df_qsly_base_all_s.unstack()
    df_qsly_base_all_u = df_qsly_base_all_u[df_eb_ind_agg.index]

    df_qsly_base_all_u_ind_agg = df_qsly_base_all_u.dot(df_eb_ind_agg)

    d_x = {}
    d_x["eu2eu"] = df_ly_all_eu2eu
    d_x["row2eu"] = df_ly_all_row2eu
    d_x["eu2row"] = df_ly_all_eu2row
    d_x["row2row"] = df_lay_all_row2row

    return df_qsly_base_all_u, df_qsly_base_all_u_ind_agg, d_x


df_qsly_base_all_u, df_qsly_base_all_u_ind_agg, d_x = calc_base_supply()
df_qsly_base_all_ind_agg = df_qsly_base_all_u_ind_agg.sum(axis=0)

explore_ly = True
if explore_ly:
    df_ly = d_x["eu2eu"] * 0
    for reg2reg in d_x:
        df_ly += d_x[reg2reg]
    df_ly_u = df_ly.unstack().dot(df_eb_ind_agg)
    df_ly_u = df_ly_u.loc[l_c_non_eu27_nor_che]
    # df_ly_s = df_ly_u.sum(axis=0)
    df_ly_s = df_ly_u.stack()

    df_qsly_base_all_u_ind_agg_non_eu27_nor_che = df_qsly_base_all_u_ind_agg.loc[
        l_c_non_eu27_nor_che
    ]
    df_qsly_base_all_ind_agg_non_eu27_nor_che = (
        df_qsly_base_all_u_ind_agg_non_eu27_nor_che.stack()
    )
    df_qsly_base_all_ind_agg_non_eu27_nor_che = (
        df_qsly_base_all_ind_agg_non_eu27_nor_che * Mt2kt
    )
"""
Potential reductions
"""


df_qs_z_lte_eu27_avg_all, df_s_z_eu27_ind = calc_qs_z_lte_eu27_avg_all()
df_s_z_eu27_ind
"""
Potential reductions: Direct
"""


def calc_lte_direct():
    df_qz_lte_all_row2eu, df_qz_lte_all_row2eu_ind_agg, df_z_row2eu = calc_ghg_base(
        df_z,
        df_qs_z_lte_eu27_avg_all,
        d_reg2reg_0["row2eu"],
        df_eb_ind_agg,
        df_ghg_agg,
    )

    df_qy_lte_all_row2eu, df_qy_lte_all_row2eu_ind_agg, df_y_row2eu = calc_ghg_base(
        df_y,
        df_qs_z_lte_eu27_avg_all,
        d_reg2reg_0["row2eu"],
        df_eb_ind_agg,
        df_ghg_agg,
    )

    df_qx_lte_all_row2eu = df_qz_lte_all_row2eu + df_qy_lte_all_row2eu
    df_qx_lte_all_row2eu = df_qx_lte_all_row2eu / kg2kt

    df_qx_lte_all_row2eu_s = df_qx_lte_all_row2eu.sum(axis=0)
    df_qx_lte_all_row2eu_u = df_qx_lte_all_row2eu_s.unstack()
    df_qx_lte_all_row2eu_u = df_qx_lte_all_row2eu_u[df_eb_ind_agg.index]

    df_qx_lte_all_row2eu_u_ind_agg = df_qx_lte_all_row2eu_u.dot(df_eb_ind_agg)

    return df_qx_lte_all_row2eu_u, df_qx_lte_all_row2eu_u_ind_agg


df_qx_lte_all_row2eu_u, df_qx_lte_all_row2eu_u_ind_agg = calc_lte_direct()
df_qx_lte_all_row2eu_ind_agg = df_qx_lte_all_row2eu_u_ind_agg.sum(axis=0)

df_qx_lte_all_row2eu_ind_agg_delta = (
    df_qx_lte_all_row2eu_ind_agg - df_qx_base_all_row2eu_ind_agg
)

df_qx_lte_all_row2eu_u_ind_agg_delta = (
    df_qx_lte_all_row2eu_u_ind_agg - df_qx_base_all_row2eu_u_ind_agg
)

df_qx_lte_all_row2eu_u_delta = df_qx_lte_all_row2eu_u - df_qx_base_all_row2eu_u
df_qx_lte_all_row2eu_u_prio = df_qx_lte_all_row2eu_u_delta[l_i_non_acefis].copy()
df_qx_lte_all_row2eu_u_prio["ACEFIS"] = df_qx_lte_all_row2eu_u_delta[l_i_acefis].sum(
    axis=1
)
df_qx_lte_all_row2eu_u_prio = df_qx_lte_all_row2eu_u_prio.loc[l_c_non_eu27_nor_che]

df_qx_lte_all_row2eu_prio_s = df_qx_lte_all_row2eu_u_prio.sum(axis=0)

df_qx_lte_all_row2eu_prio = df_qx_lte_all_row2eu_u_prio.stack()

df_x_row2eu_us = df_x_row2eu.unstack().dot(df_eb_ind_agg).sum(axis=0)

explore_lte = True
if explore_lte:
    df_qx_lte_all_row2eu_u_ind_agg_non_eu27_nor_che = (
        df_qx_lte_all_row2eu_u_ind_agg.loc[l_c_non_eu27_nor_che]
    )
    df_qx_lte_all_row2eu_ind_agg_non_eu27_nor_che = (
        df_qx_lte_all_row2eu_u_ind_agg_non_eu27_nor_che.stack()
    )
    df_qx_lte_all_row2eu_ind_agg_non_eu27_nor_che = (
        df_qx_lte_all_row2eu_ind_agg_non_eu27_nor_che * Mt2kt
    )

    df_qx_lte_all_row2eu_u_ind_agg_delta_non_eu27_nor_che = (
        df_qx_lte_all_row2eu_u_ind_agg_delta.loc[l_c_non_eu27_nor_che]
    )


"""
Potential reductions and Prioritization: Embodied, ACEFIS, All sectors
"""


def calc_lte_supply():
    df_ly_all_eu2eu = d_x["eu2eu"]
    df_ly_all_row2eu = d_x["row2eu"]
    df_ly_all_eu2row = d_x["eu2row"]
    df_lay_all_row2row = d_x["row2row"]

    # Calculate embodied flows of all sectors driven by EU2EU final demand.
    df_qsly_lte_all_eu2eu = df_ghg_agg.dot(
        df_qs_z_lte_eu27_avg_all.dot(ut.df2df_d(df_ly_all_eu2eu))
    )

    # Calculate embodied flows of all sectors driven by ROW2EU final demand.
    df_qsly_lte_all_row2eu = df_ghg_agg.dot(
        df_qs_z_lte_eu27_avg_all.dot(ut.df2df_d(df_ly_all_row2eu))
    )

    # Calculate embodied flows of all sectors driven by EU2ROW final demand.
    df_qsly_lte_all_eu2row = df_ghg_agg.dot(
        df_qs_z_lte_eu27_avg_all.dot(ut.df2df_d(df_ly_all_eu2row))
    )

    # Calculate embodied flows of all sectors driven by ROW2ROW final demand.
    df_qslay_lte_all_row2row = df_ghg_agg.dot(
        df_qs_z_lte_eu27_avg_all.dot(ut.df2df_d(df_lay_all_row2row))
    )

    df_qsly_lte_all = (
        df_qsly_lte_all_eu2eu
        + df_qsly_lte_all_row2eu
        + df_qsly_lte_all_eu2row
        + df_qslay_lte_all_row2row
    )
    # df_qsly_lte_all = df_qsly_lte_all / kg2mt
    df_qsly_lte_all = df_qsly_lte_all / kg2kt

    df_qsly_lte_all_s = df_qsly_lte_all.sum(axis=0)
    df_qsly_lte_all_u = df_qsly_lte_all_s.unstack()
    df_qsly_lte_all_u = df_qsly_lte_all_u[df_eb_ind_agg.index]

    df_qsly_lte_all_u_ind_agg = df_qsly_lte_all_u.dot(df_eb_ind_agg)

    return df_qsly_lte_all_u, df_qsly_lte_all_u_ind_agg


df_qsly_lte_all_u, df_qsly_lte_all_u_ind_agg = calc_lte_supply()
df_qsly_lte_all_ind_agg = df_qsly_lte_all_u_ind_agg.sum(axis=0)
df_qsly_lte_all_ind_agg_delta = df_qsly_lte_all_ind_agg - df_qsly_base_all_ind_agg

df_qsly_lte_all_u_ind_agg_delta = df_qsly_lte_all_u_ind_agg - df_qsly_base_all_u_ind_agg

df_qsly_lte_all_u_delta = df_qsly_lte_all_u - df_qsly_base_all_u
df_qsly_lte_all_u_prio = df_qsly_lte_all_u_delta[l_i_non_acefis].copy()
df_qsly_lte_all_u_prio["ACEFIS"] = df_qsly_lte_all_u_delta[l_i_acefis].sum(axis=1)
df_qsly_lte_all_prio_s = df_qsly_lte_all_u_prio.sum(axis=0)
df_qsly_lte_all_prio = df_qsly_lte_all_u_prio.stack()
df_qsly_lte_all_u_prio_non_eu27_nor_che = df_qsly_lte_all_u_prio.loc[
    l_c_non_eu27_nor_che
]


explore_lte = True
if explore_lte:
    df_qsly_lte_all_u_ind_agg_non_eu27_nor_che = df_qsly_lte_all_u_ind_agg.loc[
        l_c_non_eu27_nor_che
    ]
    df_qsly_lte_all_ind_agg_non_eu27_nor_che = (
        df_qsly_lte_all_u_ind_agg_non_eu27_nor_che.stack()
    )
    df_qsly_lte_all_ind_agg_non_eu27_nor_che = (
        df_qsly_lte_all_ind_agg_non_eu27_nor_che * Mt2kt
    )

    df_qsly_lte_all_u_ind_agg_delta_non_eu27_nor_che = (
        df_qsly_lte_all_u_ind_agg_delta.loc[l_c_non_eu27_nor_che]
    )
