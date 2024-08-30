# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:07:54 2022

@author: bfdeboer
"""

import cfg
import numpy as np
import pandas as pd


def parse_eb_raw_txt():
    d_eb = {}
    print(cfg.s_a)
    d_eb[cfg.s_a] = pd.read_csv(
        cfg.eb_a_file_path, index_col=[0, 1], header=[0, 1, 2], sep="\t"
    )
    d_eb[cfg.s_a] = d_eb[cfg.s_a].droplevel(level=2, axis=1)

    print(cfg.s_x)
    d_eb[cfg.s_x] = pd.read_csv(
        cfg.eb_x_file_path, index_col=[0, 1], header=0, sep="\t"
    )

    print(cfg.s_z)
    d_eb[cfg.s_z] = pd.read_csv(
        cfg.eb_z_file_path, index_col=[0, 1], header=[0, 1, 2], sep="\t"
    )
    d_eb[cfg.s_z] = d_eb[cfg.s_z].droplevel(level=2, axis=1)

    print(cfg.s_y)
    d_eb[cfg.s_y] = pd.read_csv(
        cfg.eb_y_file_path, index_col=[0, 1], header=[0, 1, 2], sep="\t"
    )
    d_eb[cfg.s_y] = d_eb[cfg.s_y].droplevel(level=2, axis=1)

    # imp_f_z
    var_name = cfg.s_imp_f_z
    var_path = cfg.eb_imp_f_z_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    # imp_s_z
    var_name = cfg.s_imp_s_z
    var_path = cfg.eb_imp_s_z_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    # imp_f_y
    var_name = cfg.s_imp_f_y
    var_path = cfg.eb_imp_f_z_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    # imp_s_y
    var_name = cfg.s_imp_s_y
    var_path = cfg.eb_imp_s_z_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    # sat_f_z
    var_name = cfg.s_sat_f_z
    var_path = cfg.eb_sat_f_z_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    # sat_s_z
    var_name = cfg.s_sat_s_z
    var_path = cfg.eb_sat_s_z_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    # sat_f_y
    var_name = cfg.s_sat_f_y
    var_path = cfg.eb_sat_f_y_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    # sat_s_y
    var_name = cfg.s_sat_s_y
    var_path = cfg.eb_sat_s_y_file_path
    var_id = 0
    var_col = [0, 1, 2]
    print(var_name)
    d_eb[var_name] = pd.read_csv(var_path, index_col=var_id, header=var_col, sep="\t")
    d_eb[var_name] = d_eb[var_name].droplevel(level=2, axis=1)

    return d_eb


def dump_eb_raw_pkl(d_eb):
    d_eb[cfg.s_a].to_pickle(cfg.eb_a_pkl_file_path)
    d_eb[cfg.s_z].to_pickle(cfg.eb_z_pkl_file_path)

    d_eb[cfg.s_y].to_pickle(cfg.eb_y_pkl_file_path)
    d_eb[cfg.s_imp_f_z].to_pickle(cfg.eb_imp_f_z_pkl_file_path)
    d_eb[cfg.s_imp_s_z].to_pickle(cfg.eb_imp_s_z_pkl_file_path)
    d_eb[cfg.s_imp_f_y].to_pickle(cfg.eb_imp_f_y_pkl_file_path)
    d_eb[cfg.s_imp_s_y].to_pickle(cfg.eb_imp_s_y_pkl_file_path)

    d_eb[cfg.s_sat_f_z].to_pickle(cfg.eb_sat_f_z_pkl_file_path)
    d_eb[cfg.s_sat_s_z].to_pickle(cfg.eb_sat_s_z_pkl_file_path)
    d_eb[cfg.s_sat_f_y].to_pickle(cfg.eb_sat_f_y_pkl_file_path)
    d_eb[cfg.s_sat_s_y].to_pickle(cfg.eb_sat_s_y_pkl_file_path)


def calc_li(d_eb):
    df_a = d_eb[cfg.s_a]
    ar_i = np.eye(len(df_a))
    df_l = ar_i - df_a
    df_li = pd.DataFrame(np.linalg.inv(df_l), index=df_l.index, columns=df_l.columns)
    return df_li


def dump_eb_proc_pkl(d_eb):
    d_eb[cfg.s_a].to_pickle(cfg.eb_a_pkl_file_path)
    d_eb[cfg.s_li].to_pickle(cfg.eb_li_pkl_file_path)
    d_eb[cfg.s_z].to_pickle(cfg.eb_z_pkl_file_path)
    d_eb[cfg.s_x].to_pickle(cfg.eb_x_pkl_file_path)

    d_eb[cfg.s_y].to_pickle(cfg.eb_y_pkl_file_path)

    d_eb[cfg.s_imp_f_z].to_pickle(cfg.eb_imp_f_z_pkl_file_path)
    d_eb[cfg.s_imp_s_z].to_pickle(cfg.eb_imp_s_z_pkl_file_path)
    d_eb[cfg.s_imp_f_y].to_pickle(cfg.eb_imp_f_y_pkl_file_path)
    d_eb[cfg.s_imp_s_y].to_pickle(cfg.eb_imp_s_y_pkl_file_path)

    d_eb[cfg.s_sat_f_z].to_pickle(cfg.eb_sat_f_z_pkl_file_path)
    d_eb[cfg.s_sat_s_z].to_pickle(cfg.eb_sat_s_z_pkl_file_path)
    d_eb[cfg.s_sat_f_y].to_pickle(cfg.eb_sat_f_y_pkl_file_path)
    d_eb[cfg.s_sat_s_y].to_pickle(cfg.eb_sat_s_y_pkl_file_path)


def parse_eb_proc_pkl():
    d_eb = {}
    d_eb[cfg.s_a] = pd.read_pickle(cfg.eb_a_pkl_file_path)
    d_eb[cfg.s_z] = pd.read_pickle(cfg.eb_z_pkl_file_path)
    d_eb[cfg.s_y] = pd.read_pickle(cfg.eb_y_pkl_file_path)
    d_eb[cfg.s_x] = pd.read_pickle(cfg.eb_x_pkl_file_path)
    d_eb[cfg.s_li] = pd.read_pickle(cfg.eb_li_pkl_file_path)

    d_eb[cfg.s_imp_f_z] = pd.read_pickle(cfg.eb_imp_f_z_pkl_file_path)
    d_eb[cfg.s_imp_s_z] = pd.read_pickle(cfg.eb_imp_s_z_pkl_file_path)
    d_eb[cfg.s_imp_f_y] = pd.read_pickle(cfg.eb_imp_f_y_pkl_file_path)
    d_eb[cfg.s_imp_s_y] = pd.read_pickle(cfg.eb_imp_s_y_pkl_file_path)

    d_eb[cfg.s_sat_f_z] = pd.read_pickle(cfg.eb_sat_f_z_pkl_file_path)
    d_eb[cfg.s_sat_s_z] = pd.read_pickle(cfg.eb_sat_s_z_pkl_file_path)
    d_eb[cfg.s_sat_f_y] = pd.read_pickle(cfg.eb_sat_f_y_pkl_file_path)
    d_eb[cfg.s_sat_s_y] = pd.read_pickle(cfg.eb_sat_s_y_pkl_file_path)

    return d_eb


def parse_eb_cntr_agg():
    df_eb_agg = pd.read_csv(cfg.eb_cntr_agg_file_path, sep="\t", index_col=0)

    l_reg = ["EU27", "non_EU27", "EU27_NOR_CHE", "non_EU27_NOR_CHE", "ALL"]

    d_eb_cntr_agg = {}

    for reg in l_reg:
        d_eb_cntr_agg[reg] = list(df_eb_agg[reg].loc[df_eb_agg[reg] > 0].index)

    return df_eb_agg, d_eb_cntr_agg


def df2df_d(df):
    """
    df_d = pd.DataFrame(
        np.diag(df),
        index=df.index,
        columns=df.index,
    )
    return df_d

    """
    return pd.DataFrame(np.diag(df.squeeze()), index=df.index, columns=df.index)


def agg_ind(df, df_eb_ind_agg):
    df_ind = df.groupby(axis=1, level=1).sum()
    df_ind = df_ind[df_eb_ind_agg.index]
    df_ind_agg = df_ind.dot(df_eb_ind_agg)
    return df_ind_agg
