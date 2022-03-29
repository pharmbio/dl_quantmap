#!/usr/bin/env python

import sys, os
import sqlite3
import numpy as np
import pandas as pd
import igraph
from igraph import Graph
# from sklearn.cluster import AgglomerativeClustering
# import scipy
# from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt




def main():
    infile = str(sys.argv[1])
    utfile = str(sys.argv[2])
    frames = {}

    chems = getdata(infile)
    for cid in chems.keys():
        
        seeds = get_seeds(cid)
        ppi = get_ppi(seeds)
        cents = centrality(ppi)
        coal = condense(cents)

        name = cid
        frames[name] = coal
    ppi_all = pd.DataFrame(frames)
    write_ppi_out(utfile,ppi_all)


def write_ppi_out(filename,panda_df):
    #os.system("mkdir ppi_results")
    panda_df.to_csv("../data/ppi_results/" + filename +".csv", sep=',')
    
def getdata(infile):
    dat = {int(line.strip().split("\t")[0]):line.strip().split("\t")[1] for line in open(infile,"r").readlines() if chk_cid(int(line.strip().split("\t")[0]))}
    assert len(dat) > 3 
    return dat


def chk_cid(cid):
    sql = f"select 1 from {chem_tbl} where cid = ?"
    sth.execute(sql, (cid,))
    return sth.fetchone()


def get_seeds(cid):
    sql = (f"select distinct protein from {chem_prot} where cid = ? "
           f"and {prot_type} >= {chem_score} order by {prot_type} "
           f"desc, sc_exp desc limit {chem_max}") # ; print(sql)
    sth.execute(sql, (cid,))
    # return sth.fetchall()
    res = [x[0] for x in sth.fetchall()]
    return res


def get_ppi(seeds):
    prots  = "('"  +  "','".join(seeds)  +  "')"
    inc_wt = ",sc_all/1000.0 weight" if weight else ""
    sql = (f"select pro1,pro2 {inc_wt} from {prot_prot} where pro1 in "
           f"{prots} and (pro1 < pro2 or pro2 not in {prots}) and "
           f"{prot_type} >= {prot_score} order by {prot_type} desc, "
           f"sc_exp desc, pro1, pro2 limit {prot_max}") # ; print(sql)
    sth.execute(sql)
    return sth.fetchall()


def centrality(dat):

    g = Graph.TupleList(dat, weights=True)
    nam = g.vs["name"]

    res = {}
    res["con"] = g.is_connected()
    res["deg"] = g.degree()
    res["bet"] = g.betweenness(weights=g.es['weight'])
    res["clo"] = g.closeness(weights=g.es['weight'])
    res["con"] = g.constraint(weights=g.es['weight'])
    res["evc"] = g.eigenvector_centrality(weights=g.es['weight'])
    res["hub"] = g.hub_score(weights=g.es['weight'])
    res["pag"] = g.pagerank(weights=g.es['weight'])
    res["ppg"] = g.personalized_pagerank(weights=g.es['weight'])
    res["str"] = g.strength(weights=g.es['weight']) # degree with weights

    df = pd.DataFrame(res, index = nam)
    df *= -1
    return df.rank()  


def condense(df):
    df = df.median(axis=1)
    df = df.sort_values()
    return df.iloc[:ppi_max]


def compare_ppi(dat):
    chem = list(dat.columns)
    res = foot_rule(dat, chem)
    res = resize(res)
    
    dists = squareform(res)
    links = linkage(dists, "complete")
    dendrogram(links, labels=chem, orientation='left')
    plt.title("Qmap")
    plt.savefig(sys.argv[2][:-4] + ".png")
    return (res)

def resize(dat):
    return (dat - dat.min()) / (dat.max() - dat.min())


def foot_rule(dat, chem):
    dat = dat.fillna(0)
    # chem = list(dat.columns)
    nchem = len(chem)
    mat = np.zeros([nchem,nchem])
    for i,x in enumerate(chem):
        for j,y in enumerate(chem):
            if i >= j: continue
            z1 = dat[x][(dat[x] > 0) & (dat[y] > 0)]
            z2 = dat[y][(dat[x] > 0) & (dat[y] > 0)]
            z3 = sum(abs(z1 - z2))
            z4 = sum((dat[x]==0) ^ (dat[y]==0))
            z5 = (sum((dat[x] > 0) | (dat[y] > 0)) + 1)
            mat[i][j] = z3 + z4 * z5
            mat[j][i] = mat[i][j]
    return mat


def warn(msg):
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    # wrap with catch, etc
    # select db as arg?
    dbf = sys.argv[3]
    stitch_table_name = sys.argv[4]
    string_table_name = sys.argv[5]

    dbh = sqlite3.connect(dbf)
    sth = dbh.cursor()


    verbose     = False
    chem_tbl    = "stitch_protchem_man" #"common_chem"
    fail_no_cid = False
    chem_max    = 10
    chem_score  = 700
    chem_type   = "all"
    chem_prot   = stitch_table_name

    prot_max    = 150
    prot_score  = 700
    prot_type   = "sc_all"
    prot_prot   = string_table_name

    data_max    = 1000000
    ppi_max     = 200
    weight      = True
    names_long  = False
    plot_fancy  = True
    plot_high   = False
    plot_title  = True
    main()
