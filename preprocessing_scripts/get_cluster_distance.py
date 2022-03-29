import sys
import glob
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool




cid_cluster_files = sys.argv[1] # Input CID cluster file
cluster_support = int(sys.argv[2])  # This puts a threshold on clusters to select. Select clusters with support >= given number.


distance_threshold = float(cid_cluster_files.split("_")[-1][:-4])
cid_cluster_filename = "../data/cluster_results/cid_cluster_" + str(distance_threshold) + ".txt"
node_details_filename = "../data/cluster_results/clustering_details_" + str(distance_threshold) + ".csv"
cid_order_list = list(map(lambda x:int(x),open("../data/cid_order_file.txt","r").readlines()))
cluster_reference_file = {int(entry.split()[1]):int(entry.split()[0]) for entry in open("../data/preprocessed_data/cluster_reference_" + str(distance_threshold) + ".txt","r").readlines()}
cluster_distance_output_filename = "../data/cluster_results/cluster_distance_" + str(distance_threshold) + ".csv"


def find_cid_positions_in_node(node):
    
    if node in node_cids:
        return node_cids[node]
        
    cids = []
    for dicts in node_details:
        if dicts["node_id"] == node:
            left_node = dicts["left"]
            right_node = dicts["right"]
            not_found = False
            break
        not_found = True
        
    if left_node < total_samples:
        cid = cid_order_list[left_node]
        if cid in cid_cluster:
            cids.append(cid)
    else:
        if left_node not in node_cids:
            found_cids = find_cid_positions_in_node(left_node)
            cids.extend(found_cids)
            node_cids[left_node] = found_cids
        else:
            cids.extend(node_cids[left_node])
        
    if right_node < total_samples:
        cid = cid_order_list[right_node]
        if cid in cid_cluster:
            cids.append(cid)
    else:
        if right_node not in node_cids:
            found_cids =  find_cid_positions_in_node(right_node)
            cids.extend(found_cids)
            node_cids[right_node] = found_cids
        else:
            cids.extend(node_cids[right_node])
    
    output_cids = []
    already_present_cluster = []
    for cid in cids: 
        if cid_cluster[cid] not in already_present_cluster:
            output_cids.append(cid)
            already_present_cluster.append(cid_cluster[cid])

    return (output_cids)

def get_cluster_distance_two_cids(left_cids,right_cids,distance):
    for left_cid in left_cids:
        for right_cid in right_cids:
            if str(left_cid) + "_" + str(right_cid) not in already_found_cids and str(right_cid) + "_" + str(left_cid) not in already_found_cids:    
                left_cluster, right_cluster = cid_cluster[left_cid],cid_cluster[right_cid]
                if str(left_cluster) + "_" + str(right_cluster) not in already_found_clusters and \
                str(right_cluster) + "_" + str(left_cluster) not in already_found_clusters:
                    cluster_distance.append({"cluster1":left_cluster,"cluster2":right_cluster,"distance":distance})
                    already_found_clusters.append(str(left_cluster) + "_" + str(right_cluster))
                    already_found_cids.append(str(left_cid) + "_" + str(right_cid))
                    
                    
                    
# Get cid:cluster dict
with open(cid_cluster_filename,"r") as f:
    cid_cluster_all = {}
    for entry in f.readlines():
        cid_cluster_all[int(entry.split()[0])] = int(entry.split()[1])
        
# Read node_details from clustering output
with open(node_details_filename,"r") as f:
    node_details = [] 
    for i,entry in enumerate(f.readlines()):
        if i != 0:
            x = entry.split(",")
            node_details.append({'node_id': int(x[0]), 'left': int(x[1]), 'right': int(x[2]), 'distance' : float(x[3])})
            
            
            
# Make cluster:[cid1,cid2]
cluster_cids_all = {cluster_id:[] for cluster_id in list(map(int,set(list(cid_cluster_all.values()))))}
for cid in cid_cluster_all:
    cluster_cids_all[cid_cluster_all[cid]].append(cid)
    
# cluster:[cid1,cid2] with support more than "cluster_support"
cluster_cids = {}
for cluster in cluster_cids_all:
    if len(cluster_cids_all[cluster]) >= cluster_support:
        cluster_cids[cluster] = cluster_cids_all[cluster]
        
        
        
# Get cid:cluster dict with support more than "cluster_support"
cid_cluster = {}
for cid in cid_cluster_all:
    if cid_cluster_all[cid] in cluster_cids:
        cid_cluster[cid] = cid_cluster_all[cid]
        
        
        
# Get order of cid
cid_order = {}
for cid in cid_order_list:
    if cid in cid_cluster:
        cid_order[cid] = True
    else:
        cid_order[cid] = False
        
        
total_clusters = len(cluster_cids_all)
total_samples = len(cid_cluster_all)

all_nodes = []
for dicts in node_details:
    all_nodes.append(dicts["node_id"])
    
    
# Find cid in each node
node_cids = {}
loop = tqdm.tqdm(all_nodes,total=len(all_nodes),leave=False)
for node in loop:
    if node not in node_cids:
        node_cids[node] = find_cid_positions_in_node(node)
    loop.set_description("STEP 1")
    
    
already_found_clusters = []
cluster_distance = []
already_found_cids = []
node_details_inverted = node_details[::-1]

loop = tqdm.tqdm(node_details_inverted,total=len(node_details_inverted),leave=False)
for dicts in loop:
    distance = dicts["distance"]
    if distance > distance_threshold:
        left_node = dicts["left"]
        right_node = dicts["right"]
        
        if left_node > total_samples:
            left_cids = node_cids[left_node]
            left_condition = True
        else:
            left_cids = [cid_order_list[left_node]]
            left_condition = cid_order[left_cids[0]]
            
        if right_node > total_samples:
            right_cids = node_cids[right_node]
            right_condition = True
        else:
            right_cids = [cid_order_list[right_node]]
            right_condition = cid_order[right_cids[0]]
            
        if left_condition and right_condition and len(left_cids) > 0 and len(right_cids) > 0:
            output_clusters = get_cluster_distance_two_cids(left_cids,right_cids,distance)
    loop.set_description("STEP 2")
    
    
out_file = open(cluster_distance_output_filename,"w")
for dicts in cluster_distance:
    out_file.write(str(cluster_reference_file[int(dicts["cluster1"])]) + "," + str(cluster_reference_file[int(dicts["cluster2"])]) + "," + str(dicts["distance"]) + "\n")
out_file.close()
