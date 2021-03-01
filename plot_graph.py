#!/usr/bin/python3

import sys
import igraph as ig


# draw graph function -------------------------------
def draw_g (n,coords,edges,dist,path,ofile,di=True):
	
	# create graph object
	g = ig.Graph(directed=di)
	
	# populate graph
	g.add_vertices(range(0,n))
	g.add_edges(edges)
	
	# name input (for reference by name instead of id)
	g.vs["name"] = list(range(1,n+1))
	g.es["name"] = edges
	
	# initialize visual_style options (just a dict)
	visual_style = {}
	
	# use coordinate-layout for vertex placement in plot
	visual_style["layout"] = coords
	visual_style["keep_aspect_ratio"] = True
	visual_style["bbox"] = (1800, 1800)
	
	# a few settings (and different ways to set them)
	visual_style["vertex_label"] = g.vs["name"]
	visual_style["vertex_size"] = 30
	g.es['label'] = dist
	g.vs["color"] = ["lightgreen"] + ["grey"]*(n-1)
	visual_style["edge_color"] = ["black" if (e["name"] in path) else "black" for e in g.es]
	
	# plot graph
	ig.plot(g, ofile, **visual_style)
	
	return 0
# ---------------------------------------------------



# main ----------------------------------------------
if __name__ == "__main__":
	
	# instructions
	if ( len(sys.argv) < 2 ):
		print("Usage: python plot_graph.py ofile")
		print("\t - ofile\t output file")
		sys.exit(-1)
	
	out_file = sys.argv[1]
	
	# example graph with 6 nodes, 7 edges and a path
	n         = 6
	coords    = [(0,0.5),(1,0),(1,1),(2,0),(2,1),(3,0.5)]
	edges     = [(0,1),(0,2),(1,3),(1,4),(2,4),(3,5),(4,5)]
	distances = [42] * len(edges)
	path      = [(0,1),(1,4),(4,5)]
	
	draw_g(n,coords,edges,distances,path,out_file)
# ---------------------------------------------------
