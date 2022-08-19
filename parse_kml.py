# Derived from code from @author: Yanchao Liu, see vorheur.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from matplotlib.backends.backend_pdf import PdfPages
import utm
import pandas as pd
from shapely.geometry import Polygon

def parse_kml(kml_file, hole_file_names):

    global corners, edges, holes_polygon, hole_files
    hole_files = hole_file_names
    # Load corners from KML file
    border = 0
    pd.set_option('display.max_colwidth',1000000)
    data = pd.read_table(kml_file,sep='\r\t',header=None,skip_blank_lines=False,engine='python')
    foundlable = 0
    for i in range(0,len(data)):
        strl = data.iloc[i].to_frame().T
        strl2 = strl.to_string()
        strlist = strl2.split()
        if strlist[2] == '<coordinates>':
            foundlable = 1
            continue
        if foundlable == 1:
            break
    minx = 1000000000000
    miny = 1000000000000
    maxx = 0
    maxy = 0
    location = list()
    utmloc = dict()
    for i in range(2,len(strlist)):
        location = strlist[i].split(",")
        templst = utm.from_latlon(float(location[1]),float(location[0]))
        #print(templst)
        if templst[0] <= minx:
            minx = templst[0]
        if templst[0] >= maxx:
            maxx = templst[0]
        if templst[1] <= miny:
            miny = templst[1]
        if templst[1] >= maxy:
            maxy = templst[1]
        temploc = {str(i-1):
                 {'x': templst[0],
                 'y': templst[1]}}
        utmloc.update(temploc)
    utmnumber = templst[2]
    utmletter = templst[3]
    lenx = maxx - minx
    leny = maxy - miny
    midx = (maxx + minx)/2
    midy = (maxy + miny)/2
    sqlen = max(lenx,leny)
    origx = midx - sqlen*(0.5 + border)
    origy = midy - sqlen*(0.5 + border)
    
    location = list()
    sqloc = dict()
    for i in range(3,len(strlist)):
        location = strlist[i].split(",")
        templst = utm.from_latlon(float(location[1]),float(location[0]))
    
        temploc = {str(i-1):
                 {'x': (templst[0]-origx)/sqlen/(1+border*2)*100,
                 'y': (templst[1]-origy)/sqlen/(1+border*2)*100}}
        sqloc.update(temploc)
    corners = np.empty((0,2))
    for i in sqloc.keys():
        corners = np.append(corners, np.array(list(sqloc[i].values())))
    corners.resize((len(sqloc),2))
    
    # Now load hole files
    holes = {}  # empty dictionary to hold hole name and coordinate pairs
    for this_hole in hole_file_names:
        data = pd.read_table(this_hole + ".kml",sep='\r\t',header=None,skip_blank_lines=False,engine='python')
        foundlable = 0
        for i in range(0,len(data)):
            strl = data.iloc[i].to_frame().T
            strl2 = strl.to_string()
            strlist = strl2.split()
            if strlist[2] == '<coordinates>':
                foundlable = 1
                continue
            if foundlable == 1:
                break
        location = list()
        sqloc = dict()
        for i in range(3,len(strlist)):
            location = strlist[i].split(",")
            templst = utm.from_latlon(float(location[1]),float(location[0]))
    
            temploc = {str(i-1):
                     {'x': (templst[0]-origx)/sqlen/(1+border*2)*100,
                     'y': (templst[1]-origy)/sqlen/(1+border*2)*100}}
            sqloc.update(temploc)
        hole = np.empty((0,2))
        for i in sqloc.keys():
            hole = np.append(hole, np.array(list(sqloc[i].values())))
        hole.resize((len(sqloc),2))
        holes.update({this_hole: hole})
    
    
    # Create outer boundary edges, edge_equations and polygon
    edges = np.empty((0,2), dtype = int)
    for i in range(len(corners)-1):
        edges = np.append(edges, np.reshape([i, i+1], (1,2)), axis = 0)
    edges = np.append(edges, np.reshape([len(corners)-1, 0], (1,2)), axis = 0)
    edge_equations = np.empty((len(edges),3))
    for i in range(len(edges)):
        x1 = corners[edges[i,0],0]
        y1 = corners[edges[i,0],1]
        x2 = corners[edges[i,1],0]
        y2 = corners[edges[i,1],1]
        edge_equations[i,0] = y2 - y1
        edge_equations[i,1] = x1 - x2
        edge_equations[i,2] = edge_equations[i,0]*x1 + edge_equations[i,1]*y1
    polygon = Polygon(corners)
    
    # Create the boundary edges, edge_equations and polygon for each hole
    holes_edges = {}
    holes_edge_equations = {}
    holes_polygon = {}
    holes_corners = []
    for this_hole in hole_file_names:
        this_hole_corners = holes[this_hole]
        holes_corners.append(this_hole_corners)
        this_hole_edges = np.empty((0,2), dtype = int)
        for i in range(len(this_hole_corners)-1):
            this_hole_edges = np.append(this_hole_edges, np.reshape([i, i+1], (1,2)), axis = 0)
        this_hole_edges = np.append(this_hole_edges, np.reshape([len(this_hole_corners)-1, 0], (1,2)), axis = 0)
        this_hole_edge_equations = np.empty((len(this_hole_edges),3))
        for i in range(len(this_hole_edges)):
            x1 = this_hole_corners[this_hole_edges[i,0],0]
            y1 = this_hole_corners[this_hole_edges[i,0],1]
            x2 = this_hole_corners[this_hole_edges[i,1],0]
            y2 = this_hole_corners[this_hole_edges[i,1],1]
            this_hole_edge_equations[i,0] = y2 - y1
            this_hole_edge_equations[i,1] = x1 - x2
            this_hole_edge_equations[i,2] = this_hole_edge_equations[i,0]*x1 + this_hole_edge_equations[i,1]*y1
        this_hole_polygon = Polygon(this_hole_corners)
        holes_edges.update({this_hole:this_hole_edges})
        holes_edge_equations.update({this_hole:this_hole_edge_equations})
        holes_polygon.update({this_hole:this_hole_polygon})
    
    # How many line segments in outer polygon?
    print("Outer polygon has {0:d} line segments".format(len(edges)))
    print("There are {:d} holes: ".format(len(holes)))
    total_hole_segs = 0
    for key in holes_edges:
        print(key + " has {0:d} line segments".format(len(holes_edges[key])))
        total_hole_segs = total_hole_segs + len(holes_edges[key])
    print("Total hole line segments: {:d}".format(total_hole_segs))
   
    return corners, holes_corners

def plot(pdf, finish_points, finish_dist, ndepots, demands = None):
    # Plot solution
    xmax = np.max(corners[:,0])
    xmin = np.min(corners[:,0])
    ymax = np.max(corners[:,1])
    ymin = np.min(corners[:,1])
    plt.close()
    plt.figure(figsize=(6.4, 4.8))
    plt.axis('scaled')
    plt.axis('off')
    pmar = 10
    plt.xlim((xmin-pmar,xmax+pmar))
    plt.ylim((ymin-pmar-10,ymax+pmar))
    for simplex in edges:
        plt.plot(corners[simplex,0], corners[simplex,1], 'k-')
    for this_hole in hole_files:
        x,y = holes_polygon[this_hole].exterior.xy
        plt.plot(x,y)
    plt.plot(finish_points[:, 0], finish_points[:, 1], 'o')
    if not demands is None:
        plt.scatter(demands[:, 0], demands[:, 1], s=1, c='bisque')
    fig = plt.gcf()
    fig.set_size_inches(8,8)
    ax = fig.gca()
    for i in range(ndepots):
        circle = plt.Circle((finish_points[i,0], finish_points[i,1]), finish_dist, color='b', fill=False)
        ax.add_artist(circle)
    plt.savefig(pdf +'.pdf')
