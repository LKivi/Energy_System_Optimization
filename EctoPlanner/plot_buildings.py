# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14

@author: lkivi
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_buildings(param, nodes):
    

    
    max_width = 20
    shift = 50
    for n in nodes:
        nodes[n]["x"] += shift
        nodes[n]["y"] += shift
        
    nodes[6]["x"] += 12
    nodes[6]["y"] += 12
        
        
        
        

    total_demands = np.zeros(len(nodes))
    for n in nodes: 
        total_demands[n] =  sum(sum((nodes[n]["heat"][d][t] + nodes[n]["cool"][d][t]) for t in range(24)) * param["day_weights"][d] for d in range(param["n_clusters"]))
    max_total_demand = np.max(total_demands)
    
    total = {}
    for demand in ["heat", "cool"]:
        total[demand] = np.zeros(len(nodes))
        for n in nodes:
            total[demand][n] =  sum(sum(nodes[n][demand][d][t] for t in range(24)) * param["day_weights"][d] for d in range(param["n_clusters"]))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("x [m]", fontweight = "bold", fontsize = 12)
    ax.set_ylabel("y [m]", fontweight = "bold", fontsize = 12)
    
    width = np.zeros(len(nodes))
    
    for n in nodes:
        width[n] = (total_demands[n]/max_total_demand)**0.25 * max_width
        theta = 360 * total["heat"][n]/(total["heat"][n] + total["cool"][n])
        plt.scatter(nodes[n]["x"], nodes[n]["y"])
        wedge_heat = patches.Wedge((nodes[n]["x"], nodes[n]["y"]), width[n], 0, theta, fill = True, facecolor = "red", edgecolor = "black")
        ax.add_patch(wedge_heat)
        wedge_cool = patches.Wedge((nodes[n]["x"], nodes[n]["y"]), width[n], theta, 360, fill = True, facecolor = "blue", edgecolor = "black")
        ax.add_patch(wedge_cool)

    # Tag buildings
    for n in nodes:
        if nodes[n]["name"] in ["15.1", "04.1", "16.4", "16.3"]:
            ax.text(nodes[n]["x"], nodes[n]["y"]+1.1*width[n], str(n+1), fontsize = 12, horizontalalignment='center', fontweight = "bold")
        elif nodes[n]["name"] in [""]:
            ax.text(nodes[n]["x"]+0.45*width[n], nodes[n]["y"]+0.6*width[n], str(n+1), fontsize = 12)
#            ax.plot([nodes[n]["x"]+0.2*width[n], nodes[n]["x"]+9], [nodes[n]["y"]+0.3*width[n],nodes[n]["y"]+19], "black")            
        elif nodes[n]["name"] in ["15.8"]:
            ax.text(nodes[n]["x"], nodes[n]["y"]-1.2*width[n], str(n+1), fontsize = 12, horizontalalignment='center', verticalalignment = "top")
#            ax.plot([nodes[n]["x"]-0.2*width[n], nodes[n]["x"]-8], [nodes[n]["y"]-0.3*width[n],nodes[n]["y"]-18], "black")
        elif nodes[n]["name"] in [""]:
            ax.text(nodes[n]["x"]-0.5*width[n]-20, nodes[n]["y"]+0.5*width[n], str(n+1), fontsize = 12)
#            ax.plot([nodes[n]["x"]-0.2*width[n], nodes[n]["x"]-8], [nodes[n]["y"]+0.3*width[n],nodes[n]["y"]+15], "black")          
        else:
            ax.text(nodes[n]["x"], nodes[n]["y"]+1.2*width[n], str(n+1), fontsize = 12, horizontalalignment='center')
#            ax.plot([nodes[n]["x"]+0.2*width[n], nodes[n]["x"]+8], [nodes[n]["y"]+0.3*width[n],nodes[n]["y"]+15], "black")
       
    
    ax.set_axisbelow(True)
    plt.grid(color = "grey")
  
#    plt.axis('equal')
    ax.set_xlim(0,500)
    ax.set_ylim(0,400)
    xticks =np.arange(0,600,100)
    yticks =np.arange(0,500,100)
    plt.xticks(xticks)
    plt.yticks(yticks)
    xlabels = ["{:2d}".format(x) for x in xticks]
    ylabels = ["{:2d}".format(x) for x in yticks]
    ax.set_xticklabels(xlabels, fontsize = 12)
    ax.set_yticklabels(ylabels, fontsize = 12)
 
    plt.show()
    
    
    
    # Create legend
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for n in nodes:
        plt.scatter(nodes[n]["x"], nodes[n]["y"], color = "white")
    wedges_legend = [patches.Wedge((70,325), 7.68, 0, 180, fill = True, facecolor = "red", edgecolor = "black"),
                       patches.Wedge((70,325), 7.68, 180, 360, fill = True, facecolor = "blue", edgecolor = "black"),
                       patches.Wedge((70,275), 9.65, 0, 180, fill = True, facecolor = "red", edgecolor = "black"),
                       patches.Wedge((70,275), 9.65, 180, 360, fill = True, facecolor = "blue", edgecolor = "black"),
                       patches.Wedge((70,225), 11.48, 0, 180, fill = True, facecolor = "red", edgecolor = "black"),
                       patches.Wedge((70,225), 11.48, 180, 360, fill = True, facecolor = "blue", edgecolor = "black"),
                       patches.Wedge((70,175), 13.65, 0, 180, fill = True, facecolor = "red", edgecolor = "black"),
                       patches.Wedge((70,175), 13.65, 180, 360, fill = True, facecolor = "blue", edgecolor = "black"),
                       patches.Wedge((70,125), 17.17, 0, 180, fill = True, facecolor = "red", edgecolor = "black"),
                       patches.Wedge((70,125), 17.17, 180, 360, fill = True, facecolor = "blue", edgecolor = "black"),
                       patches.Wedge((70,75), 20.4, 0, 180, fill = True, facecolor = "red", edgecolor = "black"),
                       patches.Wedge((70,75), 20.4, 180, 360, fill = True, facecolor = "blue", edgecolor = "black"),
                       ]
    for item in wedges_legend:
        ax.add_patch(item)
        
    ax.text(110, 325, "100 MWh", fontsize = 12, verticalalignment = "center")
    ax.text(110, 275, "250 MWh", fontsize = 12, verticalalignment = "center")
    ax.text(110, 225, "500 MWh", fontsize = 12, verticalalignment = "center")
    ax.text(110, 175, "1000 MWh", fontsize = 12, verticalalignment = "center")
    ax.text(110, 125, "2500 MWh", fontsize = 12, verticalalignment = "center")
    ax.text(110, 75, "5000 MWh", fontsize = 12, verticalalignment = "center")

    ax.set_xlim(0,500)
    ax.set_ylim(0,400)
    xticks =np.arange(0,600,100)
    yticks =np.arange(0,500,100)
    plt.xticks(xticks)
    plt.yticks(yticks)
    xlabels = ["{:2d}".format(x) for x in xticks]
    ylabels = ["{:2d}".format(x) for x in yticks]
    ax.set_xticklabels(xlabels, fontsize = 12)
    ax.set_yticklabels(ylabels, fontsize = 12)
    
#    ax.grid(False)
#    ax.set_xticks([])
#    ax.set_yticks([])

    plt.show()    
    
    
    
    

 #    # 15.1 (Labor)
#    ax.text(nodes[0]["x"]+8, nodes[0]["y"]+22, "1", fontsize = 12)
#    ax.plot([nodes[0]["x"]+3, nodes[0]["x"]+9], [nodes[0]["y"]+3,nodes[0]["y"]+19], "black")
#    # 04.01 (Restaurant)
#    ax.text(nodes[1]["x"]+8, nodes[1]["y"]+22, "2", fontsize = 12)
#    ax.plot([nodes[1]["x"]+3, nodes[1]["x"]+9], [nodes[1]["y"]+3,nodes[1]["y"]+19], "black")
#    # 16.4 (Rechenzentrum)
#    ax.text(nodes[2]["x"]+8, nodes[2]["y"]+22, "3", fontsize = 12)
#    ax.plot([nodes[2]["x"]+3, nodes[2]["x"]+9], [nodes[2]["y"]+3,nodes[2]["y"]+19], "black")
#    # 16.3 (Rechenzentrum)
#    ax.text(nodes[3]["x"]+8, nodes[3]["y"]+22, "4", fontsize = 12)
#    ax.plot([nodes[3]["x"]+3, nodes[3]["x"]+9], [nodes[3]["y"]+3,nodes[3]["y"]+19], "black")
#    # 15.13
#    ax.text(nodes[4]["x"]+8, nodes[4]["y"]+22, "5", fontsize = 12)
#    ax.plot([nodes[4]["x"]+3, nodes[4]["x"]+9], [nodes[4]["y"]+3,nodes[4]["y"]+19], "black")    
#    # 15.8
#    ax.text(nodes[5]["x"]-18, nodes[5]["y"]-35, "6", fontsize = 12)
#    ax.plot([nodes[5]["x"]-4, nodes[5]["x"]-9], [nodes[5]["y"]-5,nodes[5]["y"]-19], "black")     
#    # 15.7
#    ax.text(nodes[6]["x"]+8, nodes[6]["y"]+22, "7", fontsize = 12)
#    ax.plot([nodes[6]["x"]+3, nodes[6]["x"]+9], [nodes[6]["y"]+3,nodes[6]["y"]+19], "black")     
#    # 15.14
#    ax.text(nodes[7]["x"]+8, nodes[7]["y"]+22, "7", fontsize = 12)
#    ax.plot([nodes[7]["x"]+3, nodes[7]["x"]+9], [nodes[7]["y"]+3,nodes[6]["y"]+19], "black")     
    
    #ax.text(317705,5642825,"04.01", zorder = 1000, fontsize = size)   
