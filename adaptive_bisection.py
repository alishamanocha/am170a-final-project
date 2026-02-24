"""
Adaptive angular bisection search along the circumference of a circle; max travel distance.

This module implements a modified adaptive bisection to choose the next destination
of the linear search.
- Looks for the largest angle in the max travel distance circle
- Divides that angle by two to create a vector of interest
- Calls a linear search on that vector
- Recursively Calls itself until either the target is found or we are certain that the
  target is outside our drone flight capabilities

Authors: Alisha Moncha, Aydin Khan, Kamran Hussain, Reagan Ross
"""

"""
unfinished things:
- debugging (if needed)
- sort list func
- connect to driver
- validation
"""


import math
#import numpy as np <- replace math with this
from pathlib import Path
from search_figure import simulate_search_vector
#^ this function takes in an angle (make sure this is consistent either degrees or radius)



"""Should find the angular distance between two different points on the circumference of a circle"""
def angle_calc(pointA, pointB):
    #get vector angle
    x1, y1 = pointA
    x2, y2 = pointB
    theta1 =  math.atan2(y1, x1)
    theta2 = math.atan2(y2, x2)
    #distance between angles
    adist = theta2 - theta1
    #makes sure angle follows correct formatting for ease of use later
    if adist < 0:
        adist += 2*math.pi
    return adist

"""Finds the point we will plug into linear_search"""
def angular_bisection(adist, pointA, max_dist_rad):
    x1, y1 = pointA
    theta1 = math.atan2(y1, x1)
    if theta1 < 0:
        theta1 += 2*math.pi
    #getting midpoint angle: midpoint angle = left bound vector angle + angle distance/2
    a_mid = theta1 + adist/2
    #get actual midpoint (trig double check later)
    x_mid = max_dist_rad * math.cos(a_mid)
    y_mid = max_dist_rad * math.sin(a_mid)
    return (x_mid,y_mid)

"""sort the list in order [0,...,2pi]"""
def sort_list(list):
    return

"""Run the adaptive angular bisection until success"""
def adaptive_model(max_dist_rad, rad_search, point_list = None, max_arclength= None): 
    #point_list will need to be empty in the first step of recursion and first step only
    #max_chord is going to be precaclulated in one of the driver funcs
    #max_dist_rad is the maximum distance our drone can travel
    #rad_search is the radius of the searching device

    #tol = 2 * R *sin(r/R)
    tol = 2*max_dist_rad *math.asin(rad_search/max_dist_rad)

    #base cases
    if point_list is None:
        point_list = []
        point_list.append((max_dist_rad, 0.0)) #in list write starting point aka point at angle 0
        #use lsm on 0 degree of unit circle
        simulate_search_vector(0)
        return adaptive_model(max_dist_rad, rad_search, point_list, 2*math.pi*max_dist_rad)
    elif len(point_list) == 1:
        point_list.append((-1* max_dist_rad, 0.0))
        simulate_search_vector(math.pi)
        return adaptive_model(max_dist_rad, rad_search, point_list, max_dist_rad*math.pi)

    #recursive case
    #if the we still have space on the circumference
    elif max_arclength > tol:
        max_adist = 0
        for i in range(len(point_list) - 1): #I need to fix this its skipping things
        #we are going to pull two points
            A = point_list[i]
            B = point_list[i + 1]
            curr_adist = angle_calc(A, B)
            if curr_adist > max_adist:
                max_adist = curr_adist
                A_keep = A

        #in the case of max_adist == curr_adist, we can afford to do nothing since
        #max_adist comes from an earlier set of points than curr_adist, and we want
        #to travel ccw on a list of sorted points

        #after we add the above, we search the max adist
        next_search = angular_bisection(max_adist, A_keep, max_dist_rad)
        #call lsm for the wanted vector
        target_maybe = simulate_search_vector(next_search) #use next_search results in parameters
        if target_maybe == True: #I will edit linear search to return something boolean if target was found
            print("Target found at:", target_maybe)
            return target_maybe
        point_list.append(next_search) 
        sort_list(point_list)

        #compute the new maximum arclength based off:
        #arclength formula (if theta is in radians) s=r*theta
        new_max_al = max_dist_rad*max_adist

        #recursively iterate
        return adaptive_model(max_dist_rad, rad_search, point_list, new_max_al)

    else:
        print("No target point within maximum searching range")
    return
