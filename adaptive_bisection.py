"""
Adaptive angular bisection search along the circumference of a circle; max travel distance.

This module implements a modified adaptive bisection to choose the next destination
of the linear search.
- Looks for the largest angle in the max travel distance circle
- Divides that angle by two to create a vector of interest
- Calls a linear search on that vector
- Recursively Calls itself until either the target is found or we are certain that the
  target is outside our drone flight capabilities

Major Variables:
*Note a is typically representative of an angle related variable
- adist: The distance between 2 angle vectors
- a_mid: the bisection result of 2 angle vectors
- max_arclength: the maximum unsearched arclength on the circle
- max_adist: the largest unsearched angle on the searching circle
- max_dist_rad: the maximum distance the drone can travel in a stright line
- next_search: desired searching angle and position
- point_list: a list of points already searched on the circumference of our
  potential search cirlce
- rad_search: the radius of the scanning device's search
- target_maybe: results from our linear search
- tol: the arc between vectors that indicates we can exit the search

Please note that this is meant to be called from another driver and is not meant to operate alone

Authors: Alisha Moncha, Aydin Khan, Kamran Hussain, Reagan Ross
"""

"""
unfinished things:
- debugging (if needed)
- connect to driver
- validation
"""


import math
import numpy as np
from pathlib import Path
from search_figure import simulate_search_vector
#^ this function takes in an angle (make sure this is consistent either degrees or radians)

"""Should find the angular distance between two different points on the circumference of a circle"""
def angle_calc(pointA, pointB):
    #get vector angle
    x1, y1 = pointA
    x2, y2 = pointB
    a1 =  math.atan2(y1, x1)
    a2 = math.atan2(y2, x2)
    #distance between angles
    adist = a2 - a1
    #makes sure angle is always within [0,2pi]
    if adist < 0:
        adist += 2*math.pi
    return adist


"""Finds the point we will plug into linear_search"""
def angular_bisection(adist, pointA, max_dist_rad):
    x1, y1 = pointA
    a1 = math.atan2(y1, x1)
    if a1 < 0:
        a1 += 2*math.pi
    #getting midpoint angle: midpoint angle = left bound vector angle + angle distance/2
    a_mid = a1 + adist/2
    a_mid %= 2*math.pi
    #get actual midpoint (trig double check later)
    x_mid = max_dist_rad * math.cos(a_mid)
    y_mid = max_dist_rad * math.sin(a_mid)
    print("Current angle: ", a_mid)
    
    return a_mid, (x_mid,y_mid)

"""sort the list in order [0,...,2pi]"""
def sort_list(list_p):
    #rules for how the list is sorted
    def ccw(point):
        x, y = point
        a = math.atan2(y, x)
        if a < 0:
            a += 2*math.pi 
        return a
    #actually sorting the list
    list_p.sort(key=ccw)
    return list_p

"""Run the adaptive angular bisection until success"""
def adaptive_model(params, rad_search, max_dist_rad = None, point_list = None, max_arclength= None, n = None): 
    #point_list will need to be empty in the first step of recursion and first step only
    #max_arclength is updated each recursive iteration to be the maximum arclength along the circumference
    #max_dist_rad is the maximum distance our drone can travel
    #rad_search is the radius of the searching device


    #base cases
    if point_list is None:
        point_list = []
        #use lsm on 0 degree of unit circle
        target_maybe = simulate_search_vector(0, params)
        print("Linear search count: 1")
        if target_maybe[6] == True:
            return target_maybe
        

        #calculate max_dist_rad from linear search results
        full_trajectory = target_maybe[0]
        r = np.hypot(full_trajectory[:,0] - params.X0, full_trajectory[:,1] - params.Y0)
        max_dist_rad = np.max(r)
        print(max_dist_rad)

        point_list.append((max_dist_rad, 0.0)) #in list write starting point aka point at angle 0
        return adaptive_model(params, rad_search, max_dist_rad, point_list, 2*math.pi*max_dist_rad, 1)
    elif len(point_list) == 1:
        point_list.append((-1* max_dist_rad, 0.0))
        target_maybe = simulate_search_vector(math.pi, params)
        print("Linear Search count: 2")
        if target_maybe[6] == True:
            #print(target_maybe)
            return target_maybe
        return adaptive_model(params, rad_search, max_dist_rad, point_list, max_dist_rad*math.pi, 2)

    #recursive case
    #if we still have space on the circumference wil trigger
    #tol = 2 * R *sin(r/R)
    elif max_arclength > 2*max_dist_rad *math.asin(rad_search/max_dist_rad):
        n += 1
        print("Linear search count: ", n)
        max_adist = 0
        for i in range(len(point_list)):
        #we are going to pull two points
            A = point_list[i]
            #just making sure it also get the max between the past point and 1st point in point_list
            j = i + 1
            if j == len(point_list):
                j = 0
            B = point_list[j]
            curr_adist = angle_calc(A, B)
            if curr_adist > max_adist + 1e-12: #order debugging tol
                max_adist = curr_adist
                A_keep = A


        #in the case of max_adist == curr_adist, we can choose to do nothing since
        #max_adist comes from an earlier set of points than curr_adist, and we want
        #to travel ccw on a list of sorted points

        #after we add the above, we search the max adist
        next_search = angular_bisection(max_adist, A_keep, max_dist_rad)
        #call ssv for the wanted vector
        target_maybe = simulate_search_vector(next_search[0], params) #use next_search results in parameters
        if target_maybe[6] == True:
            #print(target_maybe)
            return target_maybe
        point_list.append(next_search[1]) 
        sort_list(point_list)

        #compute the new maximum arclength based off:
        #arclength formula (if theta is in radians) s=r*theta
        new_max_al = max_dist_rad*max_adist

        #recursively iterate
        return adaptive_model(params, rad_search, max_dist_rad,  point_list, new_max_al, n)

    else:
        print("No target point within maximum searching range")
    return


