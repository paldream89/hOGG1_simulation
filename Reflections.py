# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:20:07 2024

@author: Admin
"""

import numpy as np
from math import floor,sqrt,e,pi
from scipy import optimize

def reflect_circle(x0,y0,x1,y1,radius,digit):
    
    x_max = max(x0,x1)
    x_min = min(x0,x1)
    y_max = max(y0,y1)
    y_min = min(y0,y1)
    x_cross=0
    y_cross=0
    
    if x1 == x0:
        
        x_cross = x0
        ys1 = sqrt(radius**2-x_cross**2)
        ys2 = -sqrt(radius**2-x_cross**2)
        if ys1>=y_min and ys1<=y_max:
            y_cross = ys1
        else:
            if ys2>=y_min and ys2<=y_max:
                y_cross = ys2
            else:
                print('x0: %.3f;  y0: %.3f; x1: %.3f;  y1: %.3f; ' %(x0,y0,x1,y1))
                print('Wrong-x1=x0')
        
    else:
        
        s = (y1-y0)/(x1-x0)
        a = 1+ s**2
        b = 2*s*(y0-x0*s)
        c = (y0-x0*s)**2-radius**2
        delta = b**2-4*a*c
        if delta<0:
            delta=0
        xs1 = (-b+sqrt(delta))/2/a
        xs2 = (-b-sqrt(delta))/2/a
        
        if xs1>=x_min and xs1<=x_max:
            x_cross = xs1
            y_cross = ((y1-y0)*x_cross+(x1*y0-x0*y1))/(x1-x0)
                
        else:
            if xs2>=x_min and xs2<=x_max:
                x_cross = xs2
                y_cross = ((y1-y0)*x_cross+(x1*y0-x0*y1))/(x1-x0)

            else:
                print('x0: %.3f;  y0: %.3f; x1: %.3f;  y1: %.3f; ' %(x0,y0,x1,y1))
                print('delta: %.3f' %delta)
                print('Wrong')
    
    r = np.array([x_cross,y_cross])
    # x*r[0]+y*r[1]-radius**2 ：切线方程 A=r[0],B=r[1],C=-radius**2
    
    x_cross = np.fix(x_cross*digit)/digit
    y_cross = np.fix(y_cross*digit)/digit
    
    x1 = x1-2*r[0]*(r[0]*x1+r[1]*y1-radius**2)/(r[0]**2+r[1]**2)
    x1 = np.fix(x1*digit)/digit
    y1 = y1-2*r[1]*(r[0]*x1+r[1]*y1-radius**2)/(r[0]**2+r[1]**2)
    y1 = np.fix(y1*digit)/digit
    
    return x_cross,y_cross,x1,y1





# def reflect_circle_iteration(x0,y0,x1,y1,radius):
    
#     # x0 y0 is on the circle
#     # x_max = max(x0,x1)
#     # x_min = min(x0,x1)
#     x_cross=0
#     y_cross=0
    
#     s = (y1-y0)/(x1-x0)
#     a = 1+ s**2
#     b = 2*s*(y0-x0*s)
#     c = (y0-x0*s)**2-radius**2
#     delta = b**2-4*a*c
#     if delta<0:
#         delta=0
#     xs1 = (-b+sqrt(abs(delta)))/2/a
#     xs2 = (-b-sqrt(abs(delta)))/2/a
    
#     if abs(xs1-x0)>abs(xs2-x0):
#         # print('xs1 Yes')
#         x_cross = np.fix(xs1*100)/100
#         y_cross = ((y1-y0)*x_cross+(x1*y0-x0*y1))/(x1-x0)
#     else:
#         if abs(xs1-x0)<abs(xs2-x0):
#             # print('xs2 Yes')
#             x_cross = np.fix(xs2*100)/100
#             y_cross = ((y1-y0)*x_cross+(x1*y0-x0*y1))/(x1-x0)
#         # else:
#         #     print('Wrong')
    
#     r = np.array([x_cross,y_cross])
#     # x*r[0]+y*r[1]-radius**2 ：切线方程 A=r[0],B=r[1],C=-radius**2
#     if abs(xs1-x0)==abs(xs2-x0):
        
#         print('x0: %.3f;  y0: %.3f; x1: %.3f;  y1: %.3f; ' %(x0,y0,x1,y1))
        
#         x_cross = x0
#         y_cross = y0
#         x1 = x0
#         y1 = y0
    
#     else:
#         x1 = x1-2*r[0]*(r[0]*x1+r[1]*y1-radius**2)/(r[0]**2+r[1]**2)
#         y1 = y1-2*r[1]*(r[0]*x1+r[1]*y1-radius**2)/(r[0]**2+r[1]**2)
    

#     return x_cross,y_cross,x1,y1

                # if y_cross<0:
                #     y_cross = -sqrt(radius**2-x_cross**2)
                # else:
                #     y_cross = sqrt(radius**2-x_cross**2)s

