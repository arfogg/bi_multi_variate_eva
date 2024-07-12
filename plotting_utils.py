# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:55:29 2024

@author: S J Walker, adapted by A R Fogg
"""

import numpy as np

def contour_labels(contour, xpad=None, ypad=None, sides=['left', 'right'], rtol=.1, 
                   fmt=lambda x: f'{x}', x_splits=None, y_splits=None, **text_kwargs):
    """
    Function for creating labels for contour lines at the 
    edge of the subplot.
    
    Adapted from code by Simon J Walker see
    https://github.com/08walkersj/Plotting_Tools/blob/dd8c63f1b349518ea448ee59411dba1715d5eac2/src/Plotting_Tools/Contour_labels.py    

    Parameters
    ----------
    contour : matplotlib.contour
        matplotlib contour object.
    xpad : float/int, optional
        shift the left or right labels. The default is 
        None and uses .15% of absolute max of limits.
    ypad : float/int, optional
        shift the bottom or top labels. The default is 
        None and uses .15% of absolute max of limits.
    sides : list/string, optional
        sides the labels are for. The default is 
        ['left', 'right'].
    rtol : int/float, optional
        Defines how close the line must be to the axes 
        limit for a label to be made. The default is .1, 
        which is 10% of the limit away. Passed to 
        numpy.isclose and atol=0.
    fmt : definition, optional
        format for the label string The default is 
        lambda x: f'{x}'.
    x_splits : list/str, optional
        Used to define the area where labels should be 
        found. The default is None. This is useful of 
        the contour line intersects the axes limits in 
        more than one place. Can define a string as 
        positive (only look at the postive side of the 
        x axis) or negative. Alternatively string can 
        be provided as e.g. 'x>10' or any string with 
        boolean design where only x is used. If multiple 
        sides will be done to all if x_splits is a list 
        is not provided. Use None to not activate this.
    y_splits : TYPE, optional
        Same as x_splits but for the y axis. The default 
        is None.
    **text_kwargs : dictionary
        kwargs to be passed to matplotlib.axes.text.

    Raises
    ------
    ArgumentError
        Raised in an argument is not usable.

    Returns
    -------
    list
        list of text objects. If more than one side 
        provided then will be a list of lists where 
        the first list is a list of text objects 
        corresponding to the first side provided etc.

    """
    ax = contour.axes
    if xpad is None:
        xpad= np.max(np.abs(ax.get_xlim()))*.015
    if ypad is None:
        ypad= np.max(np.abs(ax.get_ylim()))*.015
    try:
        if isinstance(sides, (str, np.str)):
            sides= [sides]
    except AttributeError:
        if isinstance(sides, (str)):
            sides= [sides]
    if x_splits is None or isinstance(x_splits, (str, np.str)):
        x_splits= [x_splits]*len(sides)
    if y_splits is None or isinstance(y_splits, (str, np.str)):
        y_splits= [y_splits]*len(sides)
    labels= [[] for i in range(len(sides))]
    for collection, level in zip(contour.collections, contour.levels):
        if not len(collection.get_paths()):
            continue
        for i, (side, x_split, y_split) in enumerate(zip(sides, x_splits, y_splits)):
            x, y= np.concatenate([path.vertices for path in collection.get_paths()], axis=0).T
            if side=='left':
                if y_split=='negative':
                    x= x[y<0]
                    y= y[y<0]
                elif y_split=='positive':
                    x= x[y>0]
                    y= y[y>0]
                elif not y_split is None and ('<' in y_split or '>' in y_split):
                    x= x[eval(y_split)]
                    y= y[eval(y_split)]
                # elif not y_split is None:
                #     raise ArgumentError(f"y_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {y_split}")
                if not np.any(np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)):
                    continue
                y= y[np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)]
                x= x[np.isclose(x, ax.get_xlim()[0], atol=0, rtol=rtol)]
                y= y[np.argmin(abs(x-ax.get_xlim()[0]))]
                x= x[np.argmin(abs(x-ax.get_xlim()[0]))]
                x= ax.get_xlim()[0]
                Xpad=xpad*-1
                Ypad=0
            elif side=='right':
                if y_split=='negative':
                    x= x[y<0]
                    y= y[y<0]
                elif y_split=='positive':
                    x= x[y>0]
                    y= y[y>0]
                elif not y_split is None and ('<' in y_split or '>' in y_split):
                    x= x[eval(y_split)]
                    y= y[eval(y_split)]
                # elif not y_split is None:
                #     raise ArgumentError(f"y_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {y_split}")
                if not np.any(np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)):
                    continue
                y= y[np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)]
                x= x[np.isclose(x, ax.get_xlim()[1], atol=0, rtol=rtol)]
                y= y[np.argmin(abs(x-ax.get_xlim()[1]))]
                x= x[np.argmin(abs(x-ax.get_xlim()[1]))]
                x= ax.get_xlim()[1]
                Xpad=xpad
                Ypad=0
            elif side=='bottom':
                if x_split=='negative':
                    y= y[x<0]
                    x= x[x<0]
                elif x_split=='positive':
                    y= y[x>0]
                    x= x[x>0]
                elif not x_split is None and ('<' in x_split or '>' in x_split):
                    y= y[eval(x_split)]
                    x= x[eval(x_split)]
                # elif not x_split is None:
                #     raise ArgumentError(f"x_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {x_split}")
                if not np.any(np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)):
                    continue
                x= x[np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)]
                y= y[np.isclose(y, ax.get_ylim()[0], atol=0, rtol=rtol)]
                x= x[np.argmin(abs(y-ax.get_ylim()[0]))]
                y= y[np.argmin(abs(y-ax.get_ylim()[0]))]
                y= ax.get_ylim()[0]
                Xpad=0
                Ypad=ypad*-1
            elif side=='top':
                if x_split=='negative':
                    y= y[x<0]
                    x= x[x<0]
                elif x_split=='positive':
                    y= y[x>0]
                    x= x[x>0]
                elif not x_split is None and ('<' in x_split or '>' in x_split):
                    y= y[eval(x_split)]
                    x= x[eval(x_split)]
                # elif not x_split is None:
                #     raise ArgumentError(f"x_split not understood! must be either or a combination of: 'negative', 'positive' or None. Where they align with side.\n you chose {x_split}")
                if not np.any(np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)):
                    continue
                x= x[np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)]
                y= y[np.isclose(y, ax.get_ylim()[1], atol=0, rtol=rtol)]
                x= x[np.argmin(abs(y-ax.get_ylim()[1]))]
                y= y[np.argmin(abs(y-ax.get_ylim()[1]))]
                y= ax.get_ylim()[1]
                Xpad=0
                Ypad=ypad
            # else:
                # raise ArgumentError(f"Invalid choice for side. Please choose either or any combination of: 'left', 'right', 'top' or 'bottom'\n you chose: {side}")
            labels[i].append(ax.text(x+Xpad, y+Ypad, fmt(level), **text_kwargs))
    if len(sides)==1:
        return labels[0]
    return labels
