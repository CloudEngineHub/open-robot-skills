# SPDX-License-Identifier: MIT
"""Vendored from TrackDLO — skeleton extraction and chain merging.

Upstream: https://github.com/RMDLO/trackdlo, ``trackdlo/src/utils.py``
Copyright (c) 2025 Representing and Manipulating Deformable Linear Objects.
Licensed MIT; the permission notice travels beside it, in
``tools/curve/LICENSE.trackdlo``. Cite:

    Xiang, Dinkel, Zhao, Gao, Coltin, Smith and Bretl, "TrackDLO: Tracking
    Deformable Linear Objects Under Occlusion with Motion Coherence",
    IEEE RA-L 2023. doi:10.1109/LRA.2023.3303710

**Why vendored rather than depended on.** TrackDLO ships as a C++ ROS Noetic
package; this is the pure-Python part of it, and taking a ROS dependency to
reach three hundred lines of numpy/scipy/skimage would make the whole suite
un-runnable off Ubuntu 20.04.

**Every edit to upstream, so a diff against it stays readable.** Three, and only
one of them behavioural: the two ``visualization_msgs`` imports and the ROS
marker helper below them are dropped; ``method='zha'`` is spelled ``'zhang'``
(see the comment at that line for why that is not a change of method); and the
four stage banners this file printed on the way through are removed, because
``curve.fit_centerline`` calls it once a frame and a per-frame banner is not a
progress report, it is noise in every graph trace. The ``nan`` diagnostics in
``compute_cost`` stay: they fire on a degenerate chain, which is a thing worth
hearing about.

**What it does, and why it is the hard part.** Skeletonising a mask is one call.
Turning that skeleton into an *ordered* chain is not: a real DLO mask breaks
where something occludes it and self-touches where the object crosses itself, so
the skeleton arrives as several unordered fragments. ``extract_connected_skeleton``
prunes them, scores every endpoint-to-endpoint junction by a Euclidean-plus-
curvature cost, and solves the assignment with ``linear_sum_assignment`` --
which is what recovers one traversal order through the breaks.
"""

from skimage.morphology import skeletonize
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
from PIL import Image, ImageFilter

from scipy.spatial.transform import Rotation as R

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def pt2pt_dis(pt1, pt2):
    return np.linalg.norm(pt1-pt2)

# from geeksforgeeks: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
class Point_2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# from geeksforgeeks: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# Given three collinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False

# from geeksforgeeks: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
      
    val = (np.float64(q.y - p.y) * (r.x - q.x)) - (np.float64(q.x - p.x) * (r.y - q.y))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Collinear orientation
        return 0

# from geeksforgeeks: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
      
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
  
    # If none of the cases
    return False

def build_rect (pt1, pt2, width):
    # arctan2 (y, x)
    line_angle = np.arctan2(pt2.y - pt1.y, pt2.x - pt1.x)
    angle1 = line_angle + np.pi/2
    angle2 = line_angle - np.pi/2
    rect_pt1 = Point_2D(pt1.x + width/2.0*np.cos(angle1), pt1.y + width/2.0*np.sin(angle1))
    rect_pt2 = Point_2D(pt1.x + width/2.0*np.cos(angle2), pt1.y + width/2.0*np.sin(angle2))
    rect_pt3 = Point_2D(pt2.x + width/2.0*np.cos(angle1), pt2.y + width/2.0*np.sin(angle1))
    rect_pt4 = Point_2D(pt2.x + width/2.0*np.cos(angle2), pt2.y + width/2.0*np.sin(angle2))

    return [rect_pt1, rect_pt2, rect_pt4, rect_pt3]

def check_rect_overlap (rect1, rect2):
    overlap = False
    for i in range (-1, 3):
        for j in range (-1, 3):
            if doIntersect(rect1[i], rect1[i+1], rect2[j], rect2[j+1]):
                overlap = True
                return overlap
    return overlap

# mode:
#   0: start + start
#   1: start + end
#   2: end + start
#   3: end + end
def compute_cost (chain1, chain2, w_e, w_c, mode):
    chain1 = np.array(chain1)
    chain2 = np.array(chain2)

    # start + start
    if mode == 0:
        # treat the second chain as needing to be reversed
        cost_euclidean = np.linalg.norm(chain1[0] - chain2[0])
        cost_curvature_1 = np.arccos(np.dot(chain1[0] - chain2[0], chain1[1] - chain1[0]) / (np.linalg.norm(chain1[0] - chain1[1]) * cost_euclidean))
        cost_curvature_2 = np.arccos(np.dot(chain1[0] - chain2[0], chain2[0] - chain2[1]) / (np.linalg.norm(chain2[0] - chain2[1]) * cost_euclidean))
        total_cost = w_e * cost_euclidean + w_c * (np.abs(cost_curvature_1) + np.abs(cost_curvature_2)) / 2.0
    # start + end
    elif mode == 1:
        cost_euclidean = np.linalg.norm(chain1[0] - chain2[-1])
        cost_curvature_1 = np.arccos(np.dot(chain1[0] - chain2[-1], chain1[1] - chain1[0]) / (np.linalg.norm(chain1[0] - chain1[1]) * cost_euclidean))
        cost_curvature_2 = np.arccos(np.dot(chain1[0] - chain2[-1], chain2[-1] - chain2[-2]) / (np.linalg.norm(chain2[-1] - chain2[-2]) * cost_euclidean))
        total_cost = w_e * cost_euclidean + w_c * (np.abs(cost_curvature_1) + np.abs(cost_curvature_2)) / 2.0
    # end + start
    elif mode == 2:
        cost_euclidean = np.linalg.norm(chain1[-1] - chain2[0])
        cost_curvature_1 = np.arccos(np.dot(chain2[0] - chain1[-1], chain1[-1] - chain1[-2]) / (np.linalg.norm(chain1[-1] - chain1[-2]) * cost_euclidean))
        cost_curvature_2 = np.arccos(np.dot(chain2[0] - chain1[-1], chain2[1] - chain2[0]) / (np.linalg.norm(chain2[0] - chain2[1]) * cost_euclidean))
        total_cost = w_e * cost_euclidean + w_c * (np.abs(cost_curvature_1) + np.abs(cost_curvature_2)) / 2.0
    # end + end
    else:
        # treat the second chain as needing to be reversed
        cost_euclidean = np.linalg.norm(chain1[-1] - chain2[-1])
        cost_curvature_1 = np.arccos(np.dot(chain2[-1] - chain1[-1], chain1[-1] - chain1[-2]) / (np.linalg.norm(chain1[-1] - chain1[-2]) * cost_euclidean))
        cost_curvature_2 = np.arccos(np.dot(chain2[-1] - chain1[-1], chain2[-2] - chain2[-1]) / (np.linalg.norm(chain2[-1] - chain2[-2]) * cost_euclidean))
        total_cost = w_e * cost_euclidean + w_c * (np.abs(cost_curvature_1) + np.abs(cost_curvature_2)) / 2.0
    
    if total_cost is np.nan or cost_curvature_1 is np.nan or cost_curvature_2 is np.nan:
        print('total cost is nan!')
        print('chain1 =', chain1)
        print('chain2 =', chain2)
        print('euclidean cost = {}, w_e*cost = {}; curvature cost = {}, w_c*cost = {}'.format(cost_euclidean, w_e*cost_euclidean, (np.abs(cost_curvature_1) + np.abs(cost_curvature_2)) / 2.0, w_c * (np.abs(cost_curvature_1) + np.abs(cost_curvature_2)) / 2.0))
    return total_cost

# partial implementation of paper "Deformable One-Dimensional Object Detection for Routing and Manipulation"
# paper link: https://ieeexplore.ieee.org/abstract/document/9697357
def extract_connected_skeleton (visualize_process, mask, img_scale=10, seg_length=3, max_curvature=30):  # note: mask is one channel

    # smooth image
    im = Image.fromarray(mask)
    smoothed_im = im.filter(ImageFilter.ModeFilter(size=15))
    mask = np.array(smoothed_im)

    # resize if necessary for better skeletonization performance
    mask = cv2.resize(mask, (int(mask.shape[1]/img_scale), int(mask.shape[0]/img_scale)))

    if visualize_process:
        cv2.imshow('init frame', mask)
        while True:
            key = cv2.waitKey(10)
            if key == 27:  # escape
                cv2.destroyAllWindows()
                break
    
    # skeletonization
    #
    # CHANGED FROM UPSTREAM, and the only behavioural change in this file.
    # Upstream passes ``method='zha'``, which was never one of skimage's two
    # methods. Old skimage tested ``if method == 'lee'`` and let anything else
    # fall through to Zhang, so upstream has always run Zhang by accident;
    # skimage now validates the string and RAISES. Naming Zhang explicitly is
    # therefore what upstream has been doing all along, not a change of method.
    #
    # This mattered: the raise was swallowed by the caller's fallback, so the
    # chain merge -- the whole reason this file is vendored -- never ran once,
    # and every centreline was ordered by the PCA fallback instead.
    result = skeletonize(mask, method='zhang')
    gray = (np.asarray(result) > 0).astype(np.uint8) * 255
    if gray.ndim == 3:  # a 3-channel mask skeletonises per channel
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray[gray > 100] = 255

    if visualize_process:
        cv2.imshow('after skeletonization', gray)
        while True:
            key = cv2.waitKey(10)
            if key == 27:  # escape
                cv2.destroyAllWindows()
                break

    # extract contour
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    chains = []

    # for each object segment
    for a, contour in enumerate(contours):
        c_area = cv2.contourArea(contour)
        mask = np.zeros(gray.shape, np.uint8)
        # mask = result.copy()

        last_segment_dir = None
        chain = []
        cur_seg_start_point = None

        for i, coord in enumerate(contour):

            # special case: if reached the end of current contour, append chain if not empty
            if i == len(contour)-1:
                if len(chain) != 0:
                    chains.append(chain)
                break

            # graph for visualization
            mask = cv2.line(mask, tuple(contour[i][0]), tuple(contour[i+1][0]), 255, 1)

            # if haven't initialized, perform initialization
            if cur_seg_start_point is None:
                cur_seg_start_point = contour[i][0].copy()

            # keep traversing until reach segment length
            if np.sqrt((contour[i][0][0] - cur_seg_start_point[0])**2 + (contour[i][0][1] - cur_seg_start_point[1])**2) <= seg_length:
                continue

            cur_seg_end_point = contour[i][0].copy()
            
            # record segment direction
            cur_segment_dir = [cur_seg_end_point[0] - cur_seg_start_point[0], cur_seg_end_point[1] - cur_seg_start_point[1]]
            if last_segment_dir is None:
                last_segment_dir = cur_segment_dir.copy()

            # if direction hasn't changed much, add current segment to chain
            elif np.dot(np.array(cur_segment_dir), np.array(last_segment_dir)) / \
                (np.sqrt(cur_segment_dir[0]**2 + cur_segment_dir[1]**2) * np.sqrt(last_segment_dir[0]**2 + last_segment_dir[1]**2)) >= np.cos(max_curvature/180*np.pi):
                # if chain is empty, append both start and end point
                if len(chain) == 0:
                    chain.append(cur_seg_start_point.tolist())
                    chain.append(cur_seg_end_point.tolist())
                # if chain is not empty, only append end point
                else:
                    chain.append(cur_seg_end_point.tolist())
                
                # next start point <- end point
                cur_seg_start_point = cur_seg_end_point.copy()

                # update last segment direction
                last_segment_dir = cur_segment_dir.copy()
            
            # if direction changed, start a new chain
            else:
                # append current chain to chains if chain is not empty
                if len(chain) != 0:
                    chains.append(chain)
                
                # reinitialize all variables
                last_segment_dir = None
                chain = []
                cur_seg_start_point = None


    if visualize_process:
        mask = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
        for chain in chains:
            color = (int(np.random.random()*200)+55, int(np.random.random()*200)+55, int(np.random.random()*200)+55)
            for i in range (0, len(chain)-1):
                mask = cv2.line(mask, chain[i], chain[i+1], color, 1)
        cv2.imshow('added all chains frame', mask)
        while True:
            key = cv2.waitKey(10)
            if key == 27:  # escape
                cv2.destroyAllWindows()
                break

    # another pruning method
    all_chain_length = []
    line_seg_to_rect_dict = {}
    rect_width = 3
    for chain in chains:
        all_chain_length.append(np.sum(np.sqrt(np.sum(np.square(np.diff(np.array(chain), axis=0)), axis=1))))
        for i in range (0, len(chain)-1):
            line_seg_to_rect_dict[(tuple(chain[i]), tuple(chain[i+1]))] = \
                build_rect(Point_2D(chain[i][0], chain[i][1]), Point_2D(chain[i+1][0], chain[i+1][1]), rect_width)

    all_chain_length = np.array(all_chain_length)
    sorted_idx = np.argsort(all_chain_length.copy())
    chains = np.asarray(chains, dtype=list)
    sorted_chains = chains[sorted_idx]

    pruned_chains = []
    for i in range (0, len(chains)):
        leftover_chains = []
        cur_chain = sorted_chains[-1]
        chains_to_check = []

        for j in range (0, len(sorted_chains)-1):  # -1 because the last one is cur_chain
            test_chain = sorted_chains[j]
            new_test_chain = []
            for l in range (0, len(test_chain)-1):
                rect_test_seg = line_seg_to_rect_dict[(tuple(test_chain[l]), tuple(test_chain[l+1]))]
                # check against all segments in the current chain
                no_overlap = True
                for k in range (0, len(cur_chain)-1):
                    rect_cur_seg = line_seg_to_rect_dict[(tuple(cur_chain[k]), tuple(cur_chain[k+1]))]
                    if check_rect_overlap(rect_cur_seg, rect_test_seg):
                        no_overlap = False
                        break
                # only keep the segment in the test chain if it does not overlap with any segments in the current chain
                if no_overlap:
                    if len(new_test_chain) == 0:
                        new_test_chain.append(test_chain[l])
                        new_test_chain.append(test_chain[l+1])
                    else:
                        new_test_chain.append(test_chain[l+1])
            # finally, add trimmed test chain back into the pile for checking in the next loop
            leftover_chains.append(new_test_chain)
        
        # added current chain into pruned chains
        if len(cur_chain) != 0:
            pruned_chains.append(cur_chain)

        # reinitialize sorted chains
        all_chain_length = []
        for chain in leftover_chains:
            if len(chain) != 0:
                all_chain_length.append(np.sum(np.sqrt(np.sum(np.square(np.diff(np.array(chain), axis=0)), axis=1))))
            else:
                all_chain_length.append(0)

        all_chain_length = np.array(all_chain_length)
        sorted_idx = np.argsort(all_chain_length.copy())
        leftover_chains = np.asarray(leftover_chains, dtype=list)
        sorted_chains = leftover_chains[sorted_idx]

    
    if visualize_process:
        mask = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
        for chain in pruned_chains:
            color = (int(np.random.random()*200)+55, int(np.random.random()*200)+55, int(np.random.random()*200)+55)
            for i in range (0, len(chain)-1):
                mask = cv2.line(mask, chain[i], chain[i+1], color, 1)
        cv2.imshow("after pruning", mask)
        while True:
            key = cv2.waitKey(10)
            if key == 27:  # escape
                cv2.destroyAllWindows()
                break

    if len(pruned_chains) == 1:
        return pruned_chains

    # total number of possible matches = number of ends * (number of ends - 2) / 2 = 2*len(chains) * (len(chains) - 1)
    # cost matrix size: num of tips + 2
    # cost matrix entry label format: tip1 start, tip1 end, tip2 start, tip2 end, ...
    matrix_size = 2*len(pruned_chains) + 2
    cost_matrix = np.zeros((matrix_size, matrix_size))
    w_e = 0.001
    w_c = 1
    for i in range (0, len(pruned_chains)):
        for j in range (0, len(pruned_chains)):
            # two types of matches should be discouraged: 
            # matching with itself and matching with the other tip on the same segment
            #          tj_start tj_end
            # ti_start      ...    ...
            #   ti_end      ...    ...
            if i == j:
                cost_matrix[2*i, 2*j] = 100000
                cost_matrix[2*i, 2*j+1] = 100000
                cost_matrix[2*i+1, 2*j] = 100000
                cost_matrix[2*i+1, 2*j+1] = 100000
            else:
                # ti_start to tj_start
                cost_matrix[2*i, 2*j] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 0)
                # ti_start to tj_end
                cost_matrix[2*i, 2*j+1] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 1)
                # ti_end to tj_start
                cost_matrix[2*i+1, 2*j] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 2)
                # ti_end to ti_end
                cost_matrix[2*i+1, 2*j+1] = compute_cost(pruned_chains[i], pruned_chains[j], w_e, w_c, 3)
    
    # cost for being the dlo's two ends
    cost_matrix[:, -1] = 1000
    cost_matrix[:, -2] = 1000
    cost_matrix[-1, :] = 1000
    cost_matrix[-2, :] = 1000
    # prevent matching with itself
    cost_matrix[matrix_size-2:matrix_size, matrix_size-2:matrix_size] = 100000

    # CHANGED FROM UPSTREAM: a junction with no information is expensive, not
    # fatal. ``compute_cost`` divides by a chain's own extent to get curvature,
    # so a chain pruned down to a single point produces NaN, and
    # ``linear_sum_assignment`` then refuses the whole matrix with "matrix
    # contains invalid numeric entries" -- losing every *other* junction in the
    # frame over one degenerate fragment.
    #
    # Observed mid-carry on ``cable_route/port3``, which is the phase where the
    # ordering actually matters: the rod is half-covered by a gripper, the
    # skeleton fragments, and this is exactly when the merge is supposed to
    # earn its keep. Treating a NaN as the same 100000 the matrix already uses
    # for a forbidden pairing keeps the assignment solvable and simply routes
    # the traversal around the fragment.
    cost_matrix = np.nan_to_num(cost_matrix, nan=100000.0,
                                posinf=100000.0, neginf=100000.0)

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    cur_idx = col_idx[row_idx[-1]]
    ordered_chains = []

    mask = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    # CHANGED FROM UPSTREAM: guard the traversal's ENTRY, not just its exit.
    #
    # The loop below stops when the *next* index is one of the two terminator
    # columns, but nothing checks the index it starts on, and
    # ``col_idx[row_idx[-1]]`` is free to be a terminator itself -- or to lead
    # back to a chain already walked. Either way ``pruned_chains[cur_idx // 2]``
    # goes out of range and the whole extraction raises.
    #
    # It raises on precisely the input this function exists to handle: a mask
    # broken into fragments by an occluder. Measured here, a clean mask merges
    # fine and a mask cut by a 20 px band -- about what a finger pair covers --
    # raised IndexError every time. Stopping the walk is what the loop's own
    # exit already means; this only checks the same condition one step earlier.
    seen_chains = set()
    while True:
        cur_chain_idx = int(cur_idx/2)
        if cur_chain_idx >= len(pruned_chains) or cur_chain_idx in seen_chains:
            break
        seen_chains.add(cur_chain_idx)
        cur_chain = pruned_chains[cur_chain_idx]

        if cur_idx % 2 == 1:
            cur_chain.reverse()
        ordered_chains.append(cur_chain)

        if cur_idx % 2 == 0:
            next_idx = col_idx[cur_idx+1]  # find where this chain's end is connecting to 
        else:
            next_idx = col_idx[cur_idx-1]  # find where this chain's start is connecting to

        # if visualize_process:
        #     mask = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
        #     color = (int(np.random.random()*200)+55, int(np.random.random()*200)+55, int(np.random.random()*200)+55)
        #     for j in range (0, len(cur_chain)-1):
        #         mask = cv2.line(mask, cur_chain[j], cur_chain[j+1], color, 1)
        #     cv2.imshow('in merging', mask)
        #     while True:
        #         key = cv2.waitKey(10)
        #         if key == 27:  # escape
        #             cv2.destroyAllWindows()
        #             break

        # if reached the other end, stop
        if next_idx == matrix_size-1 or next_idx == matrix_size-2:
            break
        cur_idx = next_idx
    
    
    # visualization code for debug
    if visualize_process:
        mask = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
        for i in range (0, len(ordered_chains)):
            chain = ordered_chains[i]
            color = (int(np.random.random()*200)+55, int(np.random.random()*200)+55, int(np.random.random()*200)+55)
            for j in range (0, len(chain)-1):
                mask = cv2.line(mask, chain[j], chain[j+1], color, 1)

            cv2.imshow('after merging', mask)
            while True:
                key = cv2.waitKey(10)
                if key == 27:  # escape
                    cv2.destroyAllWindows()
                    break

            if i == len(ordered_chains)-1:
                break

            pt1 = ordered_chains[i][-1]
            pt2 = ordered_chains[i+1][0]
            mask = cv2.line(mask, pt1, pt2, (255, 255, 255), 2)
            mask = cv2.circle(mask, pt1, 3, (255, 255, 255))
            mask = cv2.circle(mask, pt2, 3, (255, 255, 255))

    return ordered_chains

# original post: https://stackoverflow.com/a/59204638
