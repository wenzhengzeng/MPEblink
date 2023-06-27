__author__ = 'Wenzheng Zeng'
#--------------------------------------------------------------------------
# Modified from YouTubeVIS API (https://github.com/youtubevos/cocoapi)


import numpy as np
import datetime
import time
from collections import defaultdict
from copy import deepcopy
from numpy import *
import pandas as pd

class MPEblinkEval:
    # Interface for evaluating on the MPEblink dataset.
    # Modified from YouTubeVIS API (https://github.com/youtubevos/cocoapi)
    
    # The usage for MPEblinkEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = YTVOSeval(cocoGt,cocoDt); # initialize YTVOSeval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # action_ap() : calculate Blink-AP, the code is inspired by ActivityNet. (func: compute_average_precision_detection, interpolated_prec_rec)
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API  
        self.cocoDt   = cocoDt              # detections COCO API  
        self.params   = {}                  # evaluation parameters
        self.evalVids = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.vidIds = sorted(cocoGt.getVidIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                for i, a in enumerate(ann['segmentations']):
                    if a:
                        rle = coco.annToRLE(ann, i)
                        ann['segmentations'][i] = rle
                l = [a for a in ann['areas'] if a]
                if len(l)==0:
                  ann['avg_area'] = 0
                else:
                  ann['avg_area'] = np.array(l).mean()
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(vidIds=p.vidIds, catIds=p.catIds))   # Each element in gts represents an gt instance
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(vidIds=p.vidIds, catIds=p.catIds))   # Each element in dts represents an predicted instance
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(vidIds=p.vidIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(vidIds=p.vidIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['video_id'], gt['category_id']].append(gt) 
        for dt in dts:
            self._dts[dt['video_id'], dt['category_id']].append(dt)
        self.evalVids = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalVids
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.vidIds = list(np.unique(p.vidIds))    
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))    
        p.maxDets = sorted(p.maxDets)   
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(vidId, catId): computeIoU(vidId, catId) \
                        for vidId in p.vidIds
                        for catId in catIds}

        evaluateVid = self.evaluateVid
        maxDet = p.maxDets[-1]


        self.evalImgs = [evaluateVid(vidId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for vidId in p.vidIds
             ]
        self._paramsEval = deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, vidId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[vidId,catId]
            dt = self._dts[vidId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[vidId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[vidId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')  
        dt = [dt[i] for i in inds]  # Sort the confidence of each prediction dt of the current class of the current video from largest to smallest according to the sorting index obtained in the previous line
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentations'] for g in gt]  
            d = [d['segmentations'] for d in dt]    
        elif p.iouType == 'bbox':
            g = [g['bboxes'] for g in gt]   # Get the bbox of the gt
            d = [d['bboxes'] for d in dt]   # Get the bbox of the dt
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        # iscrowd = [int(o['iscrowd']) for o in gt]
        #ious = maskUtils.iou(d,g,iscrowd)
        def iou_seq(d_seq, g_seq):
            i = .0
            u = .0
            for d, g in zip(d_seq, g_seq):  # for each frame
                # If neither pred nor gt of this frame is none, calculate the intersection and merge of these two frames and add them to the total intersection and merge in the video
                if d and g: 
                    i += self.compute_frame_i(d,g)  
                    u += self.compute_frame_u(d,g)
                elif not d and g:
                    u += g[2]*g[3]
                elif d and not g:   
                    u += d[2]*d[3]   # w*h，the area of the box
            if not u > .0:
                print("Mask sizes in video {} and category {} may not match!".format(vidId, catId))
            iou = i / u if u > .0 else .0
            return iou
        ious = np.zeros([len(d), len(g)])
        for i, j in np.ndindex(ious.shape):
            ious[i, j] = iou_seq(d[i], g[j])    # Calculate the spatio-temporal iou for the specified pred and gt in a video
        return ious 

    def compute_frame_i(self, d, g):
        # Calculate the intersection area of d and g in the current frame 
        left_column_max = max(d[0], g[0])
        right_column_min = min(d[0]+d[2], g[0]+g[2])
        up_row_max = max(d[1], g[1])
        down_row_min = min(d[1]+d[3], g[1]+g[3])
        # The case of two rectangles without intersecting regions
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        else:
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S_cross

    def compute_frame_u(self, d, g):
        # Calculate the concatenated area of d and g in the current frame 

        S1 = (d[2]) * (d[3])
        S2 = (g[2]) * (g[3])
        # Calculate the intersection area of d and g in the current frame 
        left_column_max = max(d[0], g[0])
        right_column_min = min(d[0] + d[2], g[0] + g[2])
        up_row_max = max(d[1], g[1])
        down_row_min = min(d[1] + d[3], g[1] + g[3])
        # two rectangles without intersecting regions
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return S1+S2
        else:
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S1+S2-S_cross


    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['avg_area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateVid(self, vidId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[vidId,catId] # Get get gt of the current category in current video
            dt = self._dts[vidId,catId] # Get get dt of the current category in current video
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[vidId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[vidId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            # if g['ignore'] or (g['avg_area']<aRng[0] or g['avg_area']>aRng[1]):
            #     g['_ignore'] = 1
            # if g['ignore'] or (mean(g['areas'])<aRng[0] or mean(g['areas'])>aRng[1]):
            #     g['_ignore'] = 1
            # else:
            #     g['_ignore'] = 0
            g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]   # Sort predictions by confidence level from largest to smallest
        # iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[vidId, catId][:, gtind] if len(self.ious[vidId, catId]) > 0 else self.ious[vidId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:    # Determine the matching relationship between gt and pred, which is same as temporal AP in ActivityNet
            for tind, t in enumerate(p.iouThrs):    # The IoU threshold is traversed, 0.5:0.95 for a total of 10 values
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0:  
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind] # Here iou will be updated to the current maximum iou, so after the loop, the item with the maximum iou will be matched
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']   # The 'dind' pred matches the 'm' (m=gind) gt, now record the index of the gt it matches
                    gtm[tind,m]     = d['id']   # The gt with index 'gind' is matched to pred, now record the index of its matched pred
        # set unmatched detections outside of area range to ignore
        a = np.array([d['avg_area']<aRng[0] or d['avg_area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))   
        # store results for given image and category
        return {
                'video_id':     vidId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],   
                'gtIds':        [g['id'] for g in gt],  
                'dtMatches':    dtm,    # Two-dimensional array, each row represents an iou threshold, each element in a row corresponds to each dt (after sorting), its value is the matched gt id, this id is global (the id of all samples), 0 is no match to gt
                'gtMatches':    gtm,    # Two-dimensional array, each row represents an iou threshold, each element within a row corresponds to each gt (after sorting), its value is the matched dt id, this id is also global, 0 is no match to the requested dt
                'dtScores':     [d['score'] for d in dt],   # dt confidence, ranked
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval  # some configuration, such as IoU threshold
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.vidIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.vidIds)  if i in setI]
        I0 = len(_pe.vidIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        self.blink_eval_info = []
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                ################################################
                # extract the dt and gt of the instance-level TP predictions, in preparation for the subsequent calculation of the Blink-AP
                E = [self.evalImgs[Nk + Na + i] for i in i_list]
                
                gt_match_info = np.concatenate([e['gtMatches'] for e in E], axis=1)

                for iou in range(0,10):
                    iou_type = 0.5 + iou*0.05
                    gt_collection = []
                    dt_collection = []
                    for gt_index in range(0,len(gt_match_info[iou])):
                        matched_dt_id = int(gt_match_info[iou][gt_index])
                        if matched_dt_id == 0:
                            continue
                        real_gt_id = gt_index+1
                        self.cocoGt.loadAnns(real_gt_id)
                        gt_collection.append({'gt_ID':real_gt_id, 'blinks':self.cocoGt.loadAnns(real_gt_id)[0]['blinks']})
                        dt_collection.append({'gt_ID':real_gt_id, 'blinks':self.cocoDt.loadAnns(matched_dt_id)[0]['blinks_converted']})
                    self.blink_eval_info.append({'iou':iou_type, 'areaRng':_pe.areaRng[a0], 'dt_data':dt_collection, 'gt_data':gt_collection})


                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]    # all predicitons of the current category
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E]) # topK confidence pred is taken out for each video

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds] # pred does not distinguish between video, arranged in descending order

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )   # total gt num
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )    # TP 
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )    # FP 

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)  # accumulate TP
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):  # for each iou threshold
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig  # recall
                        pr = tp / (fp+tp+np.spacing(1)) # precision
                        q  = np.zeros((R,)) # q is the PR curve, subdivided into R(101) points
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i] # Ensure that theprecision is monotonically decreasing in confidence

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]  # q is the PR curve, subdivided into 101 points
                                ss[ri] = dtScoresSorted[pi] # get the confidence level of each point corresponding to the pr curve
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall, 
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def action_ap(self):
        # calculate Blink-AP
        index=0
        for config in self.blink_eval_info:
            index +=1
            if index>1: # Now we only focus on the cases where area type is 'all', and print the Blink-AP under Instance-level IoU = 0.5
                break
            gt = np.empty(shape=[0, 3], dtype=int)
            for instance in config['gt_data']:
                for blink_event in instance['blinks']:
                    gt = np.append(gt, [[instance['gt_ID'], blink_event[0], blink_event[1]]], axis=0)
            dt = np.empty(shape=[0, 4], dtype=float)
            for instance in config['dt_data']:
                for blink_event in instance['blinks']:
                    dt = np.append(dt,[[instance['gt_ID'],blink_event[0],blink_event[1],blink_event[2]]], axis=0)
            ground_truth = pd.DataFrame({
                'video-id': gt[:, 0],
                't-start': gt[:, 1],
                't-end': gt[:, 2]
            })
            prediction = pd.DataFrame({
                'video-id': dt[:, 0],
                't-start': dt[:, 1],
                't-end': dt[:, 2],
                'score': dt[:, 3]
            })
            ap = self.compute_average_precision_detection(ground_truth, prediction,
                                                     tiou_thresholds=np.linspace(0.5, 0.95, 10))
            print(f'Instance-level IoU={config["iou"]}, area@{config["areaRng"]}, Blink-AP@0.5={ap[0]}')
            print(f'Instance-level IoU={config["iou"]}, area@{config["areaRng"]}, Blink-AP@0.75={ap[5]}')
            print(f'Instance-level IoU={config["iou"]}, area@{config["areaRng"]}, Blink-AP@0.95={ap[-1]}')
            print(f'Instance-level IoU={config["iou"]}, area@{config["areaRng"]}, Blink-AP@0.5:0.95={ap.mean()}')


    def compute_average_precision_detection(self, ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
        """Compute average precision (detection task) between ground truth and
        predictions data frames. If multiple predictions occurs for the same
        predicted segment, only the one with highest score is matches as
        true positive. This code is greatly inspired by Pascal VOC devkit.

        Parameters
        ----------
        ground_truth : df
            Data frame containing the ground truth instances.
            Required fields: ['video-id', 't-start', 't-end']
        prediction : df
            Data frame containing the prediction instances.
            Required fields: ['video-id, 't-start', 't-end', 'score']
        tiou_thresholds : 1darray, optional
            Temporal intersection over union threshold.

        Outputs
        -------
        ap : float
            Average precision score.
        """
        ap = np.zeros(len(tiou_thresholds))

        npos = float(len(ground_truth))
        lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1  # [num_tiou, num_gt] Initial values are all -1
        # Sort predictions by decreasing score order.
        sort_idx = prediction['score'].values.argsort()[::-1]
        prediction = prediction.loc[sort_idx].reset_index(drop=True)  # Sorted by confidence from highest to lowest, not distinguishing video_id

        # Initialize true positive and false positive vectors.
        tp = np.zeros((len(tiou_thresholds), len(prediction)))
        fp = np.zeros((len(tiou_thresholds), len(prediction)))

        # Adaptation to query faster
        ground_truth_gbvn = ground_truth.groupby('video-id')

        # Assigning true positive to truly grount truth instances.
        for idx, this_pred in prediction.iterrows():  # Iterate through each prediction
            # print(this_pred)
            try:
                # Check if there is at least one ground truth in the video associated.
                ground_truth_videoid = ground_truth_gbvn.get_group(
                    this_pred['video-id'])  # Get all the gt corresponding to the video_id of the current prediction
            except Exception as e:
                fp[:, idx] = 1
                continue

            this_gt = ground_truth_videoid.reset_index()
            tiou_arr = self.segment_iou(this_pred[['t-start', 't-end']].values,
                                   this_gt[['t-start', 't-end']].values)
            # We would like to retrieve the predictions with highest tiou score.
            tiou_sorted_idx = tiou_arr.argsort()[::-1]
            for tidx, tiou_thr in enumerate(tiou_thresholds):
                for jdx in tiou_sorted_idx:  # Traverse in order of tIoU from largest to smallest
                    if tiou_arr[jdx] < tiou_thr:
                        fp[tidx, idx] = 1  # If it is less than iou threshold then it is directly considered as FP
                        break
                    if lock_gt[
                        tidx, this_gt.loc[jdx]['index']] >= 0:  # If it is matched, then continue and see if the next gt with a slightly smaller tIoU can meet the threshold condition and has not been matched.
                        continue
                    # Assign as true positive after the filters above.
                    tp[tidx, idx] = 1  # Under this iou_threshold, this sample is TP
                    lock_gt[tidx, this_gt.loc[jdx]['index']] = idx  # Under this iou_threshold, lock this gt, which means it has been matched
                    break

                if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:  # If a gt meets the threshold condition but has already been matched, the prediction is still FP
                    fp[tidx, idx] = 1

        tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
        fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
        recall_cumsum = tp_cumsum / npos  # Get the recall at each confidence level

        precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)  # Get the precision at each confidence level

        for tidx in range(len(tiou_thresholds)):
            ap[tidx] = self.interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])

        return ap

    def segment_iou(self, target_segment, candidate_segments):
        """Compute the temporal intersection over union between a
        target segment and all the test segments.

        Parameters
        ----------
        target_segment : 1d array
            Temporal target segment containing [starting, ending] times.
        candidate_segments : 2d array
            Temporal candidate segments containing N x [starting, ending] times.

        Outputs
        -------
        tiou : 1d array
            Temporal intersection over union score of the N's candidate segments.
        """
        tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
        tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                         + (target_segment[1] - target_segment[0]) - segments_intersection
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        tIoU = segments_intersection.astype(float) / segments_union

        return tIoU

    def interpolated_prec_rec(self,prec, rec):
        mprec = np.hstack([[0], prec, [0]])
        mrec = np.hstack([[0], rec, [1]])
        for i in range(len(mprec) - 1)[::-1]:
            mprec[i] = max(mprec[i], mprec[i + 1])
        idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
        ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
        return ap


    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = '{:<7} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.4f}'
            titleStr = 'Inst-AP' if ap == 1 else 'Average Recall'
            typeStr = '' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])   # Weighted average
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            # stats = np.zeros((12,))
            stats = np.zeros((3,))
            stats[0] = _summarize(1)    # area=all,max_det=100，AP 0.5:0.95
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2]) # AP@0.5
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.vidIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 128 ** 2], [ 128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.vidIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
