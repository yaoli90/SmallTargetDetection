'''
This is a python implementation of the "small target detection" algorithm in
--------------------------------------------------------------------------------
Chenqiang Gao, Deyu Meng, Yi Yang, et al.,
"Infrared Patch-Image Model for Small Target Detection in a Single Image,"
Image Processing, IEEE Transactions on, vol. 22, no. 12, pp. 4996-5009, 2013.
--------------------------------------------------------------------------------
Please note that this code do NOT contain the segmentation step. If you the
locations of small targets, you have to use a segmentation algorithm.

If you have any questions, please contact:
Author: Yao Li
Email: yaoli0508@hit.edu.cn
Copyright:  Harbin Institute of Technology, College of Mathematics
--------------------------------------------------------------------------------
License: Our code is only available for non-commercial research use.
'''
import numpy as np

def pos(A):
    return A * ( A > 0 ).astype(np.float32);

'''
Input:
    D: (np.array, dtype=uint8) m x n matrix of observations/data
    Lambda: (float) weight on sparse error term in the cost function
    tol: (float) tolerance for stopping criterion, DEFAULT 1e-7
    maxIter: (int) maximum number of iterations, DEFAULT 10000
    lineSearchFlag: 1 if line search is to be done every iteration, DEFAULT 0
    continuationFlag: 1 if a continuation is to be done on the parameter mu
        DEFAULT 1
    eta: (float) (0,1) line search parameter, ignored if lineSearchFlag is 0,
        DEFAULT 0.9
    mu: relaxation parameter, ignored if continuationFlag is 1, DEFAULT 1e-3
    outputFileName: (str) Details of each iteration are dumped here, if provided
Output:
    X_k_A: estimates for the low-rank part
    X_k_E:estimates for the error part
'''
def APG_IR(D, Lambda, maxIter=1e4, tol=1e-7, lineSearchFlag=0, \
    continuationFlag=1, eta=.9, mu=1e-3, outputFileName=None):

    if outputFileName is not None:
        output_file = open(outputFileName, 'w')

    DISPLAY_EVERY = 20 ;
    DISPLAY = 0 ;

    m, n = D.shape
    t_k = 1
    t_km1 = 1
    tau_0 = 2 # square of Lipschitz constant for the RPCA problem

    X_km1_A = np.zeros(D.shape)
    X_km1_E = np.zeros(D.shape) # X^{k-1} = (A^{k-1},E^{k-1})
    X_k_A = np.zeros(D.shape)
    X_k_E = np.zeros(D.shape) # X^{k} = (A^{k},E^{k})

    U,s,V = np.linalg.svd(D, full_matrices=False)
    mu_k = s[1]
    mu_bar = .005 * s[3]
    tau_k = tau_0
    converged = 0
    numIter = 0
    NOChange_counter = 0
    pre_rank = 0
    pre_cardE = 0

    while not converged:
        Y_k_A = X_k_A + ((t_km1 - 1)/t_k)*(X_k_A-X_km1_A)
        Y_k_E = X_k_E + ((t_km1 - 1)/t_k)*(X_k_E-X_km1_E)
        G_k_A = Y_k_A - (1./tau_k)*(Y_k_A+Y_k_E-D)
        G_k_E = Y_k_E - (1./tau_k)*(Y_k_A+Y_k_E-D)

        U,s,V = np.linalg.svd(G_k_A, full_matrices=False)
        X_kp1_A = np.dot(np.dot(U, np.diag(pos(s - mu_k/tau_k))), V)
        X_kp1_E = np.sign(G_k_E)*pos(np.abs(G_k_E) - Lambda* mu_k/tau_k)

        rankA  = np.sum(s>mu_k/tau_k)
        cardE = np.sum(np.sum((np.abs(X_kp1_E)>0).astype(np.float32)))

        t_kp1 = 0.5*(1+np.sqrt(1+4*t_k*t_k))
        temp = X_kp1_A + X_kp1_E - Y_k_A - Y_k_E
        S_kp1_A = tau_k*(Y_k_A-X_kp1_A) + temp
        S_kp1_E = tau_k*(Y_k_E-X_kp1_E) + temp

        norm_S = np.linalg.norm(\
            np.concatenate((S_kp1_A, S_kp1_E), axis=1), ord='fro')
        norm_X = np.linalg.norm(\
            np.concatenate((X_kp1_A, X_kp1_E), axis=1), ord='fro')
        stoppingCriterion = norm_S / (tau_k * max(1., norm_X))
        if stoppingCriterion <= tol:
            converged = 1
        if continuationFlag:
            mu_k = max(0.9*mu_k, mu_bar)

        t_km1 = t_k
        t_k = t_kp1
        X_km1_A = X_k_A
        X_km1_E = X_k_E
        X_k_A = X_kp1_A
        X_k_E = X_kp1_E
        numIter += 1

        ########################################################################
        # The iteration process can be finished if the rank of A keeps the same
        # many times

        if pre_rank == rankA:
            NOChange_counter += 1;
            if NOChange_counter > 10 and np.abs(cardE-pre_cardE) < 20:
                converged = 1
        else:
            NOChange_counter = 0
            pre_cardE = cardE
        pre_rank = rankA
        ########################################################################
        # In practice, the APG algorithm, sometimes, cannot get a strictly
        # low-rank matrix A_hat after iteration process. Many  singular valus of
        # the obtained matrix A_hat, however, are extremely small. This can be
        # considered to be low-rank to a certain extent. Experimental results
        # show that the final recoverd backgournd image and target image are
        # good. Alternatively, we can make the rank of matrix A_hat lower using
        # the following truncation. This trick can make the APG algorithm faster
        # and the performance of our algorithm is still satisfied. Here we set
        # the truncated threshold as 0.3, while it can be adaptively set based
        # on your actual scenarios.
        if rankA > 0.3 * min(m, n):
            converged = 1
        ########################################################################
        if DISPLAY and numIter%DISPLAY_EVERY == 0:
            print('Iteration ', numIter, '  rank(A) ', rankA, ' ||E||_0 ', cardE)
        if outputFileName is not None:
            line = 'Iteration '+str(numIter)+' rank(A) '+str(rankA)+ \
                ' ||E||_0 '+str(cardE)+' Stopping Criterion '+ \
                str(stoppingCriterion)+'\n'
            output_file.write(line)
        if numIter >= maxIter:
            print('Maximum iterations reached')
            converged = 1
    return X_k_A, X_k_E
