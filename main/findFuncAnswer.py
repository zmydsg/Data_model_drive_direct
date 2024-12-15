import sympy
import math
import numpy as np
import time
from sympy import Eq
from args import args
from utils import Probabilty_outage ,through_output


def calPreFactor(factor, NumK, rate_2):
    """:cvar
    求解前置因子
    """
    f = lambda x: f(x - 1) * x if x >= 2 else 1
    f1 = lambda x, m: x ** (2 * m)
    prefactor = np.zeros([NumK], dtype= float)

    for i in range(1, NumK + 1):  # 1， 2， 3，..., K

        # A : GK(x)
        a0 = 0
        for j in range(i):  # 0, 1, 2, ..., K-1
            a1 = f(i - j - 1)
            a2 = sympy.log(rate_2)
            a0 += ((-1) ** j) * (a2 ** (i - j - 1)) / a1
        a3 = (-1) ** i + rate_2 * a0

        # B : L(factor ,K)
        a4 = 1
        a5 = 1
        for k in range(1, i + 1):
            a6 = f1(factor, k)
            a4 += a6 / (1 - a6)
            a5 *= (1 - a6)

        a7 = 1 / (a4 * a5)

        prefactor[i - 1] = a3*a7
    # print("prefactor:",prefactor)
    return prefactor


# 方案一 适用sympy库
def solvePt(prefactor, pt_max, factor, NumK, ):
    """
    求解 等式方程 解集可能含有不符合约束的答案
    :param prefactor:
    :param pt_max:
    :param factor:
    :param NumK:
    :return:
    """
    p = sympy.Symbol('p', real = True, positive = True) # 定义未知变量 p
    sympy.init_printing(use_unicode=True)
    # such as :fx =x**1+x**2+x**3+x**4-pt_max
    fx = p
    for i in range(1,NumK):
        fx += prefactor[i-1]*(pow(p,1-i))

    solution = sympy.solveset(Eq(fx,pt_max), domain=sympy.S.Reals)
    # print("equation:\n", fx)
    # print(f"solution:{solution})")
    return list(solution)


#方案2 适用scipy库
# def solvePt(prefactor, pt_max, factor, NumK, ):
#     def func(p,para):
#         fx = - pt_max
#         for i in range(1,NumK):
#             fx += para[i-1]*(p**(1-i))
#         return fx
#
#     solution = optimize.fsolve(func,[0],args=prefactor)
#     return solution


def equalAllocation(Pt_max, factor, NumK, rate, bounds, func):
    """
    等功率分配算法求解
    :param Pt_max:
    :param factor:
    :param NumK:
    :param rate:
    :param bounds:
    :return:
    """
    startime = time.time()
    rate_2 = 2**rate
    prefactor = calPreFactor(factor,NumK,rate_2)
    result_through_output = 0
    solutions = np.array(solvePt(prefactor, Pt_max, factor,NumK))
    if solutions is None:
        print("out of set")
        return 
    for item in solutions:
        pt = np.ones([1,NumK])*item
        poutList = Probabilty_outage(func,pt,[factor],rate,NumK,1,flag=True)

        # print("poutList",poutList,type(poutList))
        if(poutList[0,-1])>bounds:

            # print(f"\nsolution {item} is  out of  constrain:{poutList}")
            continue
        else:
            # print(f"\nsolution:{item}\tpt:{pt}")
            result_through_output = through_output(poutList, rate, NumK, flag=True)
            # result_through_output=np.squeeze(through_output(poutList,rate,NumK,flag=True)) # .astype(np.float32)
    endtime = time.time()
    if result_through_output != 0:
        print(f"pt:{pt}")
    # print(f"consume time/s:{endtime-startime}\nthrough_output:{result_through_output}")
        return result_through_output[-1,-1],poutList[0,-1]
    else:
        return None,None

def get_Equal_solution(factor, NumK, PDBs, func):
    """
    求解不同DB下对应的 吞吐量 及 中断概率
    :return:
    """
    Bounds = args.Bounds
    rate = args.rate
    # Bounds = {10: 10e-2, 15: 10e-4, 20: 10e-6, 25: 10e-8, 30:10e-10} unequal bound
    # Bounds = {10: 4e-5, 15: 4e-5, 20: 4e-5, 25: 4e-5, 30: 4e-5} # equal bound

    throught_list2, outage_list2 = [], []
    for pdb in PDBs:
        pt_max = 10**(pdb/10)
        result = equalAllocation(pt_max, [factor], NumK, rate,1, bounds=Bounds[pdb],func=func)
        throught_list2.append(result[0])
        outage_list2.append(result[1])

    return throught_list2,outage_list2

if __name__ == "__main__":
    print("findFuncAnswer doing")
    pdbs = args.PDBs
    factor = args.factor
    rate = args.rate
    Numk = args.NumK
    bounds = args.Bounds
    for pdb in pdbs:
        print(f"pdb:{pdb}dB")
        pt_max = 10**(pdb/10)
        bound = bounds[pdb]
        # for i in range(3,9):
        print(equalAllocation(pt_max, factor, Numk, rate, bound))
        print("="*50, "\n")


