#
#  Zezhou Huang
#  zhuang333@wisc.edu
#  Revised by aqahtan@hbku.edu.qa
#
import sys
from baseline.setup.detectors.pFAHES.fahes import sus_disguised
import baseline.setup.detectors.pFAHES.common as common


def detect_outliers(T, sus_dis_values):
    # histogram
    hist = {col: T[col].value_counts() for col in T.columns}
    # for each column in the csv
    for k, v in hist.items():
        col_hist = hist[k]
        numeric_data = dict()
        L_Nums = 0
        for k2, v2 in col_hist.items():
            bool, val = isNumber(k2)
            if bool:
                L_Nums += 1
                numeric_data[val] = v2

        if len(col_hist) - L_Nums >= 3 or L_Nums < 10:
            continue
        mean, num_tuples, std = compute_statistical_quantities(numeric_data)
        sort_num = sorted(numeric_data)
        min_val = sort_num[0]
        max_val = sort_num[-1]
        min_dist = std
        max_score = 0.99

        for i in range(len(sort_num) - 1):
            min_dist = min(min_dist, abs(sort_num[i] - sort_num[i + 1]))
        h0 = compute_bandwidth(std, num_tuples)
        f_max = compute_max_pdf(numeric_data, min_val, max_val, h0, num_tuples)
        if min_dist <= h0:
            for k2, v2 in numeric_data.items():
                if v2 <= 1:
                    continue
                epdf = 0
                for kk in range(4):
                    h = h0 - (0.2 * kk * h0)
                    f_i = evaluate_pnt(numeric_data, k2, h, num_tuples)
                    if epdf < f_i:
                        epdf = f_i
                score = max(f_max - epdf, 0) / f_max
                if score > (1.0 - 1e-16):
                    if int(k2) == k2:
                        sus_dis = sus_disguised(k, str(int(k2)), score, v2, "OD")
                    else:
                        sus_dis = sus_disguised(k, str(k2), score, v2, "OD")
                    if sus_dis not in sus_dis_values:
                        sus_dis_values.append(sus_dis)
                    else:
                        common.add_detected_by_more_than_one_tool(sus_dis_values, sus_dis)
    return sus_dis_values


def compute_max_pdf(col_profile, min_val, max_val, h0, S):
    n = 100
    eval_step = (max_val - min_val) / n
    max_pdf = 0
    for e_pnt in seq(min_val, max_val, eval_step):
        # print(e_pnt)
        pdf = evaluate_pnt(col_profile, e_pnt, h0, S)
        # print(e_pnt, pdf, max_pdf)
        if (max_pdf < pdf):
            max_pdf = pdf
    return max_pdf;


def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n)])
    elif n == 1:
        return ([start])
    else:
        return ([])


def evaluate_pnt(col_profile, x, h, S):
    sum = 0
    for k, v in col_profile.items():
        if k == x:
            sum += kernel_func((x - k) / h) / h
        else:
            sum += kernel_func((x - k) / h) / h * v
        # print(sum)
    return sum / S


def kernel_func(x):
    if abs(x) <= 1 and abs(x) != 0:
        return 0.75 * (1 - x * x)
    else:
        return 0


def compute_bandwidth(std, S):
    if (std == 0):
        return 1
    return 2.345 * std * pow(S, -0.2)


def compute_statistical_quantities(numeric_data):
    sum = 0
    sq_sum = 0
    num_tuples = 0
    for k, v in numeric_data.items():
        sum += k * v
        sq_sum += pow(k, 2) * v
        num_tuples += v
    return sum / num_tuples, num_tuples, pow((sq_sum - (1 / num_tuples) * pow(sum, 2)) / (num_tuples - 1), 0.5)


def isNumber(s):
    s = s.replace(',', '')
    if s[0] == '%':
        s = s[1:]
    elif s[-1] == '%':
        s = s[:-1]
    try:
        float(s)
    except:
        return False, 0
    return True, float(s)
