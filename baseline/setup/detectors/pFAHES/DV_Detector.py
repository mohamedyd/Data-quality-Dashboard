#
#  Zezhou Huang
#  zhuang333@wisc.edu
# 
import sys
from baseline.setup.detectors.pFAHES.fahes import sus_disguised


def check_non_conforming_patterns(T, sus_dis_values):
    sus_dis_values = detect_single_char_strings(T, sus_dis_values)
    sus_dis_values = positive_negative_inconsistency(T, sus_dis_values)
    sus_dis_values = check_repeated_substrings(T, sus_dis_values)
    return sus_dis_values


def detect_single_char_strings(T, sus_dis_values):
    hist = {col: T[col].value_counts() for col in T.columns}
    for k, v in hist.items():
        col_hist = hist[k]
        L_Str = 0
        L_Nums = 0
        SusStr = []
        for k2, v2 in col_hist.items():
            bool, val = isNumber(k2)
            if bool:
                L_Nums = L_Nums + 1
            elif not isNull(k2):
                L_Str = L_Str + 1
                if v2 > 1:
                    SusStr.append([k2, v2])
            # if single nonaphnum character appears more than once
            if len(k2) == 1 and not k2.isalnum() and v2 > 1:
                sus_dis = sus_disguised(k, k2, 1.0, v2, "SYN")
                if sus_dis not in sus_dis_values:
                    sus_dis_values.append(sus_dis)

        if L_Str <= 2 and L_Str > 0 and L_Nums > 2:
            for k2, v2 in SusStr:
                sus_dis = sus_disguised(k, k2, 1.0, v2, "SYN")
                if sus_dis not in sus_dis_values:
                    sus_dis_values.append(sus_dis)
    return sus_dis_values


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


def isNull(s):
    return s == "" or s.lower() == "null"


def positive_negative_inconsistency(T, sus_dis_values):
    hist = {col: T[col].value_counts() for col in T.columns}
    for k, v in hist.items():
        col_hist = hist[k]
        num_positives = 0
        num_negatives = 0
        positives = []
        negatives = []
        number_of_diff_ele = 2
        for k2, v2 in col_hist.items():
            bool, val = isNumber(k2)
            if bool and val >= 0:
                num_positives = num_positives + 1
                if v2 > 1:
                    positives.append([k2, v2])
            elif bool and val < 0:
                num_negatives = num_negatives + 1
                if v2 > 1:
                    negatives.append([k2, v2])

        if (num_positives == 1) and (num_negatives > number_of_diff_ele):
            for k2, v2 in positives:
                sus_dis = sus_disguised(k, k2, 1.0, v2, "SYN")
                if sus_dis not in sus_dis_values:
                    sus_dis_values.append(sus_dis)
        if (num_negatives == 1) and (num_positives > number_of_diff_ele):
            for k2, v2 in negatives:
                sus_dis = sus_disguised(k, k2, 1.0, v2, "SYN")
                if sus_dis not in sus_dis_values:
                    sus_dis_values.append(sus_dis)
    return sus_dis_values


def check_repeated_substrings(T, sus_dis_values):
    hist = {col: T[col].value_counts() for col in T.columns}
    for k, v in hist.items():
        col_hist = hist[k]
        threshold = 0.1
        num_rep_substr = 0
        repeated = []
        for k2, v2 in col_hist.items():
            std_dev = check_str_repetition(k2.lower())
            if std_dev == 0:
                num_rep_substr = num_rep_substr + 1
                if v2 > 1:
                    repeated.append([k2, v2])

        if num_rep_substr > 0 and num_rep_substr < threshold * len(col_hist):
            for k2, v2 in repeated:
                sus_dis = sus_disguised(k, k2, 1.0, v2, "SYN")
                if sus_dis not in sus_dis_values:
                    sus_dis_values.append(sus_dis)

    return sus_dis_values


def check_str_repetition(s):
    if len(s) < 5:
        return 1
    for ss in s:
        if ss != s[0]:
            return 1
    return 0
