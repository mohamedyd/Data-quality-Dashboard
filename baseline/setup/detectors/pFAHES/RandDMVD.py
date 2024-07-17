#
#  Zezhou Huang
#  zhuang333@wisc.edu
#
import sys
from baseline.setup.detectors.pFAHES.fahes import sus_disguised
import baseline.setup.detectors.pFAHES.common as common


def find_disguised_values(T, sus_dis_values):
    # print(T)
    hist = {col: T[col].value_counts() for col in T.columns}
    # print(hist)
    KK = []
    # for each column in the csv
    for k, v in hist.items():
        if (len(hist[k]) >= 3):
            KK.append(k)
    if len(KK) == 1:
        return sus_dis_values
    Temp_T = T[KK].reset_index().values.tolist()
    for i in range(len(Temp_T)):
        Temp_T[i].pop(0)
    # print(Temp_T)
    Temp_hist = {col: T[col].value_counts() for col in KK}
    RandDMVD_Index_T = Table_Index_RandDMVD(Temp_T)
    for i in range(len(KK)):
        dis_value = None
        largest_DV = 0
        col_hist = hist[KK[i]]
        # print(col_hist)
        most_com = sorted(col_hist.items(), key=lambda kv: kv[1], reverse=True)
        # print(most_com)
        for k, v in most_com:
            # print(k,v)
            if v == 1:
                break
            k = k.lower()
            if k == 'null':
                continue
            corr, PT_num_rows = subtable_correlation(Temp_T, k, i, RandDMVD_Index_T)
            DV_Score = len(Temp_T) / PT_num_rows * corr
            # print(corr)
            if DV_Score > largest_DV:
                dis_value = sus_disguised(KK[i], k, DV_Score, v, "Rand")
                largest_DV = DV_Score
        if dis_value is not None:
            ratio1 = dis_value.frequency / len(Temp_T)
            ratio2 = len(Temp_hist[KK[i]]) / len(Temp_T)
            # print(ratio1)
            # print(ratio2)
            # print(dis_value.value)
            if ratio1 > 0.01 and ratio2 > 0.01 and dis_value.frequency > 5:
                if dis_value not in sus_dis_values:
                    sus_dis_values.append(dis_value)
                else:
                    common.add_detected_by_more_than_one_tool(sus_dis_values, dis_value)

    return sus_dis_values


def subtable_correlation(Temp_T, k, idx, RandDMVD_Index_T):
    if k.lower() == "" or k.lower() == "null":
        data = RandDMVD_Index_T[idx].get("NULL", 0)
    else:
        data = RandDMVD_Index_T[idx].get(k, 0)
    if data == 0:
        print(idx)
        print(k)
        print("error not find in subtable_correlation")
        sys.exit(1)
    # print(data)
    RandDMVD_Index_subT = Table_Index_RandDMVD(data)
    # print(RandDMVD_Index_subT)
    corr = 0
    for i in range(len(data)):
        corr += record_correlation(data[i], Temp_T, data, RandDMVD_Index_subT, idx, RandDMVD_Index_T)
    return corr, len(data)


def record_correlation(Vec, Temp_T, data, RandDMVD_Index_subT, idx, RandDMVD_Index_T):
    n = len(Temp_T[0])
    for i in range(n):
        s = Vec[i].lower()
        if s == 'null' or s == '':
            return 0
    n_V_t = compute_num_cooccur(Vec, i, RandDMVD_Index_T)
    n_vi_t = []
    # print(Vec,idx)
    for i in range(n):
        if i != idx:
            n_vi_t.append(compute_num_occur2(Vec[i], i, RandDMVD_Index_T))
        else:
            n_vi_t.append(0)

    n_V_pt = compute_num_cooccur(Vec, idx, RandDMVD_Index_subT);
    n_vi_pt = []
    for i in range(n):
        if i != idx:
            n_vi_pt.append(compute_num_occur2(Vec[i], i, RandDMVD_Index_subT))
        else:
            n_vi_pt.append(0)

    p_V_t = n_V_t / len(Temp_T)
    p_V_pt = n_V_pt / len(data)
    p_t = 1
    p_pt = 1

    for i in range(n):
        if i != idx:
            # print(n_vi_t[i])
            p_t *= n_vi_t[i] / len(Temp_T)
            p_pt *= n_vi_pt[i] / len(data)

    corr_t = p_V_t / p_t
    corr_pt = p_V_pt / p_pt
    corr = p_V_t / (1 + pow(abs(corr_t - corr_pt), 1))
    return corr


def compute_num_occur2(v, A, RandDMVD_Index_T):
    value = RandDMVD_Index_T[A].get(v)
    # print(v,A,value)
    return len(value)


def compute_num_cooccur(Vec, idx, RandDMVD_Index_T):
    # print(Vec)
    index_min_subT = 0
    min_subT = 0
    first_time = True
    num_coocur = 0
    for i in range(len(Vec)):
        if i != idx:
            value = RandDMVD_Index_T[i].get(Vec[i])
            if value == 0:
                print("error not find in compute_num_cooccur")
                sys.exit(1)
            if first_time:
                min_subT = len(value)
                index_min_subT = i
                first_time = False
            else:
                if min_subT > len(value):
                    min_subT = len(value)
                    index_min_subT = i
    value = RandDMVD_Index_T[index_min_subT].get(Vec[index_min_subT])
    for K in range(len(value)):
        if equals(value[K], Vec, idx):
            num_coocur += 1
    return num_coocur


def equals(V1, V2, idx):
    if len(V1) != len(V2):
        return False
    for i in range(len(V1)):
        if i == idx:
            continue;
        if V1[i] != V2[i]:
            return False;
    return True;


def Table_Index_RandDMVD(Temp_T):
    number_of_rows = len(Temp_T)
    number_of_cols = len(Temp_T[0])
    m_tablehist = []
    for I in range(number_of_cols):
        m_tablehist.append(dict())

    for i in range(number_of_rows):
        for j in range(number_of_cols):
            row = []
            for ii in range(number_of_cols):
                row.append(str(Temp_T[i][ii]).lower())
            SS = str(Temp_T[i][j]).lower()
            if SS == "" or SS == "null":
                value = m_tablehist[j].get("NULL", 0)
                if value == 0:
                    m_tablehist[j]["NULL"] = [row]
                else:
                    m_tablehist[j]["NULL"].append(row)
            else:
                value = m_tablehist[j].get(SS, 0)
                if value == 0:
                    m_tablehist[j][SS] = [row]
                else:
                    m_tablehist[j][SS].append(row)

    return m_tablehist
