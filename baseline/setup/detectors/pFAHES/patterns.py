#
#  Zezhou Huang
#  zhuang333@wisc.edu
#
import sys
from baseline.setup.detectors.pFAHES.fahes import sus_disguised


def find_all_patterns(T, sus_dis_values):
    # print(list(T))
    # histogram
    hist = {col: T[col].value_counts() for col in T.columns}
    # print(hist)
    min_num_ptrns = 5
    # for each column in the csv
    output_ptrns = dict()
    output_ptrns.clear()
    for k, v in hist.items():
        # print(k)
        # print(v)
        col_hist = hist[k]
        # ptrns_vec contains pattern as a directory
        # pttrns_hist contains pattern string : frequency
        pttrns, pttrns_hist = L1_patterns(col_hist)
        # print(pttrns)
        # print(pttrns_hist)
        AGG_Level = 1
        if len(pttrns_hist) > min_num_ptrns:
            AGG_Level = 2
            pttrns, pttrns_hist = L_patterns(pttrns, pttrns_hist, 2)
        if len(pttrns_hist) > min_num_ptrns:
            AGG_Level = 3
            pttrns, pttrns_hist = L_patterns(pttrns, pttrns_hist, 3)
        if len(pttrns_hist) > min_num_ptrns:
            AGG_Level = 4
            pttrns, pttrns_hist = L_patterns(pttrns, pttrns_hist, 4)
        if len(pttrns_hist) > min_num_ptrns:
            AGG_Level = 5
            pttrns, pttrns_hist = L_patterns(pttrns, pttrns_hist, 5)
        dominating_pttrns = determine_dominating_patterns(pttrns_hist)
        # print(dominating_pttrns)
        output_ptrns[k] = pttrns_hist
        sus_dis = []
        for k2, v2 in col_hist.items():
            # print(k2,v2)
            # common_Strings are strings have more than one frequency
            if v2 <= 1:
                continue
            test_ptrn = get_cell_pttrn(k2, AGG_Level)
            value = dominating_pttrns.get(test_ptrn, 123)
            if value == 123:
                print("Pattern not found ..\n")
                sys.exit(1)
            elif not value:
                sus_dis = sus_disguised(k, k2, 1.0, v2, "SYN")
                if sus_dis not in sus_dis_values:
                    sus_dis_values.append(sus_dis)
    return sus_dis_values, output_ptrns
    # print(T['a'].value_counts().get(1))


def get_cell_pttrn(k, AGG_Level):
    ptrn_vec = L1_pattern(k)
    if AGG_Level >= 2:
        ptrn_vec = remove_numbers(ptrn_vec)
    if AGG_Level >= 3:
        ptrn_vec = aggregate_UL_classes(ptrn_vec)
    if AGG_Level >= 4:
        ptrn_vec = aggregate_ASCHT_classes(ptrn_vec)
    if AGG_Level >= 5:
        ptrn_vec = aggregate_WD_classes(ptrn_vec)
    return PatterntoString(ptrn_vec)


def determine_dominating_patterns(pttrns_hist):
    dom_ptrns = dict()
    total_dist_vals = 0
    ptrns_by_freq = dict()
    for k, v in pttrns_hist.items():
        total_dist_vals = total_dist_vals + v
        value = ptrns_by_freq.get(v, 0)
        if value == 0:
            ptrns_by_freq[v] = [k]
        else:
            ptrns_by_freq[v].append(k)
    covered_ratio = 0.0
    dom_ratio_reached = False
    cut_off_thresh = max(0.7, min(0.97, 1 - 3.0 / total_dist_vals))
    for k, v in reversed(sorted(ptrns_by_freq.items())):
        covered_ratio = covered_ratio + (k * len(v)) / total_dist_vals
        if covered_ratio < cut_off_thresh:
            for i in range(len(v)):
                dom_ptrns[v[i]] = True
        elif not dom_ratio_reached:
            for i in range(len(v)):
                dom_ptrns[v[i]] = True
            dom_ratio_reached = True
        else:
            for i in range(len(v)):
                if k >= 4 or len(v) * k >= 4:
                    dom_ptrns[v[i]] = True
                else:
                    dom_ptrns[v[i]] = False
    return dom_ptrns


def L_patterns(pttrns, pttrns_hist, n):
    new_pttrns_hist = dict()
    new_pttrns = []
    for i in range(len(pttrns)):
        old_ptrn = pttrns[i]
        if n == 2:
            new_pttrn = remove_numbers(old_ptrn)
        elif n == 3:
            new_pttrn = aggregate_UL_classes(old_ptrn)
        elif n == 4:
            new_pttrn = aggregate_ASCHT_classes(old_ptrn)
        elif n == 5:
            new_pttrn = aggregate_WD_classes(old_ptrn)

        new_ptrn = PatterntoString(new_pttrn)
        k = new_pttrns_hist.get(new_ptrn, 0)
        if k == 0:
            new_pttrns_hist[new_ptrn] = pttrns_hist[PatterntoString(old_ptrn)]
            new_pttrns.append(new_pttrn)
        else:
            new_pttrns_hist[new_ptrn] = k + pttrns_hist[PatterntoString(old_ptrn)]

    return new_pttrns, new_pttrns_hist


def aggregate_WD_classes(old_ptrn):
    old_ptrn = apply_L4_tricks(old_ptrn)
    new_ptrn = []
    for i in range(len(old_ptrn)):
        ch = old_ptrn[i][0]
        if match_L5(WORDP, ch):
            ch = WORDP
        if len(new_ptrn) > 0:
            cur_class = new_ptrn[len(new_ptrn) - 1][0];
            if match_L5(cur_class, ch):
                new_ptrn[len(new_ptrn) - 1] = "w+"
            else:
                if len(old_ptrn[i]) > 1:
                    new_ptrn.append(ch + "+")
                else:
                    new_ptrn.append(ch)
        else:
            if len(old_ptrn[i]) > 1:
                new_ptrn.append(ch + "+")
            else:
                new_ptrn.append(ch)
    new_ptrn = aggregate_ASCHT_classes(new_ptrn)
    while True:
        new_pttrn_after = apply_L5_tricks(new_ptrn)
        if PatterntoString(new_pttrn_after) == PatterntoString(new_ptrn):
            break
        new_ptrn = new_pttrn_after
    return new_ptrn


def apply_L5_tricks(old_ptrn):
    results_ptrn1 = []
    results_ptrn2 = []
    if len(old_ptrn) >= 3:
        if old_ptrn[0][0] == DIGIT and old_ptrn[1][0] == DOT and len(old_ptrn[1][0]) == 1 and old_ptrn[1][0] == DIGIT:
            results_ptrn1.append("d+")
    if len(old_ptrn) >= 3:
        i = 0
        while i < len(old_ptrn) - 2:
            if old_ptrn[i][0] == DIGIT and old_ptrn[i + 1][0] == DASH and old_ptrn[i + 2][0] == DIGIT:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == WORDP and old_ptrn[i + 1][0] == AT and old_ptrn[i + 2][0] == WORDP:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == WORDP and old_ptrn[i + 1][0] == AND and old_ptrn[i + 2][0] == WORDP:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == SPACE and old_ptrn[i + 1][0] == AND and old_ptrn[i + 2][0] == SPACE:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == SPACE and old_ptrn[i + 1][0] == AND and old_ptrn[i + 2][0] == WORDP:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == WORDP and old_ptrn[i + 1][0] == AND and old_ptrn[i + 2][0] == SPACE:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == WORDP and old_ptrn[i + 1][0] == SLASH and old_ptrn[i + 2][0] == WORDP:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == DIGIT and old_ptrn[i + 1][0] == SLASH and old_ptrn[i + 2][0] == WORDP:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == WORDP and old_ptrn[i + 1][0] == SLASH and old_ptrn[i + 2][0] == DIGIT:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == DIGIT and old_ptrn[i + 1][0] == SLASH and old_ptrn[i + 2][0] == DIGIT:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == HASH and old_ptrn[i + 1][0] == SPACE and old_ptrn[i + 2][0] == DIGIT:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            results_ptrn1.append(old_ptrn[i])
            i = i + 1
        while i < len(old_ptrn):
            results_ptrn1.append(old_ptrn[i])
            i = i + 1
        old_ptrn = results_ptrn1
    if len(old_ptrn) >= 2:
        i = 0
        while i < len(old_ptrn) - 1:
            if old_ptrn[i][0] == HASH and old_ptrn[i + 1][0] == DIGIT:
                results_ptrn2 = check_and_push(results_ptrn2)
                i = i + 2
                continue
            results_ptrn2.append(old_ptrn[i])
            i = i + 1
        while i < len(old_ptrn):
            results_ptrn2.append(old_ptrn[i])
            i = i + 1
        old_ptrn = results_ptrn2
    if len(results_ptrn2) > 0:
        return results_ptrn2
    else:
        return old_ptrn


def match_L5(cch, ch):
    if cch == WORDP and ch == DIGIT:
        return True
    if cch == WORDP and ch == SPALPHA:
        return True
    if cch == ch:
        return True
    return False


def apply_L4_tricks(old_ptrn):
    results_ptrn1 = []
    results_ptrn2 = []
    if len(old_ptrn) >= 3:
        i = 0
        while i < len(old_ptrn) - 2:
            if old_ptrn[i][0] == HASH and old_ptrn[i + 1][0] == SPACE and old_ptrn[i + 2][0] == DIGIT:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            if old_ptrn[i][0] == HASH and old_ptrn[i + 1][0] == SPACE and old_ptrn[i + 2][0] == WORDP:
                results_ptrn1 = check_and_push(results_ptrn1)
                i = i + 3
                continue
            results_ptrn1.append(old_ptrn[i])
            i = i + 1
        while i < len(old_ptrn):
            results_ptrn1.append(old_ptrn[i])
            i = i + 1
        old_ptrn = results_ptrn1
    if len(old_ptrn) >= 2:
        i = 0
        while i < len(old_ptrn) - 1:
            if old_ptrn[i][0] == HASH and old_ptrn[i + 1][0] == DIGIT:
                results_ptrn2 = check_and_push(results_ptrn2)
                i = i + 2
                continue
            if old_ptrn[i][0] == HASH and old_ptrn[i + 1][0] == WORDP:
                results_ptrn2 = check_and_push(results_ptrn2)
                i = i + 2
                continue
            results_ptrn2.append(old_ptrn[i])
            i = i + 1
        while i < len(old_ptrn):
            results_ptrn2.append(old_ptrn[i])
            i = i + 1
    if len(results_ptrn2) > 0:
        return results_ptrn2
    else:
        return old_ptrn


def check_and_push(results_ptrn1):
    new_results_ptrn1 = results_ptrn1
    W_plus = "w+"
    if len(new_results_ptrn1) == 0:
        new_results_ptrn1.append(W_plus)
        return new_results_ptrn1
    if new_results_ptrn1[len(new_results_ptrn1) - 1][0] == 'w':
        new_results_ptrn1[len(new_results_ptrn1) - 1] = W_plus
    else:
        new_results_ptrn1.append(W_plus)
    return new_results_ptrn1


def aggregate_ASCHT_classes(old_ptrn):
    new_ptrn = []
    for i in range(len(old_ptrn)):
        ch = old_ptrn[i][0]
        if match_L4(WORDP, ch):
            ch = WORDP
        if len(new_ptrn) > 0:
            cur_class = new_ptrn[len(new_ptrn) - 1][0];
            if match_L4(cur_class, ch):
                new_ptrn[len(new_ptrn) - 1] = ch + "+"
            else:
                if len(old_ptrn[i]) > 1:
                    new_ptrn.append(ch + "+")
                else:
                    new_ptrn.append(ch)
        else:
            if len(old_ptrn[i]) > 1:
                new_ptrn.append(ch + "+")
            else:
                new_ptrn.append(ch)
    return new_ptrn


def match_L4(cch, ch):
    if cch == WORDP and ch == SQUOTE:
        return True
    if cch == WORDP and ch == SPACE:
        return True
    if cch == WORDP and ch == DASH:
        return True
    if cch == WORDP and ch == COMMA:
        return True
    if cch == WORDP and ch == DOT:
        return True
    if cch == WORDP and ch == ALPHA:
        return True
    if cch == ch:
        return True
    return False


def aggregate_UL_classes(old_ptrn):
    new_ptrn = []
    for i in range(len(old_ptrn)):
        ch = old_ptrn[i][0]
        if match_L3(ALPHA, ch):
            ch = ALPHA
        if len(new_ptrn) > 0:
            cur_class = new_ptrn[len(new_ptrn) - 1][0];
            if match_L3(cur_class, ch):
                new_ptrn[len(new_ptrn) - 1] = ch + "+"
            else:
                if len(old_ptrn[i]) > 1:
                    new_ptrn.append(ch + "+")
                else:
                    new_ptrn.append(ch)
        else:
            if len(old_ptrn[i]) > 1:
                new_ptrn.append(ch + "+")
            else:
                new_ptrn.append(ch)
    return new_ptrn


def match_L3(cch, ch):
    if cch == ALPHA and ch == LOWER:
        return True
    if cch == ALPHA and ch == UPPER:
        return True
    if cch == ALPHA and ch == USCR:
        return True
    if cch == ch:
        return True
    return False


def remove_numbers(old_ptrn):
    new_ptrn = []
    for i in range(len(old_ptrn)):
        if len(old_ptrn[i]) > 1:
            new_ptrn.append(old_ptrn[i][0] + "+")
        else:
            new_ptrn.append(old_ptrn[i])
    return new_ptrn


def L1_patterns(col_hist):
    pttrns_hist = dict()
    pttrns = []
    # given a column, for each row
    for k, v in col_hist.items():
        pttrn = L1_pattern(k)
        ptrn = PatterntoString(pttrn)
        k = pttrns_hist.get(ptrn, 0)
        pttrns_hist[ptrn] = k + 1
        if k == 0:
            pttrns.append(pttrn)
    return pttrns, pttrns_hist


# from [a1,b2] to "a1b2"
def PatterntoString(pttrn):
    ptrn = ""
    for i in range(len(pttrn)):
        ptrn = ptrn + pttrn[i]
    return ptrn


def L1_pattern(k):
    if not isinstance(k, str):
        print("Error reading non string!\n")
        sys.exit(1)
    pttrn = []
    # class and number of previous char
    prev_c = None
    prev_n = None
    for i in range(len(k)):
        ch = get_char_class(k[i])
        if (ch == prev_c):
            if (ch == SPACE):
                continue
            prev_n = prev_n + 1
            pttrn.pop()
            pttrn.append(ch + str(prev_n))
        else:
            pttrn.append(ch)
            prev_n = 1
        prev_c = ch
    pttrn = remove_enclosing(pttrn)
    return pttrn


def remove_enclosing(pttrn):
    new_pttrn = []
    if (check_enclosing(pttrn)):
        for i in range(len(pttrn)):
            if (pttrn[i][0] != ENCLOSE):
                new_pttrn.append(pttrn[i])
        return new_pttrn
    return pttrn


def check_enclosing(pttrn):
    count = 0
    for i in range(len(pttrn)):
        if (pttrn[i][0] == ENCLOSE):
            count = count + 1
    # check even or odd
    if count & 1 and count > 0:
        return False
    else:
        return True


def get_char_class(ch):
    if (ch.isdigit()):
        return DIGIT
    if (ch.islower()):
        return LOWER
    if (ch.isupper()):
        return UPPER
    # if (Special_Alphabet(ch)):
    #     return SPALPHA
    if ((ch == ':') or (ch == '?')):
        return PUNCT
    if ((ch == ' ') or (ch == '\t')):
        return SPACE
    if ((ch == '(') or (ch == '{') or (ch == '[')):
        return ENCLOSE
    if ((ch == ')') or (ch == '}') or (ch == ']')):
        return ENCLOSE
    if (ch == '-'):
        return DASH
    if (ch == '.'):
        return DOT
    if (ch == ','):
        return COMMA
    if (ch == '&'):
        return AND
    if (ch == '#'):
        return HASH
    if (ch == '@'):
        return AT
    if (ch == '%'):
        return PERCENT
    if (ch == '^'):
        return POWER
    if (ch == '*'):
        return ASTRSK
    if (ch == '!'):
        return NOT
    if (ch == '\''):
        return SQUOTE
    if (ch == '_'):
        return USCR
    if (ch == ';'):
        return PUNCT
    if (ch == '/'):
        return SLASH
    return SYMBOL


def Special_Alphabet(ch):
    cch = ord(ch)
    if ((cch >= 1) and (cch <= 8)):
        return true
    if ((cch >= 11) and (cch <= 26)):
        return true
    if ((cch >= 192) and (cch <= 214)):
        return true
    if ((cch >= 217) and (cch <= 246)):
        return true
    if ((cch >= 249) and (cch <= 254)):
        return true
    return false


LOWER = 'l'
UPPER = 'u'
DIGIT = 'd'
SPACE = 's'
ALPHA = 'a'
ALNUM = 'x'
DASH = 'h'
DOT = 't'
COMMA = 'c'
SYMBOL = 'y'
PUNCT = 'p'
ENCLOSE = 'e'
WORDP = 'w'
AND = '&'
HASH = '#'
AT = '@'
PERCENT = '%'
POWER = '^'
ASTRSK = '*'
NOT = '!'
SQUOTE = 'q'
USCR = '_'
SLASH = '/'
SPALPHA = 'v'
