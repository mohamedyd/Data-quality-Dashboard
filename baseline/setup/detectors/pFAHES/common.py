import pandas as pd
import ntpath
import csv
import json


def read_table(tab_name):
    t_name = ntpath.basename(tab_name)
    try:
        df = pd.read_csv(filepath_or_buffer=tab_name, dtype=object, delimiter=',', low_memory=False,
                         quoting=csv.QUOTE_ALL, doublequote=True)
    except ValueError:
        try:
            df = pd.read_csv(filepath_or_buffer=tab_name, dtype=object, delimiter=',', low_memory=False,
                             quoting=csv.QUOTE_ALL, doublequote=True, encoding="ISO-8859-1")
        except:
            print("Error reading csv file .. file encoding is not recognizable")
            return None
    return df


def check_d_quotation(str):
    if "," in str:
        return "\"" + str + "\""
    return str


def print_output_data(out_directory, table_name, sus_dis_values):
    if out_directory[-1] != '/':
        out_directory = out_directory + '/'
    t = table_name.replace('\\', '/')
    tabn = t.split('/')[-1]
    f = open(out_directory + "DMV_" + tabn, "w")
    if len(sus_dis_values) < 1:
        print("nothing")
        return
    f.write("Table Name,Attribute Name,DMV,Frequency,Detecting Tool\n")
    for sus_dis in sus_dis_values:
        f.write(check_d_quotation(tabn[0:len(tabn) - 4]) + "," + check_d_quotation(
            sus_dis.attr_name) + "," + check_d_quotation(sus_dis.value) + "," + str(
            sus_dis.frequency) + "," + sus_dis.tool_name + "\n")


def add_detected_by_more_than_one_tool(dmvs, dmv):
    for item in dmvs:
        if item.attr_name == dmv.attr_name and item.value == dmv.value:
            item.tool_name += '_' + dmv.tool_name


def print_output_data_json(out_directory, table_name, sus_dis_values, ptrns):
    if out_directory[-1] != '/':
        out_directory = out_directory + '/'
    t = table_name.replace('\\', '/')
    tabn = t.split('/')[-1]
    output_f_name = out_directory + "DMV_" + tabn
    json_output_f_name = output_f_name.replace(".csv", ".json")
    output_data = dict()
    output_data.clear()
    output_data['DMVs'] = []
    for ii in range(len(output_data['DMVs'])):
        output_data['DMVs'].remove(output_data['DMVs'][0])
    fp = open(json_output_f_name, "w")
    for sus_dis in sus_dis_values:
        output_data['DMVs'].append([sus_dis.attr_name, sus_dis.value, str(sus_dis.frequency), sus_dis.tool_name])
    output_data['patterns'] = ptrns
    if len(sus_dis_values) < 1:
        print("nothing")
        return
    json.dump(output_data, fp)
