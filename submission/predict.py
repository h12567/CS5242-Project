# Import all data
import lightgbm as lgb
import numpy as np
import pandas as pd
import csv
import os
from struct import unpack

# https://marcin-chwedczuk.github.io/a-closer-look-at-portable-executable-msdos-stub
MS_DOS_HEADER_FORMAT = '<2b13h8bhh20bl'
# Assembly code to print out the stub
MS_DOS_STUB_FORMAT = '<{:d}b'
# Signature
SIGNATURE_FORMAT = '4s'
# COFF File Header
# [0] & [-1]: multi-class
COFF_FILE_HEADER_FORMAT = '<hhiiihh'
# image optional header
# [0] - 2 classes
# [23] & [24] - multi-class
IMAGE_OPTIONAL_HEADER_STANDARD_FORMAT = '<hbb5i'
MAX_LEN = 4096

def parse_hex(string):
    max_size = len(string)
    if max_size < 64:
        raise Exception('max_size error', max_size)

    ms_dos_header = unpack(MS_DOS_HEADER_FORMAT, string[:64])
    pe_sig_start = ms_dos_header[-1]
    pe_sig_end = pe_sig_start + 4

    ms_dos_stub = ()
    if pe_sig_start > 64:
        ms_dos_stub = unpack(MS_DOS_STUB_FORMAT.format(pe_sig_start - 64),
                             string[64:pe_sig_start])

    sig = unpack(SIGNATURE_FORMAT, string[pe_sig_start:pe_sig_end])

    coff_start = pe_sig_end
    coff_end = pe_sig_end + 20
    coff = unpack(COFF_FILE_HEADER_FORMAT,
                 string[pe_sig_end:coff_end])

    optional_header_size = coff[-2]
    img_opt_hdr_start = coff_end
    img_opt_hdr_mid = coff_end + 24
    img_opt_hrd_end = coff_end + optional_header_size

    optional_header = ()
    if optional_header_size != 0 and max_size > img_opt_hrd_end:
        optional_header = unpack(IMAGE_OPTIONAL_HEADER_STANDARD_FORMAT,
                                 string[img_opt_hdr_start:img_opt_hdr_mid])
        pe_format = optional_header[0]

        # Optional Header Windows-Specific Fields (Image Only)
        if pe_format == 267:
            # format is PE32
            optional_header += unpack('<iiii6h4ihh4iii16q',
                                 string[img_opt_hdr_mid:img_opt_hrd_end])

        elif pe_format == 523:
            # format is PE32+
            optional_header += (np.nan, )
            optional_header += unpack('<qii6h4ihh4qii16q',
                                     string[img_opt_hdr_mid:img_opt_hrd_end])
        else:
            raise Exception('pe_format error', pe_format)
    else:
        raise Exception('coff error', coff)

    number_of_sections = coff[1]
    section_start = img_opt_hrd_end
    section_end = img_opt_hrd_end + 40
    sections = []
    if max_size > section_end:
        for sec in range(number_of_sections):
            sections += [unpack('8s6ihhi', string[section_start:section_end])]
            section_start = section_end
            section_end += 40

    section_data = [max_size, max_size > MAX_LEN] + list(string[section_end:MAX_LEN])

    return ms_dos_header, ms_dos_stub, sig, coff, optional_header, sections, section_data

section_map = {}
def normalize_pe(parsed):
    global section_map
    max_headers = 26
    N = 4400
    ms_dos_header, ms_dos_stub, sig, coff, optional_header, sections, section_data = parsed

    output_sections = []
    for s in sections[:max_headers]:
        key = s[0]
        if not key in section_map:
            section_map[key] = len(section_map)
        output_sections += [section_map[key]] + list(s[1:])
    for i in range(len(sections), max_headers):
        # pad with null
        output_sections += [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    output_sections += [len(sections) > max_headers,]

    ms_dos_hash = [0 for i in range(0,256)]
    for x in ms_dos_stub:
        ms_dos_hash[x] += 1

    output = list(ms_dos_header) + ms_dos_hash + list(coff) + list(optional_header) + output_sections + section_data
    output += [0] * (N - len(output))
    return output


# Load model
bst = lgb.Booster(model_file="./model.txt")

test = []
indexes = []

testPath = next(p for p in ['./data/test.csv', './test.csv'] if os.path.isfile(p))
with open(testPath) as combinedcsvfile:
    combinedreader = csv.reader(combinedcsvfile)
    for i, row in enumerate(combinedreader):
        try:
            parsed = parse_hex(bytes([int(x) for x in row]))
            out = normalize_pe(parsed)
            test += [out]
            indexes += [1]
        except Exception as inst:
            print("Failed to parse")
            print(inst.args)
            indexes += [0]

ypred = bst.predict(test, num_iteration=bst.best_iteration)

counter = 0
with open('./predict.csv', 'w', newline='') as submitcsvfile:
    submitwriter = csv.writer(submitcsvfile)
    submitwriter.writerow(["sample_id", "malware"])
    for index, val in enumerate(indexes):
        if val == 0:
            submitwriter.writerow([index, 0.875])
        else:
            submitwriter.writerow([index, ypred[counter]])
            counter += 1
print('Written to ./predict.csv')