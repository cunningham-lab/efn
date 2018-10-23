# coding: utf-8

from lib.tf_util.Bron_Kerbosch.bronker_bosch1 import bronker_bosch1
from lib.tf_util.Bron_Kerbosch.bronker_bosch2 import bronker_bosch2
from lib.tf_util.Bron_Kerbosch.bronker_bosch3 import bronker_bosch3
from lib.tf_util.Bron_Kerbosch.data import *
from lib.tf_util.Bron_Kerbosch.reporter import Reporter
 
 
if __name__ == '__main__':
    funcs = [bronker_bosch1,
             bronker_bosch2,
             bronker_bosch3]

    for func in funcs:
        print(func);
        report = Reporter('## %s' % 'thing')
        func([], set(NODES), set(), report)
        report.print_report()
