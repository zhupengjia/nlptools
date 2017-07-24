#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# name:      split_sentence.py
# author:    Yao zhiqiang <yaozhq.sh@gmail.com>
# license:   GPL
# created:   2017 Mar 6
# modified:  2017 Jul 22
#

from itertools import groupby
from operator import itemgetter
from acora import AcoraBuilder

class SearchKeyword:
    def __init__(self, keywords):
        builder = AcoraBuilder()
        #assert isinstance(keywords, (list,tuple))
        for i in keywords:
            builder.add(i)

        #Generate the Acora search engine for the current keyword set:
        self.engine = builder.build()

    def find(self, input):
        return self.engine.findall(input)

    def find_longest(self, input):
        def longest_match(matches):
            for pos, match_set in groupby(matches, itemgetter(1)):
                yield max(match_set)

        return longest_match(self.engine.findall(input))

    def find_max_match(self, input):
        def subset(a, b):
            if (a[1] >= b[1]) and ((a[1] + len(a[0])) <= (b[1] + len(b[0]))):
                return True
            else:
                return False

        def max_match(matches):
            if len(matches) <= 1:
                return matches

            maxmatch = []
            for i in matches:
                for j in matches:
                    if i == j:
                        continue
                    elif subset(i, j):  # or subset(j,i):
                        break
                else:
                    maxmatch.append(i)
            return maxmatch

        return max_match(self.engine.findall(input))


if __name__ == '__main__':
    from acora import AcoraBuilder
    bc = SearchKeyword(['死亡','death'])
    bc2 = SearchKeyword(['Vaccination site pruritus','staphylococcus aureus','cataract'])
    for i in bc.find(
            'cataract,からstaphylococcus aureus同定,尿培養検査よりklebsiella pneumoniae staphylococcus aureus 同定\
death高令の患者、Vaccination site pruritus故に嚥下能力が徐々に低下し、その他、内服全般が難しくなってきた為内服を中止した。その後、少しずつ全身状態の悪化がすすみ、死亡に至った。よって、ネキシウムカプセルとの直接的な因果関係はないと考えられる。'):
        print(i[0], i[1])
    print('-'*100)
    for i in bc2.find(
                'cataract,からstaphylococcus aureus同定,尿培養検査よりklebsiella pneumoniae staphylococcus aureus 同定\
    death高令の患者、Vaccination site pruritus故に嚥下能力が徐々に低下し、その他、内服全般が難しくなってきた為内服を中止した。その後、少しずつ全身状態の悪化がすすみ、死亡に至った。よって、ネキシウムカプセルとの直接的な因果関係はないと考えられる。'):
            print(i[0], i[1])
