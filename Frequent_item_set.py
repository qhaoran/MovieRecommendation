import os
import sys
import itertools
from pyspark import SparkContext

case = sys.argv[1]
filepath = sys.argv[2]
supportval = sys.argv[3]
support = int(supportval)

outputFileName = ""
if support > 1000:
    outputFileName = "Haoran_Que_SON_MovieLens.Big.case" + str(case) + "-" + str(support) + ".txt"
elif support < 20:
    outputFileName = "Haoran_Que_SON_small2.case" + str(case) + ".txt"
else:
    outputFileName = "Haoran_Que_SON_MovieLens.Small.case" + str(case) + "-" + str(support) + ".txt"

print outputFileName

sc = SparkContext('local')
if int(case) == 1:
    basket = sc.textFile(filepath).map(lambda line: line.split(",")).filter(lambda line: not line[0].isalpha()).map(
        lambda line: (int(line[0]), int(line[1]))).groupByKey()
else:
    basket = sc.textFile(filepath).map(lambda line: line.split(",")).filter(lambda line: not line[0].isalpha()).map(
        lambda line: (int(line[1]), int(line[0]))).groupByKey()

length_each = basket.glom().map(len).collect()
print("size for each partition: %s" % length_each)

length_total = sum(length_each)
print("size for all partitions: %s" % length_total)


# Impletement A-Priori Algorithm
def firstmap(iterator):
    baskets = []
    bsklen_before = 0
    for i in iterator:
        baskets.append(list(set(list(i[1]))))
        bsklen_before += len(i[1])
    count = {}
    print("size of this partition is : %s" % len(baskets))
    supportPartition = int(support * (float(len(baskets)) / length_total))
    print("support within partition: %s" % supportPartition)

    # genarate frequent singletons
    for i in baskets:
        for element in i:
            if count.has_key(element):
                val = count[element]
                count[element] = val + 1
            else:
                count[element] = 1
    freitem = []
    for (k, v) in count.items():
        if v >= supportPartition:
            yield (k, 1)
            freitem.append(k)

    # delete non-frequent single
    for bsk in baskets:
        bsk =set(bsk).intersection(freitem)

    freitem = sorted(freitem)
    print("item set of size 1 in this partition has: %s items" % len(freitem))

    # generate pairs
    candidate = list(itertools.combinations(set(freitem), 2))
    fre_pair = []
    fre_pair_element_set = set([])
    for cnd in candidate:
        flag = 0
        cur_basket_index = 1
        cndSet = set(cnd)
        for bsk in baskets:
            if cndSet.issubset(set(bsk)):
                flag += 1
            if supportPartition - flag > len(baskets) - cur_basket_index:
                break
            if flag >= supportPartition:
                yield (tuple(sorted(cnd)), 1)
                fre_pair.append(tuple(sorted(cnd)))
                fre_pair_element_set = fre_pair_element_set.union(cnd)
                break
            cur_basket_index += 1
    print("item set of size 2 in this partition has: %s items" % len(fre_pair))

    # delete non-frequent items in set 2
    for bsk in baskets:
        bsk = set(bsk).intersection(fre_pair_element_set)

    last_ItemsetSize = 2
    last_set = fre_pair
    last_latentItems = fre_pair_element_set

    while (len(last_latentItems) != 0):
        candidate = list(itertools.combinations(sorted(last_latentItems), last_ItemsetSize + 1))

        # count for each candidate:
        cntForsize = 0
        IDset = set([])
        temp_ItemSet = []
        for cnd in candidate:
            flag = 0
            checkmark = True
            componentOfcandidate = list(itertools.combinations(cnd, last_ItemsetSize))
            if not set(componentOfcandidate).issubset(set(last_set)):
                checkmark = False
            if checkmark:
                cur_basket_index = 1
                cndSet = set(cnd)
                for bsk in baskets:
                    bsk = set(bsk).intersection(last_latentItems)
                    if cndSet.issubset(set(bsk)):
                        flag += 1
                    if supportPartition - flag > len(baskets) - cur_basket_index:
                        break
                    if flag >= supportPartition:
                        yield (tuple(sorted(cnd)), 1)
                        IDset = IDset.union(cnd)
                        temp_ItemSet.append(tuple(sorted(cnd)))
                        cntForsize += 1
                        break
                    cur_basket_index += 1
        print("item set of size %s in this partition has: %s items" % (last_ItemsetSize + 1, cntForsize))

        last_ItemsetSize = last_ItemsetSize + 1
        last_latentItems = IDset
        last_set = temp_ItemSet

mapreduce1 = basket.mapPartitions(firstmap).collect()

def secndMap(iterator):
    mapreduce2 = list(set(mapreduce1))
    for basket in iterator:
        for candidate in mapreduce2:
            if isinstance(candidate[0], int):
                if candidate[0] in list(basket[1]):
                    yield (candidate[0], 1)
            else:
                if set(list(candidate[0])).issubset(set(list(basket[1]))):
                    yield (candidate[0], 1)

def secndReduce(iterator):
    for i in iterator:
        if i[1] >= support:
            yield i[0]

Fre_item_set = basket.mapPartitions(secndMap).reduceByKey(lambda x, y: x + y).mapPartitions(secndReduce).collect()


f = open(outputFileName, "w")
divideSet = {}
singlenton = []

for item in Fre_item_set:
    if isinstance(item, int):
        singlenton.append(item)
    else:
        if divideSet.has_key(len(item)):
            templist = divideSet[len(item)]
            templist.append(item)
            divideSet[len(item)] = templist
        else:
            divideSet[len(item)] = [item]
for key in divideSet:
    divideSet[key] = sorted(divideSet[key])
singlenton = sorted(singlenton)
print("\n")
print("# of items in fre item size of 1 set: %s" % len(singlenton))
temp_str = ""
for i in singlenton:
    temp_str += "(%s), " % i
f.write(temp_str[:-2])
f.write("\n\n")
keyset = sorted(divideSet.keys())

for i in keyset:
    itemset = sorted(divideSet[i])
    print("# of items in fre item size of %s set: %s" % (i, len(itemset)))
    tem_str2 = ""
    for a in itemset:
        tem_str2 += "%s, " % str(a)
    s = tem_str2[:-2]
    f.write(s)
    f.write("\n\n")
f.close()
