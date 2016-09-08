def sortwithloops(input_list):
    input1 = input_list
    sorted_list = []
    lenny = len(input1)
    i = lenny
    while len(input1) > 0:
        i = i - 1
        i0 = input1[i]
        for j in range(0, len(input1)):
            if input1[j] < i0:
                i0 = input1[j]
                lowindex = j
        sorted_list.append(i0)
        input1 = input1[:lowindex] + input1[lowindex+1 :]
    return sorted_list #return a value

def sortwithoutloops(input1):
    input1.sort()
    return input1

def searchwithloops(input, value):
    flag = ''
    for i in input:
        if i == value:
            flag = True
    if flag:
        return True
    else:
        return False

def searchwithoutloops(input, value):
    if value in input:
        return True
    else:
        return False

if __name__ == '__main__':
    L = [5,3,6,3,13,5,6]
    #print "Sort With Loops"
    print sortwithloops(L) # [3, 3, 5, 5, 6, 6, 13]
    print sortwithoutloops(L) # [3, 3, 5, 5, 6, 6, 13]
    print searchwithloops(L, 5)
    print searchwithloops(L, 11)
    print searchwithoutloops(L, 5)
    print searchwithloops(L, 11)
