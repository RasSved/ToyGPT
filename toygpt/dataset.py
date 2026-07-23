
# Sliding windowd to create input and target pairs for next token predictions
def create_pairs(ids, context_size, stride):
    start = 0
    end = context_size
    tuple_lst = list()
    while end < len(ids):
        input = ids[start:end]
        target = ids[start+1:end+1]

        start += stride
        end += stride

        tuple_lst.append((input, target))
    return tuple_lst
 

test = create_pairs([1,2,3,4,5,6,7,8,9,10], context_size=2, stride=3)
print(test)