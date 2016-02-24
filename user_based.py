from math import sqrt

def read_data(filepath):
    '''
    read data in the format of
    user id | item id | rating | timestamp
    '''
    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split('\t')
            user = int(words[0])
            item = int(words[1])
            rating = float(words[2])
            if user not in data:
                data[user] = {}
            data[user][item] = rating
    return data


def get_users(data):
    return data.keys()


def get_user_by_item(data):
    res = {}
    for user in data:
        for item in data[user]:
            if item not in res:
                res[item] = []
            res[item].append(user)

    return res


def get_mean_score(data):
    mean_score = 0.
    mean_count = 0.
    for user in data:
        for item in data[user]:
            mean_score += data[user][item]
            mean_count += 1
    return mean_score / mean_count


USER_SIMILARITY_CACHE = {}
def user_similarity_pearson(data, user1, user2, mean_score=3):
    '''
    compute the similarity between two users based on pearson coef.
    '''
    if (user1, user2) in USER_SIMILARITY_CACHE:
        return USER_SIMILARITY_CACHE[(user1, user2)]

    if (user2, user1) in USER_SIMILARITY_CACHE:
        return USER_SIMILARITY_CACHE[(user2, user1)]

    if user1 not in data or user2 not in data:
        return 0.

    s1 = []
    for item, val in data[user1].items():
        if item not in data[user2]:
            continue
        s1.append(item)

    norm1 = sqrt(sum([(data[user1][item] - mean_score) ** 2 for item in s1]))
    norm2 = sqrt(sum([(data[user2][item] - mean_score) ** 2 for item in s1]))

    product = sum([(data[user1][item] - mean_score) * (data[user2][item] - mean_score) for item in s1])

    if norm1 * norm2 == 0.: # means users do not have items in common.
        return 0.
    else:
        coef = product / (norm1 * norm2)

        USER_SIMILARITY_CACHE[(user1, user2)] = coef
        return coef


def evaluate_pred(pred_dict, data):
    mse = 0.
    mse_count = 0.
    for user in data:
        for item in data[user]:
            mse += (pred_dict[user][item] - data[user][item]) ** 2
            mse_count += 1

    return sqrt(mse / mse_count)


if __name__ == '__main__':
    train_data = read_data('data/ml-100k/u1.base')
    user_by_item = get_user_by_item(train_data)
    test_data = read_data('data/ml-100k/u1.test')

    pred_dict = {}
    pred_baseline = {}
    mean_score = get_mean_score(train_data)

    for user in test_data:
        pred_dict[user] = {}
        pred_baseline[user] = {}
        for item in test_data[user]:
            pred_baseline[user][item] = mean_score
            train_users = user_by_item.get(item)
            if not train_users:
                pred_dict[user][item] = mean_score
                continue

            total_score = 0.
            total_weight = 0.
            for sim_user in train_users:
                weight = user_similarity_pearson(train_data, user,
                                                 sim_user, mean_score)
                score = weight * train_data[sim_user][item]
                total_score += score
                total_weight += weight # intersting finding: using abs(weight) leads to poor performance.

            pred = total_score / total_weight

            # normalize.
            if pred < 1.:
                pred = 1.
            elif pred > 5.:
                pred = 5.
            else:
                pred = round(pred)

            pred_dict[user][item] = pred

    print '----------------------------------------'
    print 'baseline by guessing mean', evaluate_pred(pred_baseline, test_data)
    print 'mean squared error = ', evaluate_pred(pred_dict, test_data)


